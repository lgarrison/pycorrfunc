#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "function_precision.h"
#include "utils.h"

#include "weight_functions.h"
#include "kernelfuncs.h"


#ifdef HAVE_AVX512F
#include "avx512_calls.h"

int avx512f_available(void) {
    return __builtin_cpu_supports("avx512f");
}

int countpairs_avx512_intrinsics(const int64_t N0, DOUBLE *x0, DOUBLE *y0, DOUBLE *z0, const weight_struct *weights0,
                                                      const int64_t N1, DOUBLE *x1, DOUBLE *y1, DOUBLE *z1, const weight_struct *weights1,
                                                      const int same_cell,
                                                      const DOUBLE sqr_rpmax, const DOUBLE sqr_rpmin, const int nbin, const DOUBLE *rupp_sqr, const DOUBLE rpmax,
                                                      const DOUBLE off_xwrap, const DOUBLE off_ywrap, const DOUBLE off_zwrap,
                                                      const DOUBLE min_xdiff, const DOUBLE min_ydiff, const DOUBLE min_zdiff,
                                                      const DOUBLE closest_icell_xpos, const DOUBLE closest_icell_ypos, const DOUBLE closest_icell_zpos,
                                                      DOUBLE *src_rpavg, uint64_t *src_npairs,
                                                      DOUBLE *src_weighted_pairs, const weight_method_t weight_method)
{
    (void) sqr_rpmax, (void) sqr_rpmin;

#ifdef COUNT_VECTORIZED
    struct timespec tcell_start;
    current_utc_time(&tcell_start);
    uint64_t serial_npairs = 0, vectorized_npairs=0;
#endif

    uint64_t npairs[nbin];
    for(int i=0;i<nbin;i++) {
        npairs[i] = 0;
    }
    AVX512_FLOATS m_rupp_sqr[nbin];
    for(int i=0;i<nbin;i++) {
        m_rupp_sqr[i] = AVX512_SET_FLOAT(rupp_sqr[i]);
    }
    const int32_t need_rpavg = src_rpavg != NULL;
    const int32_t need_weighted_pairs = src_weighted_pairs != NULL;

    /* variables required for rpavg and weighted_pairs*/
    DOUBLE rpavg[nbin], weighted_pairs[nbin];
    if(need_rpavg || need_weighted_pairs){
        for(int i=0;i<nbin;i++) {
            rpavg[i] = ZERO;
            weighted_pairs[i] = ZERO;
        }
    }

    // A copy whose pointers we can advance
    weight_struct local_w0 = {.weights={NULL}, .num_weights=0}, local_w1 = {.weights={NULL}, .num_weights=0};
    pair_struct pair = {.num_weights=0};
    avx512_weight_func_t avx512_weight_func = NULL;
    if(need_weighted_pairs){
        // Same particle list, new copy of num_weights pointers into that list
        local_w0 = *weights0;
        local_w1 = *weights1;

        pair.num_weights = local_w0.num_weights;
        avx512_weight_func   = get_avx512_weight_func_by_method(weight_method);
    }

    const DOUBLE *zstart = z1, *zend = z1 + N1;
    const DOUBLE max_all_dz = SQRT(rpmax*rpmax - min_xdiff*min_xdiff - min_ydiff*min_ydiff);
    for(int64_t i=0;i<N0;i++) {
        const DOUBLE xpos = *x0++ + off_xwrap;
        const DOUBLE ypos = *y0++ + off_ywrap;
        const DOUBLE zpos = *z0++ + off_zwrap;
        for(int w = 0; w < pair.num_weights; w++){
            // local_w0.weights[w] is a pointer to a float in the particle list of weights,
            // just as x0 is a pointer into the list of x-positions.
            // The advancement of the local_w0.weights[w] pointer should always mirror x0.
            pair.weights0[w].a512 = AVX512_SET_FLOAT(*(local_w0.weights[w])++);
        }
        DOUBLE max_dz = max_all_dz;

        /* Now consider if this i'th particle can be a valid pair with ANY of the remaining
           j' particles. The min. difference in the z-positions between this i'th particle and ANY
           of the remaining j'th particles, is the difference between the current j'th particle and
           the current i'th particle (since all remaining j'th particles will have a larger value for
           the z-ordinate and therefore a larger difference to zpos). if this `dz` does not satisfy
           the distance criteria, then NO remaining j'th particles will. Continue on to the next i'th
           particle
        */
        const DOUBLE this_dz = *z1 - zpos;
        if(this_dz >= max_all_dz) {
            continue;
        }

        /* Okay so there MAY be a pair */
        if(same_cell == 1) {
            z1++;
        } else {
            // Now add the x,y information to further limit the range of secondaries for this particle
            // But note this constraint may increase or decrease for the next particle, since x,y aren't sorted!
            const DOUBLE min_dx = min_xdiff > 0 ? min_xdiff + FABS(xpos - closest_icell_xpos):min_xdiff;
            const DOUBLE min_dy = min_ydiff > 0 ? min_ydiff + FABS(ypos - closest_icell_ypos):min_ydiff;
            const DOUBLE min_dz = min_zdiff > 0 ? (this_dz > 0 ? this_dz:min_zdiff + FABS(zpos - closest_icell_zpos)):min_zdiff;
            const DOUBLE sqr_min_sep_this_point = min_dx*min_dx + min_dy*min_dy + min_dz*min_dz;
            if(sqr_min_sep_this_point >= sqr_rpmax) {
                continue;
            }
            max_dz = SQRT(sqr_rpmax - min_dx*min_dx - min_dy*min_dy);

            // Now "fast forward" in the list of secondary particles to find the first one that satisfies the max_all_dz constraint
            // We don't consider the i particle's x,y information yet, because those aren't sorted
            const DOUBLE target_z = zpos - max_all_dz;
            while(z1 < zend && *z1 <= target_z) {
                z1++;
            }
        }

        // If no j particle satisfies the constraint for this i particle,
        // then the same holds true for all future i particles because they are sorted in increasing z order
        if(z1 == zend) {
            i = N0;
            break;
        }

        DOUBLE *localz1 = z1;
        const DOUBLE target_z = zpos - max_dz;
        while(localz1 != zend && *localz1 <= target_z) {
            localz1++;
        }

        const int64_t n_off = localz1 - zstart;
        DOUBLE *localx1 = x1 + n_off;
        DOUBLE *localy1 = y1 + n_off;
        for(int w = 0; w < local_w1.num_weights; w++){
            local_w1.weights[w] = weights1->weights[w] + n_off;
        }
        const AVX512_FLOATS m_xpos  = AVX512_SET_FLOAT(xpos);
        const AVX512_FLOATS m_ypos  = AVX512_SET_FLOAT(ypos);
        const AVX512_FLOATS m_zpos  = AVX512_SET_FLOAT(zpos);

        for(int64_t j=n_off;j<N1;j+=AVX512_NVEC) {
            AVX512_MASK m_mask_left = (N1 - j) >= AVX512_NVEC ? ~0:masks_per_misalignment_value[N1-j];
            const AVX512_FLOATS m_x1 = AVX512_MASKZ_LOAD_FLOATS_UNALIGNED(m_mask_left, localx1);
            const AVX512_FLOATS m_y1 = AVX512_MASKZ_LOAD_FLOATS_UNALIGNED(m_mask_left, localy1);
            const AVX512_FLOATS m_z1 = AVX512_MASKZ_LOAD_FLOATS_UNALIGNED(m_mask_left, localz1);

            union int16 union_rpbin;
            union float16 union_mDperp;
            union float16_weights union_mweight;

            union_rpbin.m_ibin = AVX512_SETZERO_INT();
            union_mDperp.m_Dperp = AVX512_SETZERO_FLOAT();

#ifdef COUNT_VECTORIZED
            vectorized_npairs += (N1 - j) >= AVX512_NVEC ? AVX512_NVEC:N1-j;
#endif
            localx1 += AVX512_NVEC;//this might actually exceed the allocated range but we will never dereference that
            localy1 += AVX512_NVEC;
            localz1 += AVX512_NVEC;

            for(int w = 0; w < pair.num_weights; w++){
                pair.weights1[w].a512 = AVX512_MASKZ_LOAD_FLOATS_UNALIGNED(m_mask_left, local_w1.weights[w]);
                local_w1.weights[w] += AVX512_NVEC;
            }

            const AVX512_FLOATS m_max_dz = AVX512_SET_FLOAT(max_dz);
            const AVX512_FLOATS m_sqr_rpmax = m_rupp_sqr[nbin-1];
            const AVX512_FLOATS m_sqr_rpmin = m_rupp_sqr[0];

            const AVX512_FLOATS m_xdiff = AVX512_SUBTRACT_FLOATS(m_x1, m_xpos);  //(x[j] - x0)
            const AVX512_FLOATS m_ydiff = AVX512_SUBTRACT_FLOATS(m_y1, m_ypos);  //(y[j] - y0)
            const AVX512_FLOATS m_zdiff = AVX512_SUBTRACT_FLOATS(m_z1, m_zpos);  //z2[j:j+NVEC-1] - z1

            if(need_weighted_pairs){
                pair.dx.a512 = m_xdiff;
                pair.dy.a512 = m_ydiff;
                pair.dz.a512 = m_zdiff;
            }

            const AVX512_FLOATS m_sqr_xdiff = AVX512_SQUARE_FLOAT(m_xdiff);  //(x0 - x[j])^2
            const AVX512_FLOATS x2py2  = AVX512_FMA_ADD_FLOATS(m_ydiff, m_ydiff, m_sqr_xdiff);/* dy*dy + dx^2*/
            const AVX512_FLOATS r2 = AVX512_FMA_ADD_FLOATS(m_zdiff, m_zdiff, x2py2);/* dz*dz + (dy^2 + dx^2)*/

            //the z2 arrays are sorted in increasing order. which means
            //the z2 value will increase in any future iteration of j.
            //that implies the zdiff values are also monotonically increasing
            //Therefore, if any of the zdiff values are >= pimax, then
            //no future iteration in j can produce a zdiff value less than pimax.
            AVX512_MASK m_mask_geq_pimax = AVX512_MASK_COMPARE_FLOATS(m_mask_left, m_zdiff,m_max_dz,_CMP_GE_OQ);
            if(m_mask_geq_pimax > 0) {
                //Some the dz values are >= pimax
                // => no pairs can be found in any future iterations
                // but do not break yet --> need to process this chunk
                j=N1;
            }

            const AVX512_MASK m_rpmax_mask = AVX512_MASK_COMPARE_FLOATS(m_mask_left, r2, m_sqr_rpmax, _CMP_LT_OQ);
            //Create a combined mask
            //This gives us the mask for all sqr_rpmin <= r2 < sqr_rpmax
            m_mask_left = AVX512_MASK_COMPARE_FLOATS(m_rpmax_mask, r2, m_sqr_rpmin, _CMP_GE_OQ);
            if(m_mask_left == 0) {
                continue;
            }

            if(need_rpavg) {
                union_mDperp.m_Dperp = AVX512_MASKZ_SQRT_FLOAT(m_mask_left, r2);
            }
            if(need_weighted_pairs){
                union_mweight.m_weights = avx512_weight_func(&pair);
            }

            AVX512_INTS m_rpbin = AVX512_SETZERO_INT();
            //Loop backwards through nbins. m_mask_left contains all the points that are less than rpmax
            // at the beginning of the loop.
            for(int kbin=nbin-1;kbin>=1;kbin--) {
                const AVX512_MASK m_bin_mask = AVX512_MASK_COMPARE_FLOATS(m_mask_left, r2,m_rupp_sqr[kbin-1],_CMP_GE_OS);
                npairs[kbin] += bits_set_in_avx512_mask[m_bin_mask];
                if(need_rpavg || need_weighted_pairs) {
                    m_rpbin = AVX512_BLEND_INTS_WITH_MASK(m_bin_mask, m_rpbin, AVX512_SET_INT(kbin));
                }
                m_mask_left = AVX512_MASK_BITWISE_AND_NOT(m_bin_mask, m_mask_left);//ANDNOT(X, Y) -> NOT X AND Y
                if(m_mask_left == 0) {
                    break;
                }
            }//backwards loop over the bins

            if(need_rpavg || need_weighted_pairs) {
                //Do I need this step of going via the union? accessing int[] -> AVX* vector might
                //cause alignment problems but accessing the ints from an AVX*
                //register should always be fine
                union_rpbin.m_ibin = m_rpbin;
                //protect the unroll pragma in case compiler is not icc.
#if  __INTEL_COMPILER
#pragma unroll(AVX512_NVEC)
#endif
                for(int jj=0;jj<AVX512_NVEC;jj++) {
                    const int kbin = union_rpbin.ibin[jj];
                    if(need_rpavg){
                        const DOUBLE r = union_mDperp.Dperp[jj];
                        rpavg[kbin] += r;
                    }
                    if(need_weighted_pairs){
                        const DOUBLE weight = union_mweight.weights[jj];
                        weighted_pairs[kbin] += weight;
                    }
                }
            } //OUTPUT_RPAVG
        }//end of j-loop
    }//loop over first set of particles

    uint64_t npairs_found = 0;
    for(int i=1;i<nbin;i++) {
        npairs_found += npairs[i];
        src_npairs[i - 1] += npairs[i];
        if(need_rpavg) {
            src_rpavg[i - 1]  += rpavg[i];
        }
        if(need_weighted_pairs) {
            src_weighted_pairs[i - 1] += weighted_pairs[i];
        }
    }

#ifdef COUNT_VECTORIZED
    struct timespec tcell_end;
    current_utc_time(&tcell_end);
    int64_t dt = (int64_t) REALTIME_ELAPSED_NS(tcell_start,tcell_end);
    fprintf(stderr,"%5"PRId64" %5"PRId64" %12"PRId64" %12"PRIu64" %12"PRIu64" %12"PRIu64" %2d\n", N0, N1, dt, vectorized_npairs, serial_npairs, npairs_found, same_cell);
#endif

    return EXIT_SUCCESS;
}
#endif //__AVX512F__
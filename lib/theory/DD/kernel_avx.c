#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "function_precision.h"
#include "utils.h"

#include "weights.h"
#include "kernelfuncs.h"


#ifdef HAVE_AVX
#include "avx_calls.h"

#ifdef _MSC_VER
#include<intrin.h>
int avx_available(void) {
  return 1;
}
#else
#include <immintrin.h>
#include <cpuid.h>

#ifdef __APPLE__
/*
 * Apple ships a broken __builtin_cpu_supports and
 * some machines in the CI farm seem to be too
 * old to have AVX so just always return 0 here.
 */
int avx_available(void) { return 0; }
#else

int avx_available(void) {
    return __builtin_cpu_supports("avx");
}
#endif
#endif

int countpairs_avx_intrinsics(const int64_t N0, DOUBLE *x0, DOUBLE *y0, DOUBLE *z0, DOUBLE *w0,
                                                   const int64_t N1, DOUBLE *x1, DOUBLE *y1, DOUBLE *z1, DOUBLE *w1,
                                                   const int same_cell,
                                                   const DOUBLE sqr_rmax, const DOUBLE sqr_rmin, const int nbin, const DOUBLE *rupp_sqr, const DOUBLE rmax,
                                                   const DOUBLE off_xwrap, const DOUBLE off_ywrap, const DOUBLE off_zwrap,
                                                   const DOUBLE min_xdiff, const DOUBLE min_ydiff, const DOUBLE min_zdiff,
                                                   const DOUBLE closest_icell_xpos, const DOUBLE closest_icell_ypos, const DOUBLE closest_icell_zpos,
                                                   DOUBLE *src_ravg, uint64_t *src_npairs,
                                                   DOUBLE *src_weighted_pairs, const weight_method_t weight_method)
{
    const int32_t need_ravg = src_ravg != NULL;
    const int32_t need_weighted_pairs = src_weighted_pairs != NULL;

    uint64_t npairs[nbin];
    for(int i=0;i<nbin;i++) {
        npairs[i] = 0;
    }

    AVX_FLOATS m_rupp_sqr[nbin];
    for(int i=0;i<nbin;i++) {
        m_rupp_sqr[i] = AVX_SET_FLOAT(rupp_sqr[i]);
    }

    /* variables required for ravg and weighted_pairs*/
    AVX_FLOATS m_kbin[nbin];
    DOUBLE ravg[nbin], weighted_pairs[nbin];
    if(need_ravg || need_weighted_pairs){
        for(int i=0;i<nbin;i++) {
            m_kbin[i] = AVX_SET_FLOAT((DOUBLE) i);
            if(need_ravg){
                ravg[i] = ZERO;
            }
            if(need_weighted_pairs){
                weighted_pairs[i] = ZERO;
            }
        }
    }

    // A copy whose pointers we can advance
    weight_struct local_w0 = {.weights={NULL}, .num_weights=0}, local_w1 = {.weights={NULL}, .num_weights=0};
    pair_struct pair = {.num_weights=0};
    avx_weight_func_t avx_weight_func = NULL;
    weight_func_t fallback_weight_func = NULL;
    if(need_weighted_pairs){
        // Same particle list, new copy of num_weights pointers into that list
        local_w0 = *weights0;
        local_w1 = *weights1;

        pair.num_weights = local_w0.num_weights;

        avx_weight_func = get_avx_weight_func_by_method(weight_method);
        fallback_weight_func = get_weight_func_by_method(weight_method);
    }

    const DOUBLE *zstart = z1, *zend = z1 + N1;
    const DOUBLE max_all_dz = SQRT(rmax*rmax - min_xdiff*min_xdiff - min_ydiff*min_ydiff);
    for(int64_t i=0;i<N0;i++) {
        const DOUBLE xpos = *x0++ + off_xwrap;
        const DOUBLE ypos = *y0++ + off_ywrap;
        const DOUBLE zpos = *z0++ + off_zwrap;
        for(int w = 0; w < pair.num_weights; w++){
            // local_w0.weights[w] is a pointer to a float in the particle list of weights,
            // just as x0 is a pointer into the list of x-positions.
            // The advancement of the local_w0.weights[w] pointer should always mirror x0.
            pair.weights0[w].a = AVX_SET_FLOAT(*(local_w0.weights[w])++);
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
            if(sqr_min_sep_this_point >= sqr_rmax) {
                continue;
            }
            max_dz = SQRT(sqr_rmax - min_dx*min_dx - min_dy*min_dy);

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

        int64_t j = localz1 - zstart;
        DOUBLE *localx1 = x1 + j;
        DOUBLE *localy1 = y1 + j;
        for(int w = 0; w < local_w1.num_weights; w++){
            local_w1.weights[w] = weights1->weights[w] + j;
        }

        for(;j<=(N1 - AVX_NVEC);j+=AVX_NVEC) {
            const AVX_FLOATS m_xpos    = AVX_SET_FLOAT(xpos);
            const AVX_FLOATS m_ypos    = AVX_SET_FLOAT(ypos);
            const AVX_FLOATS m_zpos    = AVX_SET_FLOAT(zpos);

            union int8 union_rbin;
            union float8 union_mDperp;
            union float8_weights union_mweight;

            const AVX_FLOATS m_x1 = AVX_LOAD_FLOATS_UNALIGNED(localx1);
            const AVX_FLOATS m_y1 = AVX_LOAD_FLOATS_UNALIGNED(localy1);
            const AVX_FLOATS m_z1 = AVX_LOAD_FLOATS_UNALIGNED(localz1);

            localx1 += AVX_NVEC;//this might actually exceed the allocated range but we will never dereference that
            localy1 += AVX_NVEC;
            localz1 += AVX_NVEC;

            for(int w = 0; w < pair.num_weights; w++){
                pair.weights1[w].a = AVX_LOAD_FLOATS_UNALIGNED(local_w1.weights[w]);
                local_w1.weights[w] += AVX_NVEC;
            }

            const AVX_FLOATS m_max_dz = AVX_SET_FLOAT(max_dz);
            const AVX_FLOATS m_sqr_rmax = m_rupp_sqr[nbin-1];
            const AVX_FLOATS m_sqr_rmin = m_rupp_sqr[0];

            const AVX_FLOATS m_xdiff = AVX_SUBTRACT_FLOATS(m_x1, m_xpos);  //(x[j] - x0)
            const AVX_FLOATS m_ydiff = AVX_SUBTRACT_FLOATS(m_y1, m_ypos);  //(y[j] - y0)
            const AVX_FLOATS m_zdiff = AVX_SUBTRACT_FLOATS(m_z1, m_zpos);  //z2[j:j+NVEC-1] - z1

            if(need_weighted_pairs){
                pair.dx.a = m_xdiff;
                pair.dy.a = m_ydiff;
                pair.dz.a = m_zdiff;
            }

            const AVX_FLOATS m_sqr_xdiff = AVX_SQUARE_FLOAT(m_xdiff);  //(x0 - x[j])^2
            const AVX_FLOATS m_sqr_ydiff = AVX_SQUARE_FLOAT(m_ydiff);  //(y0 - y[j])^2
            const AVX_FLOATS m_sqr_zdiff = AVX_SQUARE_FLOAT(m_zdiff);
            AVX_FLOATS r2  = AVX_ADD_FLOATS(m_sqr_zdiff,AVX_ADD_FLOATS(m_sqr_xdiff, m_sqr_ydiff));

            AVX_FLOATS m_mask_left;

            //Do all the distance cuts using masks here in new scope
            {
                //the z2 arrays are sorted in increasing order. which means
                //the z2 value will increase in any future iteration of j.
                //that implies the zdiff values are also monotonically increasing
                //Therefore, if any of the zdiff values are >= max_dz, then
                //no future iteration in j can produce a zdiff value less than pimax.
                AVX_FLOATS m_mask_geq_pimax = AVX_COMPARE_FLOATS(m_zdiff, m_max_dz,_CMP_GE_OS);
                if(AVX_TEST_COMPARISON(m_mask_geq_pimax) > 0) {
                    j = N1;//but do not break yet, there might be valid pairs in this chunk
                }

                const AVX_FLOATS m_rmax_mask = AVX_COMPARE_FLOATS(r2, m_sqr_rmax, _CMP_LT_OS);
                const AVX_FLOATS m_rmin_mask = AVX_COMPARE_FLOATS(r2, m_sqr_rmin, _CMP_GE_OS);
                //Create a combined mask by bitwise and of m1 and m_mask_left.
                //This gives us the mask for all sqr_rmin <= r2 < sqr_rmax
                m_mask_left = AVX_BITWISE_AND(m_rmax_mask,m_rmin_mask);

                //If no valid pairs, continue with the next iteration of j-loop
                const int num_left = AVX_TEST_COMPARISON(m_mask_left);
                if(num_left == 0) {
                    continue;
                }

                /* Check if all the possible pairs are in the last bin. But only run
                   this check if not evaluating same cell pairs or when simply counting
                   the pairs (no ravg requested)  */
                if(same_cell == 0 && need_ravg == 0 && need_weighted_pairs == 0) {
                    const AVX_FLOATS m_last_bin = AVX_BITWISE_AND(m_mask_left, AVX_COMPARE_FLOATS(r2, m_rupp_sqr[nbin-1], _CMP_GE_OS));
                    if(AVX_TEST_COMPARISON(m_last_bin) == num_left) { /* all the valid pairs are in the last bin */
                        npairs[nbin-1] += num_left;/* add the total number of pairs to the last bin and continue j-loop*/
                        continue;
                    }
                }

                //There is some r2 that satisfies sqr_rmin <= r2 < sqr_rmax && 0.0 <= dz^2 < pimax^2.
                r2 = AVX_BLEND_FLOATS_WITH_MASK(m_sqr_rmax, r2, m_mask_left);
            }

            AVX_FLOATS m_rbin = AVX_SETZERO_FLOAT();
            if(need_ravg) {
                union_mDperp.m_Dperp = AVX_SQRT_FLOAT(r2);
            }
            if(need_weighted_pairs){
                union_mweight.m_weights = avx_weight_func(&pair);
            }

            //Loop backwards through nbins. m_mask_left contains all the points that are less than rmax
            for(int kbin=nbin-1;kbin>=1;kbin--) {
                const AVX_FLOATS m1 = AVX_COMPARE_FLOATS(r2,m_rupp_sqr[kbin-1],_CMP_GE_OS);
                const AVX_FLOATS m_bin_mask = AVX_BITWISE_AND(m1,m_mask_left);
                const int test2  = AVX_TEST_COMPARISON(m_bin_mask);
                npairs[kbin] += AVX_BIT_COUNT_INT(test2);
                if(need_ravg || need_weighted_pairs) {
                    m_rbin = AVX_BLEND_FLOATS_WITH_MASK(m_rbin,m_kbin[kbin], m_bin_mask);
                }
                m_mask_left = AVX_COMPARE_FLOATS(r2,m_rupp_sqr[kbin-1],_CMP_LT_OS);
                const int test3 = AVX_TEST_COMPARISON(m_mask_left);
                if(test3 == 0) {
                    break;
                }
            }

            if(need_ravg || need_weighted_pairs) {
                union_rbin.m_ibin = AVX_TRUNCATE_FLOAT_TO_INT(m_rbin);
                //protect the unroll pragma in case compiler is not icc.
#if  __INTEL_COMPILER
#pragma unroll(AVX_NVEC)
#endif
                for(int jj=0;jj<AVX_NVEC;jj++) {
                    const int kbin = union_rbin.ibin[jj];
                    if(need_ravg){
                        const DOUBLE r = union_mDperp.Dperp[jj];
                        ravg[kbin] += r;
                    }
                    if(need_weighted_pairs){
                        const DOUBLE weight = union_mweight.weights[jj];
                        weighted_pairs[kbin] += weight;
                    }
                }
            }
        }//end of j-loop

        // remainder loop
        // pair.weights0[w].d was set as an AVX float, but is still valid as a DOUBLE here
        for(;j<N1;j++){
            const DOUBLE dz = *localz1++ - zpos;
            const DOUBLE dx = *localx1++ - xpos;
            const DOUBLE dy = *localy1++ - ypos;
            for(int w = 0; w < pair.num_weights; w++){
                pair.weights1[w].d = *local_w1.weights[w]++;
            }
            if(dz >= max_dz) break;

            const DOUBLE r2 = dx*dx + dy*dy + dz*dz;
            if(r2 >= sqr_rmax || r2 < sqr_rmin) {
                continue;
            }

            if(need_weighted_pairs){
                pair.dx.d = dx;
                pair.dy.d = dy;
                pair.dz.d = dz;
            }

            DOUBLE r = ZERO, pairweight = ZERO;
            if(need_ravg) {
                r = SQRT(r2);
            }
            if(need_weighted_pairs){
                pairweight = fallback_weight_func(&pair);
            }

            for(int kbin=nbin-1;kbin>=1;kbin--) {
                if(r2 >= rupp_sqr[kbin-1]) {
                    npairs[kbin]++;
                    if(need_ravg) {
                        ravg[kbin] += r;
                    }
                    if(need_weighted_pairs){
                        weighted_pairs[kbin] += pairweight;
                    }
                    break;
                }
            }
        }//remainder loop over second set of particles
    }//loop over first set of particles

	for(int i=1;i<nbin;i++) {
		src_npairs[i - 1] += npairs[i];
        if(need_ravg) {
            src_ravg[i - 1] += ravg[i];
        }
        if(need_weighted_pairs) {
            src_weighted_pairs[i - 1] += weighted_pairs[i];
        }
    }

    return EXIT_SUCCESS;
}

#endif //HAVE_AVX

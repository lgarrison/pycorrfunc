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

#include <immintrin.h>
#include <cpuid.h>


int avx_available(void) {
    return __builtin_cpu_supports("avx");
}

static inline AVX_FLOATS avx_pair_product(AVX_FLOATS, AVX_FLOATS, AVX_FLOATS, AVX_FLOATS w0, AVX_FLOATS w1){
    return AVX_MULTIPLY_FLOATS(w0, w1);
}

typedef AVX_FLOATS (*avx_weight_func_t)(AVX_FLOATS dx, AVX_FLOATS dy, AVX_FLOATS dz, AVX_FLOATS w0, AVX_FLOATS w1);
avx_weight_func_t get_avx_weight_func_by_method(const weight_method_t method){
    switch(method){
        case PAIR_PRODUCT:
            return &avx_pair_product;
        default:
        case NONE:
            return NULL;
    }
}

int countpairs_avx(
    uint64_t *restrict src_npairs, DOUBLE *restrict src_ravg, DOUBLE *restrict src_wavg,
    const int64_t N0, DOUBLE *x0, DOUBLE *y0, DOUBLE *z0, DOUBLE *w0,
    const int64_t N1, DOUBLE *x1, DOUBLE *y1, DOUBLE *z1, DOUBLE *w1,
    const int same_cell,
    const int nbinedge, const DOUBLE *bin_edges_sqr,
    const DOUBLE off_xwrap, const DOUBLE off_ywrap, const DOUBLE off_zwrap,
    const DOUBLE min_xdiff, const DOUBLE min_ydiff, const DOUBLE min_zdiff,
    const DOUBLE closest_icell_xpos, const DOUBLE closest_icell_ypos, const DOUBLE closest_icell_zpos,
    const weight_method_t weight_method)
{
    avx_weight_func_t avx_weight_func = get_avx_weight_func_by_method(weight_method);
    weight_func_t fallback_weight_func = get_weight_func_by_method(weight_method);
    
    const DOUBLE sqr_rmin = bin_edges_sqr[0];
    const DOUBLE sqr_rmax = bin_edges_sqr[nbinedge-1];

    const int32_t need_ravg = src_ravg != NULL;
    const int32_t need_wavg = src_wavg != NULL;

    AVX_FLOATS m_edge_sqr[nbinedge];
    for(int i=0;i<nbinedge;i++) {
        m_edge_sqr[i] = AVX_SET_FLOAT(bin_edges_sqr[i]);
    }

    /* variables required for ravg and wavg*/
    AVX_FLOATS m_kbin[nbinedge];
    if(need_ravg || need_wavg){
        for(int i=0;i<nbinedge;i++) {
            m_kbin[i] = AVX_SET_FLOAT((DOUBLE) i + 1);
        }
    }

    const DOUBLE *zstart = z1, *zend = z1 + N1;
    const DOUBLE max_all_dz = SQRT(sqr_rmax - min_xdiff*min_xdiff - min_ydiff*min_ydiff);
    for(int64_t i=0;i<N0;i++) {
        const DOUBLE xpos = *x0++ + off_xwrap;
        const DOUBLE ypos = *y0++ + off_ywrap;
        const DOUBLE zpos = *z0++ + off_zwrap;
        DOUBLE wi = 0.;
        if(weight_method != NONE) wi = *w0++;
        
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
        DOUBLE *localw1 = NULL;
        if(need_wavg){
            localw1 = w1 + j;
        }

        const AVX_FLOATS m_xpos    = AVX_SET_FLOAT(xpos);
        const AVX_FLOATS m_ypos    = AVX_SET_FLOAT(ypos);
        const AVX_FLOATS m_zpos    = AVX_SET_FLOAT(zpos);
        const AVX_FLOATS m_wi = AVX_SET_FLOAT(wi);

        for(;j<=(N1 - AVX_NVEC);j+=AVX_NVEC) {
            union int8 union_rbin;
            union float8 union_mDperp;
            union float8_weights union_mweight;

            const AVX_FLOATS m_x1 = AVX_LOAD_FLOATS_UNALIGNED(localx1);
            const AVX_FLOATS m_y1 = AVX_LOAD_FLOATS_UNALIGNED(localy1);
            const AVX_FLOATS m_z1 = AVX_LOAD_FLOATS_UNALIGNED(localz1);
            AVX_FLOATS m_wj = AVX_SETZERO_FLOAT();
            if(weight_method != NONE) m_wj = AVX_LOAD_FLOATS_UNALIGNED(localw1);

            localx1 += AVX_NVEC;//this might actually exceed the allocated range but we will never dereference that
            localy1 += AVX_NVEC;
            localz1 += AVX_NVEC;
            if(weight_method != NONE) localw1 += AVX_NVEC;

            const AVX_FLOATS m_max_dz = AVX_SET_FLOAT(max_dz);
            const AVX_FLOATS m_sqr_rmax = m_edge_sqr[nbinedge-1];
            const AVX_FLOATS m_sqr_rmin = m_edge_sqr[0];

            const AVX_FLOATS m_xdiff = AVX_SUBTRACT_FLOATS(m_x1, m_xpos);  //(x[j] - x0)
            const AVX_FLOATS m_ydiff = AVX_SUBTRACT_FLOATS(m_y1, m_ypos);  //(y[j] - y0)
            const AVX_FLOATS m_zdiff = AVX_SUBTRACT_FLOATS(m_z1, m_zpos);  //z2[j:j+NVEC-1] - z1

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
                if(same_cell == 0 && need_ravg == 0 && need_wavg == 0) {
                    const AVX_FLOATS m_last_bin = AVX_BITWISE_AND(m_mask_left, AVX_COMPARE_FLOATS(r2, m_edge_sqr[nbinedge-2], _CMP_GE_OS));
                    if(AVX_TEST_COMPARISON(m_last_bin) == num_left) { /* all the valid pairs are in the last bin */
                        src_npairs[nbinedge-2] += num_left;/* add the total number of pairs to the last bin and continue j-loop*/
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
            if(need_wavg){
                union_mweight.m_weights = avx_weight_func(m_xdiff, m_ydiff, m_zdiff, m_wi, m_wj);
            }

            //Loop backwards through nbins. m_mask_left contains all the points that are less than rmax
            for(int kbin=nbinedge-2;kbin>=0;kbin--) {
                const AVX_FLOATS m1 = AVX_COMPARE_FLOATS(r2,m_edge_sqr[kbin],_CMP_GE_OS);
                const AVX_FLOATS m_bin_mask = AVX_BITWISE_AND(m1,m_mask_left);
                const int test2  = AVX_TEST_COMPARISON(m_bin_mask);
                src_npairs[kbin] += AVX_BIT_COUNT_INT(test2);
                if(need_ravg || need_wavg) {
                    m_rbin = AVX_BLEND_FLOATS_WITH_MASK(m_rbin,m_kbin[kbin], m_bin_mask);
                }
                m_mask_left = AVX_COMPARE_FLOATS(r2,m_edge_sqr[kbin],_CMP_LT_OS);
                const int test3 = AVX_TEST_COMPARISON(m_mask_left);
                if(test3 == 0) {
                    break;
                }
            }

            if(need_ravg || need_wavg) {
                union_rbin.m_ibin = AVX_TRUNCATE_FLOAT_TO_INT(m_rbin);
                
                PRAGMA_UNROLL(AVX_NVEC)
                for(int jj=0;jj<AVX_NVEC;jj++) {
                    const int kbin = union_rbin.ibin[jj];
                    if (kbin == 0) continue;
                    if(need_ravg){
                        const DOUBLE r = union_mDperp.Dperp[jj];
                        src_ravg[kbin - 1] += r;
                    }
                    if(need_wavg){
                        const DOUBLE weight = union_mweight.weights[jj];
                        src_wavg[kbin - 1] += weight;
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
            DOUBLE wj = 0.;
            if(weight_method != NONE) wj = *localw1++;
            
            if(dz >= max_dz) break;

            const DOUBLE r2 = dx*dx + dy*dy + dz*dz;
            if(r2 >= sqr_rmax || r2 < sqr_rmin) continue;

            DOUBLE r = ZERO, pairweight = ZERO;
            if(need_ravg) {
                r = SQRT(r2);
            }
            if(need_wavg){
                pairweight = fallback_weight_func(dx, dy, dz, wi, wj);
            }

            for(int kbin=nbinedge-2;kbin>=0;kbin--) {
                if(r2 >= bin_edges_sqr[kbin]) {
                    src_npairs[kbin]++;
                    if(need_ravg) {
                        src_ravg[kbin] += r;
                    }
                    if(need_wavg){
                        src_wavg[kbin] += pairweight;
                    }
                    break;
                }
            }  // kbin loop
        }  // j loop
    }  // i loop

    return EXIT_SUCCESS;
}

#endif // HAVE_AVX

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>

#include "function_precision.h"
#include "utils.h"

#include "kernelfuncs.h"
#include "weights.h"

#ifdef __SSE4_2__
#include "sse_calls.h"

int sse_available(void) { return __builtin_cpu_supports("sse4.2"); }

static inline SSE_FLOATS
sse_pair_product(SSE_FLOATS, SSE_FLOATS, SSE_FLOATS, SSE_FLOATS w0, SSE_FLOATS w1) {
    return SSE_MULTIPLY_FLOATS(w0, w1);
}

typedef SSE_FLOATS (*sse_weight_func_t)(
   SSE_FLOATS dx, SSE_FLOATS dy, SSE_FLOATS dz, SSE_FLOATS w0, SSE_FLOATS w1);
sse_weight_func_t get_sse_weight_func_by_method(const weight_method_t method) {
    switch (method) {
        case PAIR_PRODUCT: return &sse_pair_product;
        default:
        case NONE: return NULL;
    }
}

int countpairs_sse(uint64_t *restrict src_npairs,
                   DoubleAccum *restrict src_ravg,
                   DoubleAccum *restrict src_wavg,
                   const int64_t N0,
                   DOUBLE *x0,
                   DOUBLE *y0,
                   DOUBLE *z0,
                   DOUBLE *w0,
                   const int64_t N1,
                   DOUBLE *x1,
                   DOUBLE *y1,
                   DOUBLE *z1,
                   DOUBLE *w1,
                   const int same_cell,
                   const int nbinedge,
                   const DOUBLE *bin_edges_sqr,
                   const DOUBLE off_xwrap,
                   const DOUBLE off_ywrap,
                   const DOUBLE off_zwrap,
                   const DOUBLE min_xdiff,
                   const DOUBLE min_ydiff,
                   const DOUBLE min_zdiff,
                   const DOUBLE closest_icell_xpos,
                   const DOUBLE closest_icell_ypos,
                   const DOUBLE closest_icell_zpos,
                   const weight_method_t weight_method) {

    sse_weight_func_t sse_weight_func  = get_sse_weight_func_by_method(weight_method);
    weight_func_t fallback_weight_func = get_weight_func_by_method(weight_method);

    const DOUBLE sqr_rmin = bin_edges_sqr[0];
    const DOUBLE sqr_rmax = bin_edges_sqr[nbinedge - 1];

    SSE_FLOATS m_edge_sqr[nbinedge];
    for (int i = 0; i < nbinedge; i++) {
        m_edge_sqr[i] = SSE_SET_FLOAT(bin_edges_sqr[i]);
    }

    const int32_t need_ravg = src_ravg != NULL;
    const int32_t need_wavg = src_wavg != NULL;

    SSE_FLOATS m_kbin[nbinedge];
    if (need_ravg || need_wavg) {
        for (int i = 0; i < nbinedge; i++) {
            m_kbin[i] = SSE_SET_FLOAT((DOUBLE) i + 1);
        }
    }

    const DOUBLE *zstart = z1, *zend = z1 + N1;
    const DOUBLE max_all_dz =
       SQRT(sqr_rmax - min_xdiff * min_xdiff - min_ydiff * min_ydiff);
    for (int64_t i = 0; i < N0; i++) {
        const DOUBLE xpos = *x0++ + off_xwrap;
        const DOUBLE ypos = *y0++ + off_ywrap;
        const DOUBLE zpos = *z0++ + off_zwrap;
        DOUBLE wi         = 0.;
        if (weight_method != NONE) wi = *w0++;

        DOUBLE max_dz = max_all_dz;

        /* Now consider if this i'th particle can be a valid pair with ANY of the
           remaining j' particles. The min. difference in the z-positions between this
           i'th particle and ANY of the remaining j'th particles, is the difference
           between the current j'th particle and the current i'th particle (since all
           remaining j'th particles will have a larger value for the z-ordinate and
           therefore a larger difference to zpos). if this `dz` does not satisfy the
           distance criteria, then NO remaining j'th particles will. Continue on to the
           next i'th particle
        */
        const DOUBLE this_dz = *z1 - zpos;
        if (this_dz >= max_all_dz) { continue; }

        /* Okay so there MAY be a pair */
        if (same_cell == 1) {
            z1++;
        } else {
            // Now add the x,y information to further limit the range of secondaries for
            // this particle But note this constraint may increase or decrease for the
            // next particle, since x,y aren't sorted!
            const DOUBLE min_dx =
               min_xdiff > 0 ? min_xdiff + FABS(xpos - closest_icell_xpos) : min_xdiff;
            const DOUBLE min_dy =
               min_ydiff > 0 ? min_ydiff + FABS(ypos - closest_icell_ypos) : min_ydiff;
            const DOUBLE min_dz =
               min_zdiff > 0 ?
                  (this_dz > 0 ? this_dz :
                                 min_zdiff + FABS(zpos - closest_icell_zpos)) :
                  min_zdiff;
            const DOUBLE sqr_min_sep_this_point =
               min_dx * min_dx + min_dy * min_dy + min_dz * min_dz;
            if (sqr_min_sep_this_point >= sqr_rmax) { continue; }
            max_dz = SQRT(sqr_rmax - min_dx * min_dx - min_dy * min_dy);

            // Now "fast forward" in the list of secondary particles to find the first
            // one that satisfies the max_all_dz constraint We don't consider the i
            // particle's x,y information yet, because those aren't sorted
            const DOUBLE target_z = zpos - max_all_dz;
            while (z1 < zend && *z1 <= target_z) { z1++; }
        }
        // If no j particle satisfies the constraint for this i particle,
        // then the same holds true for all future i particles because they are sorted
        // in increasing z order
        if (z1 == zend) {
            i = N0;
            break;
        }

        DOUBLE *localz1       = z1;
        const DOUBLE target_z = zpos - max_dz;
        while (localz1 != zend && *localz1 <= target_z) { localz1++; }

        int64_t j       = localz1 - zstart;
        DOUBLE *localx1 = x1 + j;
        DOUBLE *localy1 = y1 + j;
        DOUBLE *localw1 = NULL;
        if (need_wavg) { localw1 = w1 + j; }

        for (; j <= (N1 - SSE_NVEC); j += SSE_NVEC) {
            union int4 union_rbin;
            union float4 union_mDperp;
            union float4_weights union_mweight;

            const SSE_FLOATS m_xpos = SSE_SET_FLOAT(xpos);
            const SSE_FLOATS m_ypos = SSE_SET_FLOAT(ypos);
            const SSE_FLOATS m_zpos = SSE_SET_FLOAT(zpos);
            const SSE_FLOATS m_wi   = SSE_SET_FLOAT(wi);

            const SSE_FLOATS m_x1 = SSE_LOAD_FLOATS_UNALIGNED(localx1);
            const SSE_FLOATS m_y1 = SSE_LOAD_FLOATS_UNALIGNED(localy1);
            const SSE_FLOATS m_z1 = SSE_LOAD_FLOATS_UNALIGNED(localz1);
            SSE_FLOATS m_wj       = SSE_SETZERO_FLOAT();
            if (weight_method != NONE) m_wj = SSE_LOAD_FLOATS_UNALIGNED(localw1);

            localx1 += SSE_NVEC;
            localy1 += SSE_NVEC;
            localz1 += SSE_NVEC;
            if (weight_method != NONE) localw1 += SSE_NVEC;

            const SSE_FLOATS m_max_dz   = SSE_SET_FLOAT(max_dz);
            const SSE_FLOATS m_sqr_rmax = SSE_SET_FLOAT(sqr_rmax);
            const SSE_FLOATS m_sqr_rmin = SSE_SET_FLOAT(sqr_rmin);

            const SSE_FLOATS m_xdiff = SSE_SUBTRACT_FLOATS(m_x1, m_xpos);  //(x[j] - x0)
            const SSE_FLOATS m_ydiff = SSE_SUBTRACT_FLOATS(m_y1, m_ypos);  //(y[j] - y0)
            const SSE_FLOATS m_zdiff =
               SSE_SUBTRACT_FLOATS(m_z1, m_zpos);  // z2[j:j+NVEC-1] - z1

            const SSE_FLOATS m_mask_geq_pimax =
               SSE_COMPARE_FLOATS_GE(m_zdiff, m_max_dz);
            if (SSE_TEST_COMPARISON(m_mask_geq_pimax) > 0) {
                j = N1;  // but do not break yet, there might be pairs in this chunk
            }

            const SSE_FLOATS m_sqr_xdiff = SSE_SQUARE_FLOAT(m_xdiff);
            const SSE_FLOATS m_sqr_ydiff = SSE_SQUARE_FLOAT(m_ydiff);
            const SSE_FLOATS m_sqr_zdiff = SSE_SQUARE_FLOAT(m_zdiff);

            SSE_FLOATS r2 =
               SSE_ADD_FLOATS(m_sqr_zdiff, SSE_ADD_FLOATS(m_sqr_xdiff, m_sqr_ydiff));
            SSE_FLOATS m_mask_left;
            {
                const SSE_FLOATS m_rmin_mask = SSE_COMPARE_FLOATS_GE(r2, m_sqr_rmin);
                const SSE_FLOATS m_rmax_mask = SSE_COMPARE_FLOATS_LT(r2, m_sqr_rmax);
                m_mask_left = SSE_BITWISE_AND(m_rmin_mask, m_rmax_mask);
                if (SSE_TEST_COMPARISON(m_mask_left) == 0) { continue; }
                r2 = SSE_BLEND_FLOATS_WITH_MASK(m_sqr_rmax, r2, m_mask_left);
            }

            SSE_FLOATS m_rbin = SSE_SETZERO_FLOAT();
            if (need_ravg) { union_mDperp.m_Dperp = SSE_SQRT_FLOAT(r2); }
            if (need_wavg) {
                union_mweight.m_weights =
                   sse_weight_func(m_xdiff, m_ydiff, m_zdiff, m_wi, m_wj);
            }

            for (int kbin = nbinedge - 2; kbin >= 0; kbin--) {
                SSE_FLOATS m1         = SSE_COMPARE_FLOATS_GE(r2, m_edge_sqr[kbin]);
                SSE_FLOATS m_bin_mask = SSE_BITWISE_AND(m1, m_mask_left);
                m_mask_left           = SSE_COMPARE_FLOATS_LT(r2, m_edge_sqr[kbin]);
                int test2             = SSE_TEST_COMPARISON(m_bin_mask);
                src_npairs[kbin] += SSE_BIT_COUNT_INT(test2);
                if (need_ravg || need_wavg) {
                    m_rbin =
                       SSE_BLEND_FLOATS_WITH_MASK(m_rbin, m_kbin[kbin], m_bin_mask);
                }
                int test3 = SSE_TEST_COMPARISON(m_mask_left);
                if (test3 == 0) { break; }
            }

            if (need_ravg || need_wavg) {
                union_rbin.m_ibin = SSE_TRUNCATE_FLOAT_TO_INT(m_rbin);
                // protect the unroll pragma in case compiler is not icc.

                PRAGMA_UNROLL(SSE_NVEC)
                for (int jj = 0; jj < SSE_NVEC; jj++) {
                    const int kbin = union_rbin.ibin[jj];
                    if (need_ravg) {
                        const DOUBLE r = union_mDperp.Dperp[jj];
                        src_ravg[kbin] += r;
                    }
                    if (need_wavg) {
                        const DOUBLE weight = union_mweight.weights[jj];
                        src_wavg[kbin] += weight;
                    }
                }
            }  // ravg
        }

        for (; j < N1; j++) {
            const DOUBLE dx = *localx1++ - xpos;
            const DOUBLE dy = *localy1++ - ypos;
            const DOUBLE dz = *localz1++ - zpos;
            DOUBLE wj       = 0.;
            if (weight_method != NONE) wj = *localw1++;

            if (dz >= max_dz) break;

            const DOUBLE r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= sqr_rmax || r2 < sqr_rmin) continue;

            DOUBLE r = ZERO, pairweight = ZERO;
            if (need_ravg) { r = SQRT(r2); }
            if (need_wavg) { pairweight = fallback_weight_func(dx, dy, dz, wi, wj); }

            for (int kbin = nbinedge - 2; kbin >= 0; kbin--) {
                if (r2 >= bin_edges_sqr[kbin]) {
                    src_npairs[kbin]++;
                    if (need_ravg) { src_ravg[kbin + 1] += r; }
                    if (need_wavg) { src_wavg[kbin + 1] += pairweight; }
                    break;
                }
            }  // kbin loop
        }  // j loop
    }  // i loop

    return EXIT_SUCCESS;
}
#endif  // __SSE4_2__

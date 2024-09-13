#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "function_precision.h"
#include "utils.h"

#include "weight_functions.h"
#include "kernelfuncs.h"


int countpairs_fallback(const int64_t N0, DOUBLE *x0, DOUBLE *y0, DOUBLE *z0, const weight_struct *weights0,
                                             const int64_t N1, DOUBLE *x1, DOUBLE *y1, DOUBLE *z1, const weight_struct *weights1,
                                             const int same_cell,
                                             const DOUBLE sqr_rpmax, const DOUBLE sqr_rpmin, const int nbin, const DOUBLE *rupp_sqr, const DOUBLE rpmax,
                                             const DOUBLE off_xwrap, const DOUBLE off_ywrap, const DOUBLE off_zwrap,
                                             const DOUBLE min_xdiff, const DOUBLE min_ydiff, const DOUBLE min_zdiff,
                                             const DOUBLE closest_icell_xpos, const DOUBLE closest_icell_ypos, const DOUBLE closest_icell_zpos,
                                             DOUBLE *restrict src_rpavg, uint64_t *restrict src_npairs,
                                             DOUBLE *restrict src_weighted_pairs, const weight_method_t weight_method)
{
    /*----------------- FALLBACK CODE --------------------*/
    /* implementation that is guaranteed to compile */

    pair_struct pair = {.num_weights=0};
    weight_func_t weight_func = NULL;
    if(src_weighted_pairs != NULL){
        pair.num_weights = weights0->num_weights;
        weight_func = get_weight_func_by_method(weight_method);
    }

    const DOUBLE *zstart = z1, *zend = z1 + N1;
    const DOUBLE max_all_dz = SQRT(rpmax*rpmax - min_xdiff*min_xdiff - min_ydiff*min_ydiff);
    for(int64_t i=0;i<N0;i++) {
        const DOUBLE xpos = *x0++ + off_xwrap;
        const DOUBLE ypos = *y0++ + off_ywrap;
        const DOUBLE zpos = *z0++ + off_zwrap;
        for(int w = 0; w < pair.num_weights; w++){
            pair.weights0[w].d = *(weights0->weights[w] + i);
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
        const int64_t nleft = zend - localz1;
        DOUBLE *localx1 = x1 + n_off;
        DOUBLE *localy1 = y1 + n_off;

        for(int64_t j=0;j<nleft;j++) {
            const DOUBLE dx = *localx1++ - xpos;
            const DOUBLE dy = *localy1++ - ypos;
            const DOUBLE dz = *localz1++ - zpos;
            if(dz >= max_dz) break;

            const DOUBLE r2 = dx*dx + dy*dy + dz*dz;
            if(r2 >= sqr_rpmax || r2 < sqr_rpmin) continue;

            DOUBLE pairweight = ZERO;
            if(src_weighted_pairs != NULL){
                for(int w = 0; w < pair.num_weights; w++){
                    pair.weights1[w].d = *(weights1->weights[w] + n_off + j);
                }
                
                pair.dx.d = dx;
                pair.dy.d = dy;
                pair.dz.d = dz;
                pairweight = weight_func(&pair);
            }

            for(int kbin=nbin-1;kbin>=1;kbin--){
                if(r2 >= rupp_sqr[kbin-1]) {
                    src_npairs[kbin - 1]++;

                    printf("src_npairs is: ");
                    for (int i = 0; i < 2; i++) {
                        printf("%lu ", src_npairs[i]);
                    }
                    printf("\n");
                    
                    if(src_rpavg != NULL) {
                        src_rpavg[kbin - 1] += SQRT(r2);
                    }
                    if(src_weighted_pairs != NULL){
                        src_weighted_pairs[kbin - 1] += pairweight;
                    }
                    break;
                }
            }//searching for kbin loop
        }
    }

    /*----------------- FALLBACK CODE --------------------*/
    return EXIT_SUCCESS;
}
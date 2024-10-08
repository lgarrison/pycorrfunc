#pragma once

#include <stdint.h>

#include "defs.h"

extern int get_gridsize(DOUBLE *xgridsize,
                        int *nlattice,
                        const DOUBLE xdiff,
                        const DOUBLE xwrap,
                        const DOUBLE rmax,
                        const int refine_factor,
                        const int max_ncells) __attribute__((warn_unused_result));

extern void get_max_min(const int64_t ND1,
                        const DOUBLE *restrict X1,
                        const DOUBLE *restrict Y1,
                        const DOUBLE *restrict Z1,
                        DOUBLE *min_x,
                        DOUBLE *min_y,
                        DOUBLE *min_z,
                        DOUBLE *max_x,
                        DOUBLE *max_y,
                        DOUBLE *max_z);

extern void get_max_min_ra_dec(const int64_t ND1,
                               const DOUBLE *RA,
                               const DOUBLE *DEC,
                               DOUBLE *ra_min,
                               DOUBLE *dec_min,
                               DOUBLE *ra_max,
                               DOUBLE *dec_max);

extern DOUBLE find_closest_pos(const DOUBLE first_xbounds[2],
                               const DOUBLE second_xbounds[2],
                               DOUBLE *closest_pos0)
   __attribute__((warn_unused_result));

extern void find_min_and_max_sqr_sep_between_cell_pairs(const DOUBLE first_xbounds[2],
                                                        const DOUBLE first_ybounds[2],
                                                        const DOUBLE first_zbounds[2],
                                                        const DOUBLE second_xbounds[2],
                                                        const DOUBLE second_ybounds[2],
                                                        const DOUBLE second_zbounds[2],
                                                        DOUBLE *sqr_sep_min,
                                                        DOUBLE *sqr_sep_max);

#define CHECK_AND_CONTINUE_FOR_DUPLICATE_NGB_CELLS(icell,                                                      \
                                                   icell2,                                                     \
                                                   off_xwrap,                                                  \
                                                   off_ywrap,                                                  \
                                                   off_zwrap,                                                  \
                                                   num_cell_pairs,                                             \
                                                   num_ngb_this_cell,                                          \
                                                   all_cell_pairs)                                             \
    {                                                                                                          \
        int duplicate_flag = 0;                                                                                \
        XRETURN(                                                                                               \
           num_cell_pairs - num_ngb_this_cell >= 0,                                                            \
           NULL,                                                                                               \
           "Error: While working on detecting (potential) duplicate cell-pairs on primary cell = %" PRId64     \
           "\n"                                                                                                \
           "The total number of cell-pairs (across all primary cells) = %" PRId64                              \
           " should be >= the number of cell-pairs for "                                                       \
           "this primary cell = %" PRId64 "\n",                                                                \
           icell,                                                                                              \
           num_cell_pairs,                                                                                     \
           num_ngb_this_cell);                                                                                 \
                                                                                                               \
        for (int jj = 0; jj < num_ngb_this_cell; jj++) {                                                       \
            struct cell_pair *this_cell_pair =                                                                 \
               &all_cell_pairs[num_cell_pairs - jj - 1];                                                       \
            XRETURN(                                                                                           \
               this_cell_pair->cellindex1 == icell,                                                            \
               NULL,                                                                                           \
               "Error: While working on detecting (potential) duplicate cell-pairs on primary cell = %" PRId64 \
               "\n"                                                                                            \
               "For cell-pair # %" PRId64                                                                      \
               ", the primary cellindex (within cell-pair) = %" PRId64                                         \
               " should be *exactly* "                                                                         \
               "equal to current primary cellindex = %" PRId64                                                 \
               ". Num_cell_pairs = %" PRId64 " num_ngb_this_cell = %" PRId64 "\n",                             \
               icell,                                                                                          \
               num_cell_pairs - jj,                                                                            \
               this_cell_pair->cellindex1,                                                                     \
               icell,                                                                                          \
               num_cell_pairs,                                                                                 \
               num_ngb_this_cell);                                                                             \
            _Pragma("GCC diagnostic push")                                                                     \
               _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"") if (                                        \
                  this_cell_pair->cellindex2 == icell2                                                         \
                  && this_cell_pair->xwrap == off_xwrap                                                        \
                  && this_cell_pair->ywrap == off_ywrap                                                        \
                  && this_cell_pair->zwrap == off_zwrap) {                                                     \
                duplicate_flag = 1;                                                                            \
                break;                                                                                         \
            }                                                                                                  \
            _Pragma("GCC diagnostic pop")                                                                      \
        }                                                                                                      \
        if (duplicate_flag == 1) continue;                                                                     \
    }

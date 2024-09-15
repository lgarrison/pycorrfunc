#pragma once

#include "defs.h"
#include "weight_defs.h"
#include <inttypes.h>

typedef int (*countpairs_func_ptr)(
    const int64_t N0, DOUBLE *x0, DOUBLE *y0, DOUBLE *z0, const weight_struct *weights0,
    const int64_t N1, DOUBLE *x1, DOUBLE *y1, DOUBLE *z1, const weight_struct *weights1,
    const int same_cell,
    const DOUBLE sqr_rpmax, const DOUBLE sqr_rpmin, const int nbin, const DOUBLE *bin_edges_sqr, const DOUBLE rpmax,
    const DOUBLE off_xwrap, const DOUBLE off_ywrap, const DOUBLE off_zwrap,
    const DOUBLE min_xdiff, const DOUBLE min_ydiff, const DOUBLE min_zdiff,
    const DOUBLE closest_icell_xpos, const DOUBLE closest_icell_ypos, const DOUBLE closest_icell_zpos,
    DOUBLE *src_rpavg, uint64_t *src_npairs,
    DOUBLE *src_weighted_pairs, const weight_method_t weight_method
);


int countpairs(
    const int64_t ND1, DOUBLE *X1, DOUBLE *Y1, DOUBLE *Z1,
    const int64_t ND2, DOUBLE *X2, DOUBLE *Y2, DOUBLE *Z2,
    const int64_t N_bin_edges, const DOUBLE *bin_edges,
    config_options *options,
    uint64_t *npairs,
    DOUBLE *ravg,
    DOUBLE *weighted_pairs
);

#pragma once

#include <stdint.h>

#include "defs.h"
#include "cellarray.h"
#include "cell_pair.h"

cellarray *gridlink(
    const int64_t NPART,
    DOUBLE * const X, DOUBLE * const Y, DOUBLE * const Z, DOUBLE * const W,
    const DOUBLE xmin, const DOUBLE xmax,
    const DOUBLE ymin, const DOUBLE ymax,
    const DOUBLE zmin, const DOUBLE zmax,
    const DOUBLE max_x_size,
    const DOUBLE max_y_size,
    const DOUBLE max_z_size,
    const DOUBLE xwrap,
    const DOUBLE ywrap,
    const DOUBLE zwrap,
    const int xbin_refine_factor,
    const int ybin_refine_factor,
    const int zbin_refine_factor,
    const int sort_on_z,
    const config_options *options
) __attribute__((warn_unused_result));

struct cell_pair *generate_cell_pairs(
    int64_t *ncell_pairs,
    const cellarray *lattice1,
    const cellarray *lattice2,
    const int xbin_refine_factor, const int ybin_refine_factor, const int zbin_refine_factor,
    const DOUBLE xdiff, const DOUBLE ydiff, const DOUBLE zdiff,
    const DOUBLE max_3D_sep, const DOUBLE max_2D_sep, const DOUBLE max_1D_sep,
    const int enable_min_sep_opt,
    const int autocorr,
    const int periodic_x, const int periodic_y, const int periodic_z)
    __attribute__((warn_unused_result));

void free_cellarray(cellarray **lattice);

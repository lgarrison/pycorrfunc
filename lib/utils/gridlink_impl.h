// # -*- mode: c -*-
/* File: gridlink_impl.h.src */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/

#pragma once

#include <stdint.h>

#include "defs.h"  //for definition of config_options
#include "cellarray.h" //for definition of struct cellarray_double/float
#include "cell_pair.h" //for definition of struct cell_pair_double/float
#include "weight_defs.h" //for definition of weight_struct

cellarray * gridlink(
    const int64_t np,
    DOUBLE *x, DOUBLE *y, DOUBLE *z, weight_struct *weights,
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
    int *nlattice_x,
    int *nlattice_y,
    int *nlattice_z,
    const config_options *options) __attribute__((warn_unused_result));

struct cell_pair * generate_cell_pairs(
    struct cellarray *lattice1,
    struct cellarray *lattice2,
    const int64_t totncells,
    int64_t *ncell_pairs,
    const int xbin_refine_factor, const int ybin_refine_factor, const int zbin_refine_factor,
    const int nmesh_x, const int nmesh_y, const int nmesh_z,
    const DOUBLE xdiff, const DOUBLE ydiff, const DOUBLE zdiff,
    const DOUBLE max_3D_sep, const DOUBLE max_2D_sep, const DOUBLE max_1D_sep,
    const int enable_min_sep_opt,
    const int autocorr,
    const int periodic_x, const int periodic_y, const int periodic_z)
    __attribute__((warn_unused_result));

void free_cellarray(cellarray *lattice, const int64_t totncells);

// # -*- mode: c -*-
/* File: cell_pair.h.src */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "weight_defs.h"

struct cell_pair{
    int64_t cellindex1;
    int64_t cellindex2;

    DOUBLE xwrap;
    DOUBLE ywrap;
    DOUBLE zwrap;

    DOUBLE min_dx;
    DOUBLE min_dy;
    DOUBLE min_dz;

    DOUBLE closest_x1;
    DOUBLE closest_y1;
    DOUBLE closest_z1;

    int8_t same_cell;
};

#ifdef __cplusplus
}
#endif

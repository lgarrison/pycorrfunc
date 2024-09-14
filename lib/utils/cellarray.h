// # -*- mode: c -*-
/* File: cellarray.h.src */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/

#pragma once

#include <stdint.h>

#include "weight_defs.h"

typedef struct cellarray cellarray;
struct cellarray{
    int64_t nelements;//Here the xyz positions will be stored in their individual pointers. More amenable to sorting -> used by wp and xi
    DOUBLE *x;
    DOUBLE *y;
    DOUBLE *z;
    weight_struct weights;
    DOUBLE xbounds[2];//xmin and xmax for entire cell
    DOUBLE ybounds[2];
    DOUBLE zbounds[2];

    int64_t *original_index;//the input order for particles
    uint8_t owns_memory;// boolean flag if the x/y/z pointers were separately malloc'ed -> need to be freed once calculations are done

    /*
      boolean flag (only relevant when external particle positions
      are used) to re-order particles back into original order
      after calculations are done. Only relevant if external
      pointers are being used for x/y/z
    */
};

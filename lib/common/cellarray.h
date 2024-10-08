#pragma once

#include <stdint.h>

#include "defs.h"

typedef struct {
    int64_t *offsets;  // where each cell starts
    DOUBLE *X;
    DOUBLE *Y;
    DOUBLE *Z;
    DOUBLE *W;
    DOUBLE *xbounds[2];  // xmin and xmax for each cell
    DOUBLE *ybounds[2];
    DOUBLE *zbounds[2];

    int nmesh_x;
    int nmesh_y;
    int nmesh_z;
    int64_t tot_ncells;

    int have_weights;
} cellarray;

cellarray *allocate_cellarray(
   const int64_t np, const int nx, const int ny, const int nz, const int with_weights);

int validate_cellarray(const cellarray *lattice);

void free_cellarray(cellarray **lattice);

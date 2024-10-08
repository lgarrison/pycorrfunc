#pragma once

#include <stdint.h>

typedef struct {
    int64_t nelements;
    DOUBLE *x;
    DOUBLE *y;
    DOUBLE *z;
    weight_struct weights;
    DOUBLE xbounds[2];
    DOUBLE ybounds[2];
    DOUBLE zbounds[2];

    // these two fields are only relevant for angular calculations
    DOUBLE dec_bounds[2];
    DOUBLE ra_bounds[2];

    int64_t *original_index;  // the input order for particles
    uint8_t owns_memory;      // boolean flag if the x/y/z pointers were separately
                          // malloc'ed -> need to be freed once calculations are done

    /*
      boolean flag (only relevant when external particle positions
      are used) to re-order particles back into original order
      after calculations are done. Only relevant if external
      pointers are being used for x/y/z
    */

    uint8_t unused[7];  // to maintain alignment explicitly (the compiler would insert
                        // this anyway)
} cellarray_mocks;

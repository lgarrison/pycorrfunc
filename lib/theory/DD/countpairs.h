#pragma once

#include <inttypes.h>

#include "defs.h"

int countpairs(const int64_t ND1,
               DOUBLE *X1,
               DOUBLE *Y1,
               DOUBLE *Z1,
               DOUBLE *W1,
               const int64_t ND2,
               DOUBLE *X2,
               DOUBLE *Y2,
               DOUBLE *Z2,
               DOUBLE *W2,
               const int64_t N_bin_edges,
               const DOUBLE *bin_edges,
               config_options *options,
               uint64_t *npairs,
               DoubleAccum *ravg,
               DoubleAccum *wavg);

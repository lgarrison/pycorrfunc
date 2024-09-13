// # -*- mode: c -*-
#pragma once

#include <stdint.h>
#define MAX_NUM_WEIGHTS 10
#ifdef CORRFUNC_DOUBLE
    #define DOUBLE double
#else
    #define DOUBLE float
#endif

typedef struct
{
    DOUBLE *weights[MAX_NUM_WEIGHTS];  // This will be of shape weights[num_weights][num_particles]
    int64_t num_weights;
} weight_struct;

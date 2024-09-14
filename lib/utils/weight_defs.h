// # -*- mode: c -*-
#pragma once

#include <stdint.h>

#include "defs.h"

#define MAX_NUM_WEIGHTS 10

typedef struct
{
    DOUBLE *weights[MAX_NUM_WEIGHTS];  // This will be of shape weights[num_weights][num_particles]
    int64_t num_weights;
} weight_struct;

#pragma once

#include <stdint.h>

#include "defs.h"

int get_num_weights_by_method(const weight_method_t method);
int get_weight_method_by_name(const char *name, weight_method_t *method);

/* Weight function pointer type definitions */

typedef DOUBLE (*weight_func_t)(DOUBLE dx, DOUBLE dy, DOUBLE dz, DOUBLE w0, DOUBLE w1);

/*
 * The pair weight is the product of the particle weights
 */
static inline DOUBLE pair_product(
   DOUBLE UNUSED(dx), DOUBLE UNUSED(dy), DOUBLE UNUSED(dz), DOUBLE w0, DOUBLE w1) {
    return w0 * w1;
}

weight_func_t get_weight_func_by_method(const weight_method_t method);

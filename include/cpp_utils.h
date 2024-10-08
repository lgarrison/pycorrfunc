#pragma once

#include <stdint.h>

#include "defs.h"

// This is currently not used for anything critical, so we can just use a
// reasonable default value.
static const size_t CACHELINE = 64;

void raise_python_exception();
void raise_RuntimeError(const char *msg);

void argsort(int64_t *iord, DOUBLE *arr, int64_t n);

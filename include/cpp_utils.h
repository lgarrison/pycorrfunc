#pragma once

#include <stdint.h>

#include "defs.h"

#ifdef __cpp_lib_hardware_interference_size
static const size_t CACHELINE = std::hardware_destructive_interference_size;
#else
static const size_t CACHELINE = 64;
#endif

void raise_python_exception();
void raise_RuntimeError(const char *msg);

void argsort(int64_t *iord, DOUBLE *arr, int64_t n);

#pragma once

#include <stdint.h>

#include "defs.h"

void raise_python_exception();
void raise_RuntimeError(const char *msg);

void argsort(int64_t *iord, DOUBLE *arr, int64_t n);

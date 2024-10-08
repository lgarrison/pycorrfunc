#include <algorithm>
#include <numeric>
#include <vector>

#include <nanobind/nanobind.h>

extern "C" {
#include "cpp_utils.h"
#include "defs.h"
}

namespace nb = nanobind;

void raise_python_exception() {
    // Raises the current Python exception, e.g. KeyboardInterrupt
    throw nb::python_error();
}

void raise_RuntimeError(const char *msg) {
    // Raises a Python RuntimeError with the given message
    throw std::runtime_error(msg);
}

void argsort(int64_t *iord, DOUBLE *arr, int64_t n) {
    // Sorts the array `arr` and returns the indices in `iord`
    std::iota(iord, iord + n, 0);
    std::sort(
       iord, iord + n, [&arr](int64_t i1, int64_t i2) { return arr[i1] < arr[i2]; });
}

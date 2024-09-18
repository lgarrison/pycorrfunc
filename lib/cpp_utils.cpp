#include <nanobind/nanobind.h>

extern "C" {
    #include "cpp_utils.h"
}

namespace nb = nanobind;

void raise_python_error(){
    // Raises the current Python exception, e.g. KeyboardInterrupt
    throw nb::python_error();
}

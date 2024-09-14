#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {
    #include "theory/DD/countpairs.h"
}

#ifdef CORRFUNC_USE_DOUBLE
#define PYBIND_NAME _corrfunc
#else 
#define PYBIND_NAME _corrfuncf
#endif

namespace py = pybind11;

void countpairs_wrapper(py::array_t<DOUBLE> X1, py::array_t<DOUBLE> Y1, py::array_t<DOUBLE> Z1,
                        py::array_t<DOUBLE> X2, py::array_t<DOUBLE> Y2, py::array_t<DOUBLE> Z2,
                        py::array_t<DOUBLE> bin_edges, py::array_t<uint64_t> npairs,
                        py::array_t<DOUBLE> ravg, py::array_t<DOUBLE> weighted_pairs,
                        int numthreads, DOUBLE boxsize) {

    py::buffer_info X1_buf = X1.request();
    py::buffer_info Y1_buf = Y1.request();
    py::buffer_info Z1_buf = Z1.request();
    py::buffer_info X2_buf = X2.request();
    py::buffer_info Y2_buf = Y2.request();
    py::buffer_info Z2_buf = Z2.request();
    py::buffer_info bin_edges_buf = bin_edges.request();
    py::buffer_info npairs_buf = npairs.request();
    py::buffer_info ravg_buf = ravg.request();
    py::buffer_info weighted_pairs_buf = weighted_pairs.request();

    DOUBLE* X1_ptr = static_cast<DOUBLE*>(X1_buf.ptr);
    DOUBLE* Y1_ptr = static_cast<DOUBLE*>(Y1_buf.ptr);
    DOUBLE* Z1_ptr = static_cast<DOUBLE*>(Z1_buf.ptr);
    DOUBLE* X2_ptr = static_cast<DOUBLE*>(X2_buf.ptr);
    DOUBLE* Y2_ptr = static_cast<DOUBLE*>(Y2_buf.ptr);
    DOUBLE* Z2_ptr = static_cast<DOUBLE*>(Z2_buf.ptr);
    DOUBLE* bin_edges_ptr = static_cast<DOUBLE*>(bin_edges_buf.ptr);

    uint64_t* npairs_ptr = static_cast<uint64_t*>(npairs_buf.ptr);
    DOUBLE* ravg_ptr = static_cast<DOUBLE*>(ravg_buf.ptr);
    DOUBLE* weighted_pairs_ptr = static_cast<DOUBLE*>(weighted_pairs_buf.ptr);

    if (numthreads < 1) {
#ifdef _OPENMP
        numthreads = omp_get_max_threads();
#else
        numthreads = 1;
#endif
    }

    struct config_options options = get_config_options(NONE);
    options.numthreads = numthreads;
    options.boxsize_x = boxsize;
    options.boxsize_y = boxsize;
    options.boxsize_z = boxsize;
    options.periodic = boxsize > (DOUBLE) 0.;
    options.autocorr = 0;
    options.verbose = 1;

    int64_t ND1 = X1_buf.shape[0];
    int64_t ND2 = X2_buf.shape[0];
    int64_t N_bin_edges = bin_edges_buf.shape[0];

    countpairs(ND1, X1_ptr, Y1_ptr, Z1_ptr, ND2, X2_ptr, Y2_ptr, Z2_ptr, N_bin_edges, bin_edges_ptr, 
        &options, npairs_ptr,ravg_ptr, weighted_pairs_ptr);
}

PYBIND11_MODULE(PYBIND_NAME, m) {
    m.def("countpairs", &countpairs_wrapper, py::arg("X1"), py::arg("Y1"), py::arg("Z1"),
          py::arg("X2"), py::arg("Y2"), py::arg("Z2"), py::arg("bin_edges"), py::arg("npairs"),
          py::arg("ravg"), py::arg("weighted_pairs"), py::arg("num_threads"), py::arg("boxsize"));
}

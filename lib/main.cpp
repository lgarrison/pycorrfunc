#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {
    #include "theory/DD/countpairs.h"
}

#ifdef CORRFUNC_USE_DOUBLE
#define NB_NAME _corrfunc
#else 
#define NB_NAME _corrfuncf
#endif

namespace nb = nanobind;
using namespace nb::literals;

template<typename T>
using array_t = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

void countpairs_wrapper(
    array_t<DOUBLE> X1,
    array_t<DOUBLE> Y1,
    array_t<DOUBLE> Z1,
    std::optional<array_t<DOUBLE>> X2,
    std::optional<array_t<DOUBLE>> Y2,
    std::optional<array_t<DOUBLE>> Z2,
    array_t<const DOUBLE> bin_edges,
    array_t<uint64_t> npairs,
    array_t<DOUBLE> ravg,
    array_t<DOUBLE> weighted_pairs,
    int numthreads,
    std::optional<DOUBLE> boxsize,
    std::optional<std::string> weight_type,
    bool verbose,
    int isa
    ) {

    if (numthreads < 1) {
#ifdef _OPENMP
        numthreads = omp_get_max_threads();
#else
        numthreads = 1;
#endif
    }

    const char *weight_str = weight_type.has_value() ? weight_type->c_str() : NULL;
    config_options options = get_config_options(weight_str);
    options.numthreads = numthreads;
    options.periodic = boxsize.has_value();
    if(options.periodic){
        // TODO: anisotropic boxsize
        options.boxsize_x = boxsize.value();
        options.boxsize_y = boxsize.value();
        options.boxsize_z = boxsize.value();
    }
    options.autocorr = !X2.has_value();
    options.verbose = verbose;
    options.instruction_set = static_cast<isa_t>(isa);

    int64_t ND1 = X1.shape(0);
    int64_t ND2 = X2.has_value() ? X2->shape(0) : 0;
    int64_t N_bin_edges = bin_edges.shape(0);

    DOUBLE *X2_ptr = X2.has_value() ? X2->data() : nullptr;
    DOUBLE *Y2_ptr = Y2.has_value() ? Y2->data() : nullptr;
    DOUBLE *Z2_ptr = Z2.has_value() ? Z2->data() : nullptr;

    countpairs(
        ND1,
        X1.data(), Y1.data(), Z1.data(),
        ND2,
        X2_ptr, Y2_ptr, Z2_ptr,
        N_bin_edges,
        bin_edges.data(), 
        &options,
        npairs.data(),
        ravg.data(),
        weighted_pairs.data()
    );
}

NB_MODULE(NB_NAME, m) {
    m.def(
        "countpairs",
        &countpairs_wrapper,
        "X1"_a.noconvert(),
        "Y1"_a.noconvert(),
        "Z1"_a.noconvert(),
        "X2"_a.noconvert().none(),
        "Y2"_a.noconvert().none(),
        "Z2"_a.noconvert().none(),
        "bin_edges"_a.noconvert(),
        "npairs"_a.noconvert(),
        "ravg"_a.noconvert(),
        "weighted_pairs"_a.noconvert(),
        "numthreads"_a.noconvert(),
        "boxsize"_a.noconvert().none(),
        "weight_type"_a.noconvert().none(),
        "verbose"_a.noconvert(),
        "isa"_a.noconvert()
        );
}

#include <array>
#include <optional>
#include <variant>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

extern "C" {
    #include "theory/DD/countpairs.h"
}

#ifdef PYCORRFUNC_USE_DOUBLE
#define NB_NAME _pycorrfunc
#else 
#define NB_NAME _pycorrfuncf
#endif

namespace nb = nanobind;
using namespace nb::literals;

template<typename T>
using array_t = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template<typename T>
using array2D_t = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

void countpairs_wrapper(
    array_t<DOUBLE> X1,
    array_t<DOUBLE> Y1,
    array_t<DOUBLE> Z1,
    std::optional<array_t<DOUBLE>> W1,
    std::optional<array_t<DOUBLE>> X2,
    std::optional<array_t<DOUBLE>> Y2,
    std::optional<array_t<DOUBLE>> Z2,
    std::optional<array_t<DOUBLE>> W2,
    array_t<const DOUBLE> bin_edges,
    array_t<uint64_t> npairs,
    array_t<DOUBLE> ravg,
    array_t<DOUBLE> wavg,
    int num_threads,
    std::optional<nb::ndarray<const DOUBLE, nb::ndim<1>, nb::device::cpu>> boxsize,
    std::optional<const std::string> weight_method,
    bool verbose,
    isa_t isa,
    std::variant<int, std::array<int,3>> grid_refine,
    int max_cells
    ) {

    if (num_threads < 1) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }

    const char *weight_str = weight_method.has_value() ? weight_method->c_str() : NULL;
    config_options options;
    if(get_config_options(&options, weight_str) != EXIT_SUCCESS) {
        throw std::runtime_error(ERRMSG);
    }
    options.numthreads = num_threads;

    if(boxsize.has_value()){
        auto b = boxsize.value().view();
        options.boxsize_x = b(0);
        options.boxsize_y = b(1);
        options.boxsize_z = b(2);

    }
    options.autocorr = !X2.has_value();
    options.verbose = verbose;
    options.instruction_set = isa;
    options.max_cells_per_dim = max_cells;
    
    std::visit([&options](auto&& arg) {
        if constexpr (std::is_same_v<int, std::decay_t<decltype(arg)>>) {
            for(int i=0; i<3; i++) {
                options.grid_refine_factors[i] = arg;
            }
        } else {
            auto arr = arg;
            for(int i=0; i<3; i++) {
                options.grid_refine_factors[i] = arr[i];
            }
        }
        set_grid_refine_scheme(&options, GRIDDING_CUST);
    }, grid_refine);

    int64_t ND1 = X1.shape(0);
    int64_t ND2 = X2.has_value() ? X2->shape(0) : 0;
    int64_t N_bin_edges = bin_edges.shape(0);

    DOUBLE *W1_ptr = W1.has_value() ? W1->data() : nullptr;
    DOUBLE *X2_ptr = X2.has_value() ? X2->data() : nullptr;
    DOUBLE *Y2_ptr = Y2.has_value() ? Y2->data() : nullptr;
    DOUBLE *Z2_ptr = Z2.has_value() ? Z2->data() : nullptr;
    DOUBLE *W2_ptr = W2.has_value() ? W2->data() : nullptr;

    int status = countpairs(
        ND1,
        X1.data(), Y1.data(), Z1.data(), W1_ptr,
        ND2,
        X2_ptr, Y2_ptr, Z2_ptr, W2_ptr,
        N_bin_edges,
        bin_edges.data(), 
        &options,
        npairs.data(),
        ravg.data(),
        wavg.data()
    );

    if(status != EXIT_SUCCESS) {
        // FUTURE if countpairs was using C++, we could easily raise more informative exceptions
        // with proper contextual/chained error messages.
        // For now, we'll use the last message in the ERRMSG buffer.
        throw std::runtime_error(ERRMSG);
    }
}

NB_MODULE(NB_NAME, m) {
    m.def(
        "countpairs",
        &countpairs_wrapper,
        "X1"_a.noconvert(),
        "Y1"_a.noconvert(),
        "Z1"_a.noconvert(),
        "W1"_a.noconvert().none(),
        "X2"_a.noconvert().none(),
        "Y2"_a.noconvert().none(),
        "Z2"_a.noconvert().none(),
        "W2"_a.noconvert().none(),
        "bin_edges"_a.noconvert(),
        "npairs"_a.noconvert(),
        "ravg"_a.noconvert(),
        "wavg"_a.noconvert(),
        "num_threads"_a.noconvert(),
        "boxsize"_a.noconvert().none(),
        "weight_method"_a.noconvert().none(),
        "verbose"_a.noconvert(),
        "isa"_a.noconvert(),
        "grid_refine"_a.noconvert(),
        "max_cells"_a.noconvert()
    );

    nb::enum_<isa_t>(m, "isa_t")
        .value("FASTEST", FASTEST)
        .value("FALLBACK", FALLBACK)
        .value("SSE42", SSE42)
        .value("AVX", AVX)
        .value("AVX512", AVX512)
        .export_values();
}

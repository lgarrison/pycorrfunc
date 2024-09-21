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

template<typename T>
using tuple3 = std::array<T, 3>;

// visitor helper type
template<class... Ts>
struct overloads : Ts... { using Ts::operator()...; };

// deduction guide
template<class... Ts>
overloads(Ts...) -> overloads<Ts...>;

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
    array_t<DOUBLE> weighted_pairs,
    int num_threads,
    std::optional<std::variant<DOUBLE, tuple3<DOUBLE>>> boxsize,
    std::optional<std::string> weight_type,
    bool verbose,
    int isa
    // bin_refine_factors
    ) {

    if (num_threads < 1) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }

    const char *weight_str = weight_type.has_value() ? weight_type->c_str() : NULL;
    config_options options = get_config_options(weight_str);
    options.numthreads = num_threads;
    if(boxsize.has_value()){
        std::visit(
            overloads{
                [&options](const tuple3<DOUBLE> &t) {
                    options.boxsize_x = t[0];
                    options.boxsize_y = t[1];
                    options.boxsize_z = t[2];
                },
                [&options](DOUBLE d) {
                    options.boxsize_x = d;
                    options.boxsize_y = d;
                    options.boxsize_z = d;
                }
            },
            boxsize.value()
        );
    }
    options.autocorr = !X2.has_value();
    options.verbose = verbose;
    options.instruction_set = static_cast<isa_t>(isa);

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
        weighted_pairs.data()
    );

    if(status != EXIT_SUCCESS) {
        // FUTURE if countpairs was using C++, we could easily raise more informative exceptions
        // For now, errors will write their message to stderr, then we'll raise a generic exception
        throw std::runtime_error("countpairs failed");
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
        "weighted_pairs"_a.noconvert(),
        "num_threads"_a.noconvert(),
        "boxsize"_a.noconvert().none(),
        "weight_type"_a.noconvert().none(),
        "verbose"_a.noconvert(),
        "isa"_a.noconvert()
    );
}

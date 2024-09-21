#pragma once

#include <stdint.h>

#include "defs.h"

#ifdef HAVE_AVX512F
#include "avx512_calls.h"
#endif

#ifdef HAVE_AVX
#include "avx_calls.h"
#endif

#ifdef HAVE_SSE42
#include "sse_calls.h"
#endif

int get_num_weights_by_method(const weight_method_t method);
int get_weight_method_by_name(const char *name, weight_method_t *method);

/* Weight function pointer type definitions */

typedef DOUBLE (*weight_func_t)(DOUBLE dx, DOUBLE dy, DOUBLE dz, DOUBLE w0, DOUBLE w1);
#ifdef HAVE_AVX512F
typedef AVX512_FLOATS (*avx512_weight_func_t)(AVX512_FLOATS dx, AVX512_FLOATS dy, AVX512_FLOATS dz, AVX512_FLOATS w0, AVX512_FLOATS w1);
#endif
#ifdef HAVE_AVX
typedef AVX_FLOATS (*avx_weight_func_t)(AVX_FLOATS dx, AVX_FLOATS dy, AVX_FLOATS dz, AVX_FLOATS w0, AVX_FLOATS w1);
#endif
#ifdef HAVE_SSE42
typedef SSE_FLOATS (*sse_weight_func_t)(SSE_FLOATS dx, SSE_FLOATS dy, SSE_FLOATS dz, SSE_FLOATS w0, SSE_FLOATS w1);
#endif

/* Functions to map weight methods to weight functions */

weight_func_t get_weight_func_by_method(const weight_method_t method);

#ifdef __AVX512F__
avx512_weight_func_t get_avx512_weight_func_by_method(const weight_method_t method);
#endif

#ifdef HAVE_AVX
avx_weight_func_t get_avx_weight_func_by_method(const weight_method_t method);
#endif

#ifdef HAVE_SSE42
sse_weight_func_t get_sse_weight_func_by_method(const weight_method_t method);
#endif

/* Functions to compute pair weights from particle weights */

// We'll leave these definitions in the header in hopes of (future) inlining

/*
 * The pair weight is the product of the particle weights
 */
static inline DOUBLE pair_product(DOUBLE, DOUBLE, DOUBLE, DOUBLE w0, DOUBLE w1){
    return w0 * w1;
}

#ifdef __AVX512F__
static inline AVX512_FLOATS avx512_pair_product(AVX512_FLOATS dx, AVX512_FLOATS dy, AVX512_FLOATS dz, AVX512_FLOATS w0, AVX512_FLOATS w1){
    return AVX512_MULTIPLY_FLOATS(w0, w1);
}
#endif

#ifdef HAVE_AVX
static inline AVX_FLOATS avx_pair_product(AVX_FLOATS dx, AVX_FLOATS dy, AVX_FLOATS dz, AVX_FLOATS w0, AVX_FLOATS w1){
    return AVX_MULTIPLY_FLOATS(w0, w1);
}
#endif

#ifdef HAVE_SSE42
static inline SSE_FLOATS sse_pair_product(SSE_FLOATS dx, SSE_FLOATS dy, SSE_FLOATS dz, SSE_FLOATS w0, SSE_FLOATS w1){
    return SSE_MULTIPLY_FLOATS(w0, w1);
}
#endif

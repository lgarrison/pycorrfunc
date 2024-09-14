// # -*- mode: c -*-
#pragma once

#include "defs.h"
#include "weight_defs.h"

#ifdef __AVX512F__
#include "avx512_calls.h"
#endif

#ifdef HAVE_AVX
#include "avx_calls.h"
#endif

#ifdef HAVE_SSE42
#include "sse_calls.h"
#endif

#include <stdint.h>

typedef union {
#ifdef __AVX512F__
  AVX512_FLOATS a512;/* add the bit width for vector register*/
#endif
#ifdef HAVE_AVX
  union {
    AVX_FLOATS a;
    AVX_FLOATS a256;
  };
#endif
#ifdef HAVE_SSE42
    union {
      SSE_FLOATS s;
      SSE_FLOATS s128;
    };
#endif
    DOUBLE d;
} weight_union;

// Info about a particle pair that we will pass to the weight function
typedef struct
{
    weight_union weights0[MAX_NUM_WEIGHTS];
    weight_union weights1[MAX_NUM_WEIGHTS];
    weight_union dx, dy, dz;
    
    // These will only be present for mock catalogs
    weight_union parx, pary, parz;
    
    int64_t num_weights;
} pair_struct;

typedef DOUBLE (*weight_func_t)(const pair_struct*);
#ifdef __AVX512F__
typedef AVX512_FLOATS (*avx512_weight_func_t)(const pair_struct*);
#endif
#ifdef HAVE_AVX
typedef AVX_FLOATS (*avx_weight_func_t)(const pair_struct*);
#endif
#ifdef HAVE_SSE42
typedef SSE_FLOATS (*sse_weight_func_t)(const pair_struct*);
#endif

//////////////////////////////////
// Weighting functions
//////////////////////////////////

/*
 * The pair weight is the product of the particle weights
 */
static inline DOUBLE pair_product(const pair_struct *pair){
    return pair->weights0[0].d*pair->weights1[0].d;
}

#ifdef __AVX512F__
static inline AVX512_FLOATS avx512_pair_product(const pair_struct *pair){
    return AVX512_MULTIPLY_FLOATS(pair->weights0[0].a512, pair->weights1[0].a512);
}
#endif

#ifdef HAVE_AVX
static inline AVX_FLOATS avx_pair_product(const pair_struct *pair){
    return AVX_MULTIPLY_FLOATS(pair->weights0[0].a, pair->weights1[0].a);
}
#endif

#ifdef HAVE_SSE42
static inline SSE_FLOATS sse_pair_product(const pair_struct *pair){
    return SSE_MULTIPLY_FLOATS(pair->weights0[0].s, pair->weights1[0].s);
}
#endif

//////////////////////////////////
// Utility functions
//////////////////////////////////


/* Gives a pointer to the weight function for the given weighting method
 * and instruction set.
 */
static inline weight_func_t get_weight_func_by_method(const weight_method_t method){
    switch(method){
        case PAIR_PRODUCT:
            return &pair_product;
        default:
        case NONE:
            return NULL;
    }
}

#ifdef __AVX512F__
static inline avx512_weight_func_t get_avx512_weight_func_by_method(const weight_method_t method){
    switch(method){
        case PAIR_PRODUCT:
            return &avx512_pair_product;
        default:
        case NONE:
            return NULL;
    }
}
#endif


#ifdef HAVE_AVX
static inline avx_weight_func_t get_avx_weight_func_by_method(const weight_method_t method){
    switch(method){
        case PAIR_PRODUCT:
            return &avx_pair_product;
        default:
        case NONE:
            return NULL;
    }
}
#endif

#ifdef HAVE_SSE42
static inline sse_weight_func_t get_sse_weight_func_by_method(const weight_method_t method){
    switch(method){
        case PAIR_PRODUCT:
            return &sse_pair_product;
        default:
        case NONE:
            return NULL;
    }
}
#endif

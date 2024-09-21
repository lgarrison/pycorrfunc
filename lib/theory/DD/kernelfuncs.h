#pragma once

#include <simdconfig.h>

#define KERNEL_ARGS uint64_t *src_npairs, DOUBLE *src_rpavg, DOUBLE *src_weighted_pairs, \
                   const int64_t N0, DOUBLE *x0, DOUBLE *y0, DOUBLE *z0, DOUBLE *w0, \
                   const int64_t N1, DOUBLE *x1, DOUBLE *y1, DOUBLE *z1, DOUBLE *w1, \
                   const int same_cell, \
                   const DOUBLE sqr_rpmax, const DOUBLE sqr_rpmin, const int nbin, const DOUBLE *rupp_sqr, const DOUBLE rpmax, \
                   const DOUBLE off_xwrap, const DOUBLE off_ywrap, const DOUBLE off_zwrap, \
                   const DOUBLE min_xdiff, const DOUBLE min_ydiff, const DOUBLE min_zdiff, \
                   const DOUBLE closest_icell_xpos, const DOUBLE closest_icell_ypos, const DOUBLE closest_icell_zpos, \
                   const weight_method_t weight_method

typedef int (*kernel_func_ptr)(KERNEL_ARGS);

int countpairs_fallback(KERNEL_ARGS);

#if defined (HAVE_SSE42)
int sse_available(void);
int countpairs_sse_intrinsics(KERNEL_ARGS);
#endif

#if defined (HAVE_AVX)
int avx_available(void);
int countpairs_avx_intrinsics(KERNEL_ARGS);
#endif

//#ifdef HAVE_AVX512F
//int avx512_available(void);
int countpairs_avx512_intrinsics(KERNEL_ARGS);
//#endif

#undef KERNEL_ARGS

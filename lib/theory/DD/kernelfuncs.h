#pragma once

#include <simdconfig.h>

int countpairs_fallback(const int64_t N0, DOUBLE *x0, DOUBLE *y0, DOUBLE *z0, const weight_struct *weights0,
                                             const int64_t N1, DOUBLE *x1, DOUBLE *y1, DOUBLE *z1, const weight_struct *weights1,
                                             const int same_cell,
                                             const DOUBLE sqr_rpmax, const DOUBLE sqr_rpmin, const int nbin, const DOUBLE *rupp_sqr, const DOUBLE rpmax,
                                             const DOUBLE off_xwrap, const DOUBLE off_ywrap, const DOUBLE off_zwrap,
                                             const DOUBLE min_xdiff, const DOUBLE min_ydiff, const DOUBLE min_zdiff,
                                             const DOUBLE closest_icell_xpos, const DOUBLE closest_icell_ypos, const DOUBLE closest_icell_zpos,
                                             DOUBLE *restrict src_rpavg, uint64_t *restrict src_npairs,
                                             DOUBLE *restrict src_weighted_pairs, const weight_method_t weight_method);

#if defined (HAVE_SSE42)
int sse_available(void);
int countpairs_sse_intrinsics(const int64_t N0, DOUBLE *x0, DOUBLE *y0, DOUBLE *z0, const weight_struct *weights0,
                                                   const int64_t N1, DOUBLE *x1, DOUBLE *y1, DOUBLE *z1, const weight_struct *weights1,
                                                   const int same_cell,
                                                   const DOUBLE sqr_rpmax, const DOUBLE sqr_rpmin, const int nbin, const DOUBLE *rupp_sqr, const DOUBLE rpmax,
                                                   const DOUBLE off_xwrap, const DOUBLE off_ywrap, const DOUBLE off_zwrap,
                                                   const DOUBLE min_xdiff, const DOUBLE min_ydiff, const DOUBLE min_zdiff,
                                                   const DOUBLE closest_icell_xpos, const DOUBLE closest_icell_ypos, const DOUBLE closest_icell_zpos,
                                                   DOUBLE *src_rpavg, uint64_t *src_npairs,
                                                   DOUBLE *src_weighted_pairs, const weight_method_t weight_method);
#endif

#if defined (HAVE_AVX)
int avx_available(void);
int countpairs_avx_intrinsics(const int64_t N0, DOUBLE *x0, DOUBLE *y0, DOUBLE *z0, const weight_struct *weights0,
                                                   const int64_t N1, DOUBLE *x1, DOUBLE *y1, DOUBLE *z1, const weight_struct *weights1,
                                                   const int same_cell,
                                                   const DOUBLE sqr_rpmax, const DOUBLE sqr_rpmin, const int nbin, const DOUBLE *rupp_sqr, const DOUBLE rpmax,
                                                   const DOUBLE off_xwrap, const DOUBLE off_ywrap, const DOUBLE off_zwrap,
                                                   const DOUBLE min_xdiff, const DOUBLE min_ydiff, const DOUBLE min_zdiff,
                                                   const DOUBLE closest_icell_xpos, const DOUBLE closest_icell_ypos, const DOUBLE closest_icell_zpos,
                                                   DOUBLE *src_rpavg, uint64_t *src_npairs,
                                                   DOUBLE *src_weighted_pairs, const weight_method_t weight_method);
#endif

//#ifdef HAVE_AVX512F
//int avx512_available(void);
//void increment_avx2(float arr[4]);
//#endif
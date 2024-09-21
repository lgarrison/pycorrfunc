#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "function_precision.h"
#include "utils.h"

#include "weights.h"
#include "kernelfuncs.h"


#if defined (HAVE_SSE42)
#include "sse_calls.h"

#ifdef _MSC_VER
#include<intrin.h>
int sse_available(void) {
  return 1;
}
#else

#include<xmmintrin.h>
#include<cpuid.h>
#include<stdint.h>

#if defined(__APPLE__)
int sse_available(void) { return 1; }
#else
int sse_available(void) {
    return __builtin_cpu_supports("sse4.2");
}
#endif
#endif

int countpairs_sse_intrinsics(const int64_t N0, DOUBLE *x0, DOUBLE *y0, DOUBLE *z0, const weight_struct *weights0,
                                                   const int64_t N1, DOUBLE *x1, DOUBLE *y1, DOUBLE *z1, const weight_struct *weights1,
                                                   const int same_cell,
                                                   const DOUBLE sqr_rpmax, const DOUBLE sqr_rpmin, const int nbin, const DOUBLE *rupp_sqr, const DOUBLE rpmax,
                                                   const DOUBLE off_xwrap, const DOUBLE off_ywrap, const DOUBLE off_zwrap,
                                                   const DOUBLE min_xdiff, const DOUBLE min_ydiff, const DOUBLE min_zdiff,
                                                   const DOUBLE closest_icell_xpos, const DOUBLE closest_icell_ypos, const DOUBLE closest_icell_zpos,
                                                   DOUBLE *src_rpavg, uint64_t *src_npairs,
                                                   DOUBLE *src_weighted_pairs, const weight_method_t weight_method)
{
  uint64_t npairs[nbin];
  for(int i=0;i<nbin;i++) {
    npairs[i] = 0;
  }

  SSE_FLOATS m_rupp_sqr[nbin];
  for(int i=0;i<nbin;i++) {
    m_rupp_sqr[i] = SSE_SET_FLOAT(rupp_sqr[i]);
  }

  const int32_t need_rpavg = src_rpavg != NULL;
  const int32_t need_weighted_pairs = src_weighted_pairs != NULL;
  SSE_FLOATS m_kbin[nbin];
  DOUBLE rpavg[nbin], weighted_pairs[nbin];
  if(need_rpavg || need_weighted_pairs){
    for(int i=0;i<nbin;i++) {
        m_kbin[i] = SSE_SET_FLOAT((DOUBLE) i);
        if(need_rpavg) {
            rpavg[i] = ZERO;
        }
        if(need_weighted_pairs){
            weighted_pairs[i] = ZERO;
        }
    }
  }

  // A copy whose pointers we can advance
  weight_struct local_w0 = {.weights={NULL}, .num_weights=0}, local_w1 = {.weights={NULL}, .num_weights=0};
  pair_struct pair = {.num_weights=0};
  sse_weight_func_t sse_weight_func = NULL;
  weight_func_t fallback_weight_func = NULL;
  if(need_weighted_pairs){
      // Same particle list, new copy of num_weights pointers into that list
      local_w0 = *weights0;
      local_w1 = *weights1;

      pair.num_weights = local_w0.num_weights;

      sse_weight_func = get_sse_weight_func_by_method(weight_method);
      fallback_weight_func = get_weight_func_by_method(weight_method);
  }

  const DOUBLE *zstart = z1, *zend = z1 + N1;
  const DOUBLE max_all_dz = SQRT(rpmax*rpmax - min_xdiff*min_xdiff - min_ydiff*min_ydiff);
  for(int64_t i=0;i<N0;i++) {
      const DOUBLE xpos = *x0++ + off_xwrap;
      const DOUBLE ypos = *y0++ + off_ywrap;
      const DOUBLE zpos = *z0++ + off_zwrap;
      for(int w = 0; w < pair.num_weights; w++){
          // local_w0.weights[w] is a pointer to a float in the particle list of weights,
          // just as x0 is a pointer into the list of x-positions.
          // The advancement of the local_w0.weights[w] pointer should always mirror x0.
          pair.weights0[w].s = SSE_SET_FLOAT(*local_w0.weights[w]++);
      }
      DOUBLE max_dz = max_all_dz;

      /* Now consider if this i'th particle can be a valid pair with ANY of the remaining
         j' particles. The min. difference in the z-positions between this i'th particle and ANY
         of the remaining j'th particles, is the difference between the current j'th particle and
         the current i'th particle (since all remaining j'th particles will have a larger value for
         the z-ordinate and therefore a larger difference to zpos). if this `dz` does not satisfy
         the distance criteria, then NO remaining j'th particles will. Continue on to the next i'th
         particle
      */
      const DOUBLE this_dz = *z1 - zpos;
      if(this_dz >= max_all_dz) {
          continue;
      }

      /* Okay so there MAY be a pair */
      if(same_cell == 1) {
          z1++;
      } else {
          // Now add the x,y information to further limit the range of secondaries for this particle
          // But note this constraint may increase or decrease for the next particle, since x,y aren't sorted!
          const DOUBLE min_dx = min_xdiff > 0 ? min_xdiff + FABS(xpos - closest_icell_xpos):min_xdiff;
          const DOUBLE min_dy = min_ydiff > 0 ? min_ydiff + FABS(ypos - closest_icell_ypos):min_ydiff;
          const DOUBLE min_dz = min_zdiff > 0 ? (this_dz > 0 ? this_dz:min_zdiff + FABS(zpos - closest_icell_zpos)):min_zdiff;
          const DOUBLE sqr_min_sep_this_point = min_dx*min_dx + min_dy*min_dy + min_dz*min_dz;
          if(sqr_min_sep_this_point >= sqr_rpmax) {
              continue;
          }
          max_dz = SQRT(sqr_rpmax - min_dx*min_dx - min_dy*min_dy);

          // Now "fast forward" in the list of secondary particles to find the first one that satisfies the max_all_dz constraint
          // We don't consider the i particle's x,y information yet, because those aren't sorted
          const DOUBLE target_z = zpos - max_all_dz;
          while(z1 < zend && *z1 <= target_z) {
              z1++;
          }
      }
      // If no j particle satisfies the constraint for this i particle,
      // then the same holds true for all future i particles because they are sorted in increasing z order
      if(z1 == zend) {
          i = N0;
          break;
      }

      DOUBLE *localz1 = z1;
      const DOUBLE target_z = zpos - max_dz;
      while(localz1 != zend && *localz1 <= target_z) {
          localz1++;
      }

      int64_t j = localz1 - zstart;
      DOUBLE *localx1 = x1 + j;
      DOUBLE *localy1 = y1 + j;
      for(int w = 0; w < local_w1.num_weights; w++){
          local_w1.weights[w] = weights1->weights[w] + j;
      }

      for(;j<=(N1 - SSE_NVEC);j+=SSE_NVEC){
          union int4 union_rpbin;
          union float4 union_mDperp;
          union float4_weights union_mweight;

          const SSE_FLOATS m_xpos = SSE_SET_FLOAT(xpos);
          const SSE_FLOATS m_ypos = SSE_SET_FLOAT(ypos);
          const SSE_FLOATS m_zpos = SSE_SET_FLOAT(zpos);

          const SSE_FLOATS m_x1 = SSE_LOAD_FLOATS_UNALIGNED(localx1);
          const SSE_FLOATS m_y1 = SSE_LOAD_FLOATS_UNALIGNED(localy1);
          const SSE_FLOATS m_z1 = SSE_LOAD_FLOATS_UNALIGNED(localz1);

          localx1 += SSE_NVEC;
          localy1 += SSE_NVEC;
          localz1 += SSE_NVEC;

          for(int w = 0; w < pair.num_weights; w++){
              pair.weights1[w].s = SSE_LOAD_FLOATS_UNALIGNED(local_w1.weights[w]);
              local_w1.weights[w] += SSE_NVEC;
          }

          const SSE_FLOATS m_max_dz = SSE_SET_FLOAT(max_dz);
          const SSE_FLOATS m_sqr_rpmax = SSE_SET_FLOAT(sqr_rpmax);
          const SSE_FLOATS m_sqr_rpmin = SSE_SET_FLOAT(sqr_rpmin);

          const SSE_FLOATS m_xdiff = SSE_SUBTRACT_FLOATS(m_x1, m_xpos);  //(x[j] - x0)
          const SSE_FLOATS m_ydiff = SSE_SUBTRACT_FLOATS(m_y1, m_ypos);  //(y[j] - y0)
          const SSE_FLOATS m_zdiff = SSE_SUBTRACT_FLOATS(m_z1, m_zpos);  //z2[j:j+NVEC-1] - z1

          const SSE_FLOATS m_mask_geq_pimax = SSE_COMPARE_FLOATS_GE(m_zdiff,m_max_dz);
          if(SSE_TEST_COMPARISON(m_mask_geq_pimax) > 0) {
              j = N1;//but do not break yet, there might be pairs in this chunk
          }

          const SSE_FLOATS m_sqr_xdiff = SSE_SQUARE_FLOAT(m_xdiff);
          const SSE_FLOATS m_sqr_ydiff = SSE_SQUARE_FLOAT(m_ydiff);
          const SSE_FLOATS m_sqr_zdiff = SSE_SQUARE_FLOAT(m_zdiff);

          if(need_weighted_pairs){
              pair.dx.s = m_xdiff;
              pair.dy.s = m_ydiff;
              pair.dz.s = m_zdiff;
          }

          SSE_FLOATS r2  = SSE_ADD_FLOATS(m_sqr_zdiff,SSE_ADD_FLOATS(m_sqr_xdiff, m_sqr_ydiff));
          SSE_FLOATS m_mask_left;
          {
              const SSE_FLOATS m_rpmin_mask = SSE_COMPARE_FLOATS_GE(r2, m_sqr_rpmin);
              const SSE_FLOATS m_rpmax_mask = SSE_COMPARE_FLOATS_LT(r2,m_sqr_rpmax);
              m_mask_left = SSE_BITWISE_AND(m_rpmin_mask, m_rpmax_mask);
              if(SSE_TEST_COMPARISON(m_mask_left) == 0) {
                  continue;
              }
              r2 = SSE_BLEND_FLOATS_WITH_MASK(m_sqr_rpmax, r2, m_mask_left);
          }

          SSE_FLOATS m_rpbin = SSE_SETZERO_FLOAT();
          if(need_rpavg) {
              union_mDperp.m_Dperp = SSE_SQRT_FLOAT(r2);
          }
          if(need_weighted_pairs){
              union_mweight.m_weights = sse_weight_func(&pair);
          }

          for(int kbin=nbin-1;kbin>=1;kbin--) {
              SSE_FLOATS m1 = SSE_COMPARE_FLOATS_GE(r2,m_rupp_sqr[kbin-1]);
              SSE_FLOATS m_bin_mask = SSE_BITWISE_AND(m1,m_mask_left);
              m_mask_left = SSE_COMPARE_FLOATS_LT(r2,m_rupp_sqr[kbin-1]);
              int test2  = SSE_TEST_COMPARISON(m_bin_mask);
              npairs[kbin] += SSE_BIT_COUNT_INT(test2);
              if(need_rpavg || need_weighted_pairs){
                  m_rpbin = SSE_BLEND_FLOATS_WITH_MASK(m_rpbin,m_kbin[kbin], m_bin_mask);
              }
              int test3 = SSE_TEST_COMPARISON(m_mask_left);
              if(test3 == 0) {
                  break;
              }
          }

          if(need_rpavg || need_weighted_pairs) {
              union_rpbin.m_ibin = SSE_TRUNCATE_FLOAT_TO_INT(m_rpbin);
              //protect the unroll pragma in case compiler is not icc.
#if  __INTEL_COMPILER
#pragma unroll(SSE_NVEC)
#endif
              for(int jj=0;jj<SSE_NVEC;jj++) {
                  const int kbin = union_rpbin.ibin[jj];
                  if(need_rpavg){
                      const DOUBLE r = union_mDperp.Dperp[jj];
                      rpavg[kbin] += r;
                  }
                  if(need_weighted_pairs){
                      const DOUBLE weight = union_mweight.weights[jj];
                      weighted_pairs[kbin] += weight;
                  }
              }
          } //rpavg
      }

      for(;j<N1;j++) {
          const DOUBLE dx = *localx1++ - xpos;
          const DOUBLE dy = *localy1++ - ypos;
          const DOUBLE dz = *localz1++ - zpos;
          for(int w = 0; w < pair.num_weights; w++){
              pair.weights1[w].d = *local_w1.weights[w]++;
          }

          if(dz >= max_dz) break;

          const DOUBLE r2 = dx*dx + dy*dy + dz*dz;
          if(r2 >= sqr_rpmax || r2 < sqr_rpmin) continue;

          if(need_weighted_pairs){
              pair.dx.d = dx;
              pair.dy.d = dy;
              pair.dz.d = dz;
          }

          DOUBLE r = ZERO, pairweight = ZERO;
          if(need_rpavg) {
              r = SQRT(r2);
          }
          if(need_weighted_pairs){
              pairweight = fallback_weight_func(&pair);
          }

          for(int kbin=nbin-1;kbin>=1;kbin--){
              if(r2 >= rupp_sqr[kbin-1]) {
                  npairs[kbin]++;
                  if(need_rpavg){
                      rpavg[kbin] += r;
                  }
                  if(need_weighted_pairs){
                      weighted_pairs[kbin] += pairweight;
                  }
                  break;
              }
          }//searching for kbin loop
      }//loop over remnant second set of particles
  }//loop over first set of particles

  for(int i=1;i<nbin;i++) {
      src_npairs[i - 1] += npairs[i];
      if(need_rpavg) {
          src_rpavg[i - 1] += rpavg[i];
      }
      if(need_weighted_pairs) {
          src_weighted_pairs[i - 1] += weighted_pairs[i];
      }
  }

  return EXIT_SUCCESS;
}
#endif //HAVE_SSE42
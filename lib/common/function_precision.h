#pragma once

#ifndef M_PI
#define M_PI            3.14159265358979323846264338327950288
#endif

#define PI_OVER_180       0.017453292519943295769236907684886127134428718885417254560971
#define INV_PI_OVER_180   57.29577951308232087679815481410517033240547246656432154916024

//Define the Macros
#ifdef PYCORRFUNC_USE_DOUBLE
#define COSD(X)            cos(X*PI_OVER_180)
#define SIND(X)            sin(X*PI_OVER_180)
#else
#define COSD(X)            cosf(X*PI_OVER_180)
#define SIND(X)            sinf(X*PI_OVER_180)
#endif

#ifdef HAVE_AVX
    
#define REGISTER_WIDTH 256  //cpu supports avx instructions
#define NVECF  8  //8 floats per ymm register
#define NVECD  4  //4 doubles per ymm register

#else

#ifdef HAVE_SSE42
    
#define REGISTER_WIDTH 128  //cpu supports sse instructions
#define NVECF  4  //8 floats per xmm register
#define NVECD  2  //4 doubles per xmm register

#else

#define REGISTER_WIDTH 64
#define NVECF 2
#define NVECD 1

#endif//SSE
#endif//AVX


#include <float.h>
    
#ifdef PYCORRFUNC_USE_DOUBLE
#define REAL_FORMAT "lf"
#define NVEC   NVECD
#define ZERO   0.0
#define SQRT   sqrt
#define LOG    log
#define LOG10  log10
#define LOG2   log2
#define FABS   fabs
#define COS    cos
#define SIN    sin
#define ACOS   acos
#define ASIN   asin
#define POW    pow
#define ABS    fabs
#define FMIN   fmin
#define FMAX   fmax
#define MAX_DOUBLE DBL_MAX
#else
#define REAL_FORMAT "f"
#define NVEC   NVECF
#define ZERO   0.0f
#define SQRT   sqrtf
#define LOG    logf
#define LOG10  log10f
#define LOG2   log2f
#define FABS   fabsf
#define COS    cosf
#define SIN    sinf
#define ACOS   acosf
#define ASIN   asinf
#define POW    powf
#define ABS    fabsf
#define FMIN   fminf
#define FMAX   fmaxf
#define MAX_DOUBLE FLT_MAX
#endif
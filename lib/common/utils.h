#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/types.h>

//general utilities
extern int my_snprintf(char *buffer,int len,const char *format, ...) __attribute__((format(printf,3,4)));
extern char * get_time_string(struct timeval t0,struct timeval t1);
extern void print_time(struct timeval t0,struct timeval t1,const char *s);
extern void current_utc_time(struct timespec *ts);

//memory routines
extern void* my_malloc(size_t size,int64_t N);
extern void* my_calloc(size_t size,int64_t N);
extern void my_free(void ** x);
extern void **matrix_malloc(size_t size,int64_t nx,int64_t ny);
extern void **matrix_calloc(size_t size,int64_t nx,int64_t ny);
extern void matrix_free(void **m,int64_t ny);
    
extern void parallel_cumsum(const int64_t *a, const int64_t N, int64_t *cumsum);

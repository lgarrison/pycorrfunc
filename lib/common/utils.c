#include <inttypes.h>//defines PRId64 for printing int64_t + includes stdint.h
#include <math.h>
#include <string.h>
#include <limits.h>
#include <stdarg.h>
#include <ctype.h>
#include <time.h>

#include "macros.h"
#include "utils.h"

#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
#include <mach/mach_time.h> /* mach_absolute_time -> really fast */
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// A real wrapper to snprintf that will exit() if the allocated buffer length
// was not sufficient. Usage is the same as snprintf

int my_snprintf(char *buffer,int len,const char *format, ...)
{
    va_list args;
    int nwritten=0;

    va_start(args,format);
    nwritten=vsnprintf(buffer, (size_t) len, format, args );
    va_end(args);
    if (nwritten > len || nwritten < 0) {
        fprintf(stderr,"ERROR: printing to string failed (wrote %d characters while only %d characters were allocated)\n",nwritten,len);
        fprintf(stderr,"Increase `len'=%d in the header file\n",len);
        return -1;
    }
    return nwritten;
}


/*
Can not remember where I (MS) got this from. Fairly sure
stackoverflow was involved.
Finally taken from http://stackoverflow.com/a/6719178/2237582 */
void current_utc_time(struct timespec *ts)
{

#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    static mach_timebase_info_data_t    sTimebaseInfo = {.numer=0, .denom=0};
    uint64_t start = mach_absolute_time();
    if ( sTimebaseInfo.denom == 0 ) {
        mach_timebase_info(&sTimebaseInfo);
    }

    ts->tv_sec = 0;//(start * sTimebaseInfo.numer/sTimebaseInfo.denom) * tv_nsec;
    ts->tv_nsec = start * sTimebaseInfo.numer / sTimebaseInfo.denom;

#if 0
    //Much slower implementation for clock
    //Slows down the code by up to 4x
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts->tv_sec = mts.tv_sec;
    ts->tv_nsec = mts.tv_nsec;
#endif

#else
    clock_gettime(CLOCK_REALTIME, ts);
#endif
}



/*
  I like this particular function. Generic replacement for printing
  (in meaningful units) the actual execution time of a code/code segment.

  The function call should be like this:

  ---------------------------
  struct timeval t_start,t_end;
  gettimeofday(&t_start,NULL);
  do_something();
  gettimeofday(&t_end,NULL);
  print_time(t_start,t_end,"do something");
  ---------------------------

  if the code took 220 mins 30.1 secs
  -> print_time will output `Time taken to execute `do something' = 3 hours 40 mins 30.1 seconds


  (code can be easily extended to include `weeks' as a system of time unit. left to the reader)
*/


char * get_time_string(struct timeval t0,struct timeval t1)
{
  const size_t MAXLINESIZE = 1024;
  char *time_string = my_malloc(sizeof(char), MAXLINESIZE);
  double timediff = t1.tv_sec - t0.tv_sec;
  double ratios[] = {24*3600.0,  3600.0,  60.0,  1};

  if(timediff < ratios[2]) {
      my_snprintf(time_string, MAXLINESIZE,"%6.3lf secs",1e-6*(t1.tv_usec-t0.tv_usec) + timediff);
  }  else {
      double timeleft = timediff;
      size_t curr_index = 0;
      int which = 0;
      while (which < 4) {
          double time_to_print = floor(timeleft/ratios[which]);
          if (time_to_print > 1) {
              timeleft -= (time_to_print*ratios[which]);
              char units[4][10]  = {"days", "hrs" , "mins", "secs"};
              char tmp[MAXLINESIZE];
              my_snprintf(tmp, MAXLINESIZE, "%5d %s",(int)time_to_print,units[which]);
              const size_t len = strlen(tmp);
#ifndef NDEBUG
              const size_t required_len = curr_index + len + 1;
              XRETURN(MAXLINESIZE >= required_len, NULL,
                      "buffer overflow will occur: string has space for %zu bytes while concatenating requires at least %zu bytes\n",
                      MAXLINESIZE, required_len);
#endif
              strcpy(time_string + curr_index, tmp);
              curr_index += len;
          }
          which++;
      }
  }

  return time_string;
}

void print_time(struct timeval t0,struct timeval t1,const char *s)
{
    double timediff = t1.tv_sec - t0.tv_sec;
    double ratios[] = {24*3600.0,  3600.0,  60.0,  1};
    fprintf(stderr,"Time taken to execute '%s'  = ",s);
    if(timediff < ratios[2]) {
        fprintf(stderr,"%6.3lf secs",1e-6*(t1.tv_usec-t0.tv_usec) + timediff);
    }  else {
        double timeleft = timediff;
        int which = 0;
        while (which < 4) {
            double time_to_print = floor(timeleft/ratios[which]);
            if (time_to_print > 1) {
                char units[4][10]  = {"days", "hrs" , "mins", "secs"};
                timeleft -= (time_to_print*ratios[which]);
                fprintf(stderr,"%5d %s",(int)time_to_print,units[which]);
            }
            which++;
        }
    }
    fprintf(stderr,"\n");
}


void* my_malloc(size_t size,int64_t N)
{
    void *x = malloc(N*size);
    if (x==NULL){
        fprintf(stderr,"malloc for %"PRId64" elements with %zu bytes failed...\n",N,size);
        perror(NULL);
    }

    // poison
    // memset(x, 0x5a, N*size);

    return x;
}



void* my_calloc(size_t size,int64_t N)
{
    void *x = calloc((size_t) N, size);
    if (x==NULL)    {
        fprintf(stderr,"malloc for %"PRId64" elements with %zu size failed...\n",N,size);
        perror(NULL);
    }

    return x;
}



//real free. Use only if you are going to check the
//pointer variable afterwards for NULL.
void my_free(void ** x)
{
    /* my_free(void *x) would also free the
       memory but then x would be a local variable
       and the pointer itself in the calling routine
       could not be set to NULL. Hence the pointer
       to pointer business.
    */

    if(*x!=NULL)
        free(*x);//free the memory

    *x=NULL;//sets the pointer in the calling routine to NULL.
}


void **matrix_malloc(size_t size,int64_t nrow,int64_t ncol)
{
    void **m = (void **) my_malloc(sizeof(void *),nrow);
    if(m == NULL) {
        return NULL;
    }

    for(int i=0;i<nrow;i++) {
        m[i] = (void *) my_malloc(size,ncol);
        /* Check if allocation failed */
        if(m[i] == NULL) {
            /* Free up all the memory allocated so far */
            for(int j=i-1;j>=0;j--) {
                free(m[j]);
            }
            free(m);
            return NULL;
        }
    }

    return m;
}

void **matrix_calloc(size_t size,int64_t nrow,int64_t ncol)
{
    void **m = (void **) my_calloc(sizeof(void *),nrow);
    if(m == NULL) {
        return m;
    }
    for(int i=0;i<nrow;i++) {
        m[i] = (void *) my_calloc(size,ncol);
        /* Check if allocation failed */
        if(m[i] == NULL) {
            /* Free up all the memory allocated so far */
            for(int j=i-1;j>=0;j--) {
                free(m[j]);
            }
            free(m);
            return NULL;
        }
    }

    return m;
}


void matrix_free(void **m,int64_t nrow)
{
    if(m == NULL)
        return;

    for(int i=0;i<nrow;i++)
        free(m[i]);

    free(m);
}

/* A parallel cumulative sum
   Output convention is: cumsum[0] = 0; cumsum[N-1] = sum(a[0:N-1]);
   The algorithm is:
   - Divide the array into `nthreads` chunks
   - cumsum within each chunk
   - compute the "offset" for each chunk by summing the cumsum at the tail of all previous chunks
   - apply the offset
*/
void parallel_cumsum(const int64_t *a, const int64_t N, int64_t *cumsum){
    if (N <= 0){
        return;  // nothing to do
    }
    
    #ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    #else
    int nthreads = 1;
    #endif
    
    // We will heuristically limit the number of threads
    // if there isn't enough work for multithreading to be efficient.
    // This is also important for the correctness of the algorithm below,
    // since it enforces nthreads <= N
    int64_t min_N_per_thread = 10000;
    if(N/min_N_per_thread < nthreads){
        nthreads = N/min_N_per_thread;
    }
    if(nthreads < 1){
        nthreads = 1;
    }
    
    #ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
    #endif
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif
        
        int64_t cstart = N*tid/nthreads;
        int64_t cend = N*(tid+1)/nthreads;
        cumsum[cstart] = cstart > 0 ? a[cstart-1] : 0;
        for(int64_t c = cstart+1; c < cend; c++){
            cumsum[c] = a[c-1] + cumsum[c-1];
        }
        
        #ifdef _OPENMP
        #pragma omp barrier
        #endif
        
        int64_t offset = 0;
        for(int t = 0; t < tid; t++){
            offset += cumsum[N*(t+1)/nthreads-1];
        }
        
        #ifdef _OPENMP
        #pragma omp barrier
        #endif
        
        if(offset != 0){
            for(int64_t c = cstart; c < cend; c++){
                cumsum[c] += offset;
            }
        }
    }
}

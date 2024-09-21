#pragma once

#define NLATMAX           100

#define BOOST_CELL_THRESH    10
#define BOOST_NUMPART_THRESH 250
#define BOOST_BIN_REF        1

//what fraction of particles have to be sorted
//to switch from quicksort to heapsort
#define FRACTION_SORTED_REQD_TO_HEAP_SORT   0.6

#define ADD_DIFF_TIME(t0,t1)            ((t1.tv_sec - t0.tv_sec) + 1e-6*(t1.tv_usec - t0.tv_usec))
#define REALTIME_ELAPSED_NS(t0, t1)     ((t1.tv_sec - t0.tv_sec)*1000000000.0 + (t1.tv_nsec - t0.tv_nsec))

#define ALIGNMENT                32

#define MAX(a,b) \
   ({ typeof (a) _a = (a); \
       typeof (b) _b = (b); \
     _a > _b ? _a : _b; })

#define MIN(a,b) \
    ({ typeof (a) _a = (a); \
         typeof (b) _b = (b); \
      _a < _b ? _a : _b; })


#define ASSIGN_CELL_TIMINGS(thread_timings, nx1, nx2, timediff, tid, first_cellid, second_cellid) \
    {                                                                   \
        thread_timings->N1 = nx1;                                       \
        thread_timings->N2 = nx2;                                       \
        thread_timings->time_in_ns = timediff;                          \
        thread_timings->tid = tid;                                      \
        thread_timings->first_cellindex = first_cellid;                 \
        thread_timings->second_cellindex = second_cellid;               \
    }

/* Macro Constants */
//Just to output some colors
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_RESET   "\x1b[0m"


#define PI_UNICODE    "\u03C0"
#define XI_UNICODE    "\u03BE"
/* #define PIMAX_UNICODE PI_UNICODE"\u2098""\u2090""\u2093" */
/* #define RP_UNICODE    "\u209A" */
#define PIMAX_UNICODE "pimax"
#define RP_UNICODE    "rp"
#define THETA_UNICODE "\u03B8"
#define OMEGA_UNICODE "\u03C9"
#define MU_UNICODE    "\u03BC"

#define PI_SAFE    "pi"
#define XI_SAFE    "xi"
#define PIMAX_SAFE "pimax"
#define RP_SAFE "rp"
#define THETA_SAFE "theta"
#define OMEGA      "omega"
#define MU_SAFE    "mu"


#ifdef USE_UNICODE
#define PI_CHAR PI_UNICODE
#define XI_CHAR XI_UNICODE
#define PIMAX_CHAR PIMAX_UNICODE
#define RP_CHAR  RP_UNICODE
#define MU_CHAR  MU_UNICODE
#define THETA_CHAR THETA_UNICODE
#define OMEGA_CHAR OMEGA_UNICODE
#else
#define PI_CHAR PI_SAFE
#define XI_CHAR XI_SAFE
#define MU_CHAR MU_SAFE
#define PIMAX_CHAR PIMAX_SAFE
#define RP_CHAR    RP_SAFE
#define THETA_CHAR THETA_SAFE
#define OMEGA_CHAR OMEGA_SAFE
#endif

/* Function-like macros */
#ifdef NDEBUG
#define XASSERT(EXP, ...)                                do{} while(0)
#else
#define XASSERT(EXP, ...)                                               \
     do { if (!(EXP)) {                                                 \
             fprintf(stderr,"An internal error has occurred: %s\tfunc: %s\tline: %d with expression `"#EXP"'\n", __FILE__, __FUNCTION__, __LINE__); \
             fprintf(stderr,ANSI_COLOR_BLUE "Please file an issue on GitHub." ANSI_COLOR_RESET "\n"); \
             return EXIT_FAILURE;                                       \
         }                                                              \
     } while (0)
#endif

#ifdef NDEBUG
#define XPRINT(EXP, ...)                                do{} while(0)
#else
#define XPRINT(EXP, ...)                                               \
     do { if (!(EXP)) {                                                 \
             fprintf(stderr,"An internal error has occurred: %s\tfunc: %s\tline: %d with expression `"#EXP"'\n", __FILE__, __FUNCTION__, __LINE__); \
             fprintf(stderr,__VA_ARGS__);                               \
             fprintf(stderr,ANSI_COLOR_BLUE "Please file an issue on GitHub." ANSI_COLOR_RESET "\n"); \
         }                                                              \
     } while (0)
#endif


#ifdef NDEBUG
#define XRETURN(EXP, VAL, ...)                                do{} while(0)
#else
#define XRETURN(EXP, VAL, ...)                                           \
     do { if (!(EXP)) {                                                 \
             fprintf(stderr,"An internal error has occurred: %s\tfunc: %s\tline: %d with expression `"#EXP"'\n", __FILE__, __FUNCTION__, __LINE__); \
             fprintf(stderr,__VA_ARGS__);                               \
             fprintf(stderr,ANSI_COLOR_BLUE "Please file an issue on GitHub." ANSI_COLOR_RESET "\n"); \
             return VAL;                                                \
         }                                                              \
     } while (0)
#endif

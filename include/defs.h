#ifndef DEFS_H
#define DEFS_H

#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PYCORRFUNC_USE_DOUBLE
typedef double DOUBLE;
#else
typedef float DOUBLE;
#endif

#ifdef PYCORRFUNC_USE_DOUBLEACCUM
typedef double DoubleAccum;
#else
// otherwise default to native
typedef DOUBLE DoubleAccum;
#endif

extern char ERRMSG[1024];

#include "macros.h"

#define QUOTE(name) #name
#define STR(macro)  QUOTE(macro)

/* Macros as mask for the gridding_flags */
/* These constitute the 32 bytes for
the ``uint32_t gridding_flags`` */

#define GRIDDING_REF_MASK \
    0x0000000F  // Last 4 bits for how the grid sizes are calculated. Also indicates if
                // refines are in place
#define GRIDDING_ORD_MASK                                             \
    0x000000F0  // Next 4 bits for how the 3-D-> 1-D index conversion
/* The upper 24 bits are unused currently */

#define GRIDDING_DFL  0x0
#define GRIDDING_CUST 0x1

typedef struct {
    int64_t N1;         /* Number of points in the first cell */
    int64_t N2;         /* Number of points in the second cell */
    int64_t time_in_ns; /* Time taken in the compute kernel, measured in nano-seconds */
    int first_cellindex;
    int second_cellindex;
    int tid; /* Thread-id, 0 for serial case, wastes 4 bytes, since thread id is 4 bytes
                integer and not 8 bytes */
} api_cell_timings;

#define MAX_FAST_DIVIDE_NR_STEPS 3

typedef enum {
    NONE         = 0, /* default */
    PAIR_PRODUCT = 1,
} weight_method_t;

typedef enum {
    FASTEST  = -1,
    FALLBACK = 0,
    SSE      = 1,
    SSE2     = 2,
    SSE3     = 3,
    SSSE3    = 4,
    SSE4     = 5,
    SSE42    = 6,
    AVX      = 7,
    AVX2     = 8,
    AVX512   = 9,
    ARM64    = 10
} isa_t;

typedef struct {
    /* Theory option for periodic boundaries */
    double boxsize_x;
    double boxsize_y;
    double boxsize_z;

    /* Per cell timers. Keeps track of the number of particles per cell pair
       and time spent to compute the pairs. Might slow down code */
    api_cell_timings *cell_timings;
    int64_t totncells_timings;

    isa_t instruction_set; /* select instruction set to run on */

    uint8_t verbose; /* Outputs progressbar and times */
    uint8_t
       c_cell_timer; /* Measures time spent per cell-pair. Might slow down the code */

    /* Options valid for both theory and mocks */
    uint8_t need_avg_sep; /* <rp> or <\theta> is required */
    uint8_t autocorr;     /* Only one dataset is required */

    /* the link_in_* variables control how the 3-D cell structure is created */
    uint8_t link_in_dec; /* relevant for DDtheta_mocks */
    uint8_t link_in_ra;  /* relevant for DDtheta_mocks. */

    /* Replaces the divide in DDrppi_mocks in AVX mode by a reciprocal and a
     * Newton-Raphson step. */
    uint8_t fast_divide_and_NR_steps; /* Used in AVX512/AVX; if set to 0, the standard
                                         (slow) divide is used If > 0, the value is
                                         interpreted as the number of NR steps i.e.,
                                         fast_divide_and_NR_steps = 2, performs two
                                         steps of Newton-Raphson Anything greater than
                                         ~5, probably makes the code slower than the
                                         divide without any improvement in precision
                                       */

    /* Fast arccos for wtheta (effective only when OUTPUT_THETAAVG is enabled) */
    uint8_t fast_acos;

    /* Enabled by default */
    int8_t grid_refine_factors[3]; /* Array for the custom grid refine factors in each
                                     dim xyz for theory routines and ra/dec/cz for mocks
                                     Must be signed integers since some for loops might
                                     use -grid_refine_factor as the starting point */

    uint16_t max_cells_per_dim; /* max number of cells per dimension. same for both
                                   theory and mocks */

    union {
        uint32_t gridding_flags; /* flag for all linking features,
                                   Will contain OR'ed flags from enum from
                                   `gridding_scheme` Intentionally set as unsigned int,
                                   since in the future we might want to support some
                                   bit-wise OR'ed functionality */
        uint8_t grid_masks[4];
    };

    weight_method_t weight_method;  // the function that will get called to give the
                                    // weight of a particle pair

    /* Additional fields */
    int numthreads; /* Number of threads */

} config_options;

void set_grid_refine_scheme(config_options *options, const int8_t flag);
void reset_grid_refine_scheme(config_options *options);
int8_t get_grid_refine_scheme(const config_options *options);
void set_grid_refine_factors(config_options *options, const int grid_refine_factors[3]);
void set_custom_grid_refine_factors(config_options *options,
                                    const int grid_refine_factors[3]);
void reset_grid_refine_factors(config_options *options);

int get_config_options(config_options *options, const char *weight_method);

void free_cell_timings(config_options *options);

void allocate_cell_timer(config_options *options, const int64_t num_cell_pairs);

void assign_cell_timer(api_cell_timings *cell_timings,
                       const int64_t num_cell_pairs,
                       config_options *options);
#endif

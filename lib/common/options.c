#include <stdio.h>

#include "defs.h"
#include "weights.h"

config_options get_config_options(const char *weight_method){
    config_options options;
    memset(&options, 0, sizeof(options));

    if(get_weight_method_by_name(weight_method, &options.weight_method) != EXIT_SUCCESS){
        // TODO: we should be able to raise proper Python exceptions via nanobind
        // Although arguably, most checks that could raise an exception should be done in Python
        fprintf(stderr,"Error: Unknown weight method `%s'\n", weight_method);
        exit(EXIT_FAILURE);
    }

    // A value of <= 0 is non-periodic, > 0 is periodic.
    options.boxsize_x = 0.;
    options.boxsize_y = 0.;
    options.boxsize_z = 0.;

    options.verbose = 1;
    options.need_avg_sep = 1;

#ifdef __AVX512F__
    options.instruction_set = AVX512F;
#elif defined(__AVX2__)
    options.instruction_set = AVX2;
#elif defined(HAVE_AVX)
    options.instruction_set = AVX;
#elif defined(HAVE_SSE42)
    options.instruction_set = SSE42;
#else
    options.instruction_set = FALLBACK;
#endif

    /* Options specific to mocks */
    /* Options for DDrppi_mocks (FAST_DIVIDE is also applicable for both DDsmu, and DDsmu_mocks) */
#if defined(FAST_DIVIDE)
#if FAST_DIVIDE > MAX_FAST_DIVIDE_NR_STEPS
    options.fast_divide_and_NR_steps = MAX_FAST_DIVIDE_NR_STEPS;
#else
    options.fast_divide_and_NR_steps = FAST_DIVIDE;
#endif
#endif

    options.link_in_ra=1;
    options.link_in_dec=1;

    //Introduced in Corrfunc v2.3
    /* optimizations based on min. separation between cell-pairs. Enabled by default */
    options.enable_min_sep_opt=1;
    
    options.fast_acos=1;

    /* For the thread timings */
    options.totncells_timings = 0;
    /* If the API level timers are requested, then
       this pointer will have to be allocated */
    options.cell_timings = NULL;

    /*Setup the binning options */
    reset_max_cells(&options);
    reset_bin_refine_factors(&options);
    return options;
}

void free_cell_timings(config_options *options)
{
    if(options->totncells_timings > 0 && options->cell_timings != NULL) {
        free(options->cell_timings);
    }
    options->totncells_timings = 0;

    return;
}

void allocate_cell_timer(config_options *options, const int64_t num_cell_pairs)
{
    if(options->totncells_timings >= num_cell_pairs) return;

    free_cell_timings(options);
    options->cell_timings = (api_cell_timings*) calloc(num_cell_pairs, sizeof(*(options->cell_timings)));
    if(options->cell_timings == NULL) {
        fprintf(stderr,"Warning: In %s> Could not allocate memory to store the API timings per cell. \n",
                __FUNCTION__);
    } else {
        options->totncells_timings = num_cell_pairs;
    }

    return;
}

void assign_cell_timer(api_cell_timings *cell_timings, const int64_t num_cell_pairs, config_options *options)
{
    /* Does the existing thread timings pointer have enough memory allocated ?*/
    allocate_cell_timer(options, num_cell_pairs);

    /* This looks like a repeated "if" condition but it is not. Covers the case for the calloc failure above */
    if(options->totncells_timings >= num_cell_pairs) {
        memmove(options->cell_timings, cell_timings, sizeof(api_cell_timings) * num_cell_pairs);
    }
}

void set_bin_refine_scheme(config_options *options, const int8_t flag) {
    // Set the top (nbits-4) to whatever already exists in binning_flag
    // and then set the bottom 4 bits to BIN_DFL
    options->binning_flags = (options->binning_flags & ~BINNING_REF_MASK) | (flag & BINNING_REF_MASK);
}

void reset_bin_refine_scheme(config_options *options) {
    set_bin_refine_scheme(options, BINNING_DFL);
}

int8_t get_bin_refine_scheme(config_options *options) {
    // Return the last 4 bits as 8 bits int
    return (int8_t)(options->binning_flags & BINNING_REF_MASK);
}

void set_bin_refine_factors(config_options *options, const int bin_refine_factors[3]) {
    for(int i = 0; i < 3; i++) {
        int8_t bin_refine = bin_refine_factors[i];
        if(bin_refine_factors[i] > INT8_MAX) {
            fprintf(stderr,"Warning: bin refine factor[%d] can be at most %d. Found %d instead\n", i,
                    INT8_MAX, bin_refine_factors[i]);
            bin_refine = 1;
        }
        options->bin_refine_factors[i] = bin_refine;
    }
    /*
      Note, programmatically setting the refine factors resets the binning flag to "BINNING_DFL"
      BINNING_CUST is only set via function parameters, or explicitly 
    */
    reset_bin_refine_scheme(options);
}

void set_custom_bin_refine_factors(config_options *options, const int bin_refine_factors[3]) {
    set_bin_refine_factors(options, bin_refine_factors);
    set_bin_refine_scheme(options, BINNING_CUST);
}

void reset_bin_refine_factors(config_options *options)
{
    /* refine factors of 2,2,1 in the xyz dims
       seems to produce the fastest code */
    options->bin_refine_factors[0] = 2;
    options->bin_refine_factors[1] = 2;
    options->bin_refine_factors[2] = 1;
    reset_bin_refine_scheme(options);
}


void set_max_cells(config_options *options, const int max)
{
    if(max <=0) {
        fprintf(stderr,"Warning: Max. cells per dimension was requested to be set to "
                "a negative number = %d...returning\n", max);
        return;
    }

    if(max > INT16_MAX) {
        fprintf(stderr,"Warning: Max cells per dimension is a 2-byte integer and can not "
                "hold supplied value of %d. Max. allowed value for max_cells_per_dim is %d\n",
                max, INT16_MAX);
    }

    options->max_cells_per_dim = max;
}

void reset_max_cells(config_options *options)
{
    options->max_cells_per_dim = NLATMAX;
}

#include <stdio.h>

#include "defs.h"
#include "weights.h"

int get_config_options(config_options *options, const char *weight_method){
    memset(options, 0, sizeof(*options));

    if(get_weight_method_by_name(weight_method, &options->weight_method) != EXIT_SUCCESS){
        sprintf(ERRMSG,"Error: Unknown weight method `%s'\n", weight_method);
        return EXIT_FAILURE;
    }

    // A value of <= 0 is non-periodic, > 0 is periodic.
    options->boxsize_x = 0.;
    options->boxsize_y = 0.;
    options->boxsize_z = 0.;

    options->verbose = 1;
    options->need_avg_sep = 1;

    options->instruction_set = FASTEST;

    /* Options specific to mocks */
    /* Options for DDrppi_mocks (FAST_DIVIDE is also applicable for both DDsmu, and DDsmu_mocks) */
#if defined(FAST_DIVIDE)
#if FAST_DIVIDE > MAX_FAST_DIVIDE_NR_STEPS
    options->fast_divide_and_NR_steps = MAX_FAST_DIVIDE_NR_STEPS;
#else
    options->fast_divide_and_NR_steps = FAST_DIVIDE;
#endif
#endif

    options->link_in_ra=1;
    options->link_in_dec=1;
    
    options->fast_acos=1;

    /* For the thread timings */
    options->totncells_timings = 0;
    /* If the API level timers are requested, then
       this pointer will have to be allocated */
    options->cell_timings = NULL;

    /*Setup the gridding options */
    options->max_cells_per_dim = NCELLMAX;
    reset_grid_refine_factors(options);
    return EXIT_SUCCESS;
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

void set_grid_refine_scheme(config_options *options, const int8_t flag) {
    // Set the top (nbits-4) to whatever already exists in gridding_flag
    // and then set the bottom 4 bits to BIN_DFL
    options->gridding_flags = (options->gridding_flags & ~GRIDDING_REF_MASK) | (flag & GRIDDING_REF_MASK);
}

void reset_grid_refine_scheme(config_options *options) {
    set_grid_refine_scheme(options, GRIDDING_DFL);
}

int8_t get_grid_refine_scheme(const config_options *options) {
    // Return the last 4 bits as 8 bits int
    return (int8_t)(options->gridding_flags & GRIDDING_REF_MASK);
}

void set_grid_refine_factors(config_options *options, const int grid_refine_factors[3]) {
    for(int i = 0; i < 3; i++) {
        int8_t grid_refine = grid_refine_factors[i];
        if(grid_refine_factors[i] > INT8_MAX) {
            fprintf(stderr,"Warning: grid refine factor[%d] can be at most %d. Found %d instead\n", i,
                    INT8_MAX, grid_refine_factors[i]);
            grid_refine = 1;
        }
        options->grid_refine_factors[i] = grid_refine;
    }
    /*
      Note, programmatically setting the refine factors resets the gridding flag to "GRIDDING_DFL"
      GRIDDING_CUST is only set via function parameters, or explicitly 
    */
    reset_grid_refine_scheme(options);
}

void set_custom_grid_refine_factors(config_options *options, const int grid_refine_factors[3]) {
    set_grid_refine_factors(options, grid_refine_factors);
    set_grid_refine_scheme(options, GRIDDING_CUST);
}

void reset_grid_refine_factors(config_options *options)
{
    /* refine factors of 2,2,1 in the xyz dims
       seems to produce the fastest code */
    options->grid_refine_factors[0] = 2;
    options->grid_refine_factors[1] = 2;
    options->grid_refine_factors[2] = 1;
    reset_grid_refine_scheme(options);
}

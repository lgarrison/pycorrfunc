#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

#include <Python.h>

#include "cpp_utils.h"

#include "countpairs.h"
#include "kernelfuncs.h"

#include "function_precision.h"
#include "utils.h"

#include "weights.h"

#include "defs.h"
#include "utils.h"
#include "progressbar.h"

#include "gridlink.h"
#include "gridlink_utils.h"

#include "simdconfig.h"

#ifdef _OPENMP
#include <omp.h>
#else
int omp_get_thread_num(void) { return 0; }
#endif

char ERRMSG[1024];


kernel_func_ptr countpairs_driver(const config_options *options) {
    // If options.isa == FASTEST, use the fastest where both:
    //  1) the HAVE_<ISA> macro is defined, and 
    //  2) the <isa>_available() function returns true.
    // If options.isa != FASTEST, use the specified ISA and raise an error if it is not available, without falling back.

    int err_if_not_avail = options->instruction_set != FASTEST;

    kernel_func_ptr function = NULL;
    switch(options->instruction_set) {
        case FASTEST:
            // fallthrough
        case AVX512:
            #ifdef HAVE_AVX512
            if (avx512_available()) {
                function = countpairs_avx512;
                if(options->verbose) fprintf(stderr, "Using AVX512 kernel\n");
                break;
            }
            #endif
            if (err_if_not_avail) {
                sprintf(ERRMSG, "AVX512 not available\n");
                return NULL;
            }
            // fallthrough
        case AVX:
            #ifdef HAVE_AVX
            if (avx_available()) {
                function = countpairs_avx;
                if (options->verbose) fprintf(stderr, "Using AVX kernel\n");
                break;
            }
            #endif
            if (err_if_not_avail) {
                sprintf(ERRMSG, "AVX not available\n");
                return NULL;
            }
            // fallthrough
        case SSE42:
            #ifdef HAVE_SSE42
            if (sse_available()) {
                function = countpairs_sse;
                if (options->verbose) fprintf(stderr, "Using SSE42 kernel\n");
                break;
            }
            #endif
            if (err_if_not_avail) {
                sprintf(ERRMSG, "SSE42 not available\n");
                return NULL;
            }
            // fallthrough
        case FALLBACK:
            function = countpairs_fallback;
            if (options->verbose) fprintf(stderr, "Using fallback kernel\n");
            break;
        default:
            sprintf(ERRMSG, "Unknown ISA\n");
            return NULL;
    }

    return function;
}

int countpairs(const int64_t ND1, DOUBLE *X1, DOUBLE *Y1, DOUBLE *Z1, DOUBLE *W1,
               const int64_t ND2, DOUBLE *X2, DOUBLE *Y2, DOUBLE *Z2, DOUBLE *W2,
               const int64_t N_bin_edges, const DOUBLE *bin_edges,
               config_options *options,
               uint64_t *npairs,
               DoubleAccum *ravg,
               DoubleAccum *wavg){

    const int need_wavg = options->weight_method != NONE;
    const int sort_on_z = 1;
    const int enable_min_sep_opt = 1;

    /* runtime dispatch - get the function pointer */
    kernel_func_ptr countpairs_function = countpairs_driver(options);
    
    if(countpairs_function == NULL) {
        return EXIT_FAILURE;
    }

    // Setup the bin edges
    const DOUBLE rmax = bin_edges[N_bin_edges - 1];

    DOUBLE bin_edges_sqr[N_bin_edges];
    for(int i=0; i < N_bin_edges;i++) {
        bin_edges_sqr[i] = bin_edges[i]*bin_edges[i];
    }

    // determine the periodicity request
    const int periodic_x = options->boxsize_x > 0;
    const int periodic_y = options->boxsize_y > 0;
    const int periodic_z = options->boxsize_z > 0;

    const DOUBLE xwrap = periodic_x ? options->boxsize_x : 0.;
    const DOUBLE ywrap = periodic_y ? options->boxsize_y : 0.;
    const DOUBLE zwrap = periodic_z ? options->boxsize_z : 0.;
  
    // Determine the spatial particle extents
    DOUBLE xmin, xmax, ymin, ymax, zmin, zmax;
    get_max_min(ND1, X1, Y1, Z1, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);

    if(options->autocorr==0) {
        if(options->verbose) {
            fprintf(stderr,"ND1 = %12"PRId64" [xmin,ymin,zmin] = [%lf,%lf,%lf], [xmax,ymax,zmax] = [%lf,%lf,%lf]\n",ND1,xmin,ymin,zmin,xmax,ymax,zmax);
        }

        get_max_min(ND2, X2, Y2, Z2, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        if(options->verbose) {
            fprintf(stderr,"ND2 = %12"PRId64" [xmin,ymin,zmin] = [%lf,%lf,%lf], [xmax,ymax,zmax] = [%lf,%lf,%lf]\n",ND2,xmin,ymin,zmin,xmax,ymax,zmax);
        }
    }
    
    if(options->verbose) {
        if(periodic_x) {
            fprintf(stderr,"Running with points in [xmin,xmax] = %lf,%lf with periodic wrapping = %lf\n",xmin,xmax,xwrap);
        } else {
            fprintf(stderr,"Running with points in [xmin,xmax] = %lf,%lf (non-periodic)\n",xmin,xmax);
        }
        if(periodic_y) {
            fprintf(stderr,"Running with points in [ymin,ymax] = %lf,%lf with periodic wrapping = %lf\n",ymin,ymax,ywrap);
        } else {
            fprintf(stderr,"Running with points in [ymin,ymax] = %lf,%lf (non-periodic)\n",ymin,ymax);
        }
        if(periodic_z) {
            fprintf(stderr,"Running with points in [zmin,zmax] = %lf,%lf with periodic wrapping = %lf\n",zmin,zmax,zwrap);
        } else {
            fprintf(stderr,"Running with points in [zmin,zmax] = %lf,%lf (non-periodic)\n",zmin,zmax);
        }
    }
    
    // Set the grid refine factors
    if(get_grid_refine_scheme(options) == GRIDDING_DFL) {
        // default is 2,2,1
        // FUTURE probably should not use rmax heuristic and stick to np
        if(rmax < 0.05*xwrap) {
            options->grid_refine_factors[0] = 1;
        }
        if(rmax < 0.05*ywrap) {
            options->grid_refine_factors[1] = 1;
        }
        if(rmax < 0.05*zwrap) {
            options->grid_refine_factors[2] = 1;
        }

        int nmesh_x, nmesh_y, nmesh_z;
        DOUBLE gridsize[3];
        int status = get_gridsize(&gridsize[0], &nmesh_x, xmax-xmin, xwrap, rmax, options->grid_refine_factors[0], options->max_cells_per_dim);
        status += get_gridsize(&gridsize[1], &nmesh_y, ymax-ymin, ywrap, rmax, options->grid_refine_factors[1], options->max_cells_per_dim);
        status += get_gridsize(&gridsize[2], &nmesh_z, zmax-zmin, zwrap, rmax, options->grid_refine_factors[2], options->max_cells_per_dim);

        if(status != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }
        
        int max_nmesh = MAX(nmesh_x, MAX(nmesh_y, nmesh_z));
        double avg_np = ((double)ND1)/(nmesh_x*nmesh_y*nmesh_z);
        
        // Use the old heuristic of boosting the first two grid refs if the average number of particles per cell is too high
        // FUTURE lots of ways to improve this!
        if(avg_np >= 256 || max_nmesh <= 10) {
            for(int i=0;i<2;i++) {
                if(gridsize[i] <= 0.) continue;
                options->grid_refine_factors[i]++;
            }
        }
    }

    // Divide the particles into cells
    cellarray *lattice1 = gridlink(
        ND1, X1, Y1, Z1, W1,
        xmin, xmax, ymin, ymax, zmin, zmax,
        rmax, rmax, rmax,
        xwrap, ywrap, zwrap,
        options->grid_refine_factors[0],
        options->grid_refine_factors[1],
        options->grid_refine_factors[2],
        sort_on_z,
        options
    );

    if(lattice1 == NULL) {
        return EXIT_FAILURE;
    }

    cellarray *lattice2 = NULL;
    if(options->autocorr==0) {
        lattice2 = gridlink(ND2, X2, Y2, Z2, W2,
                                   xmin, xmax, ymin, ymax, zmin, zmax,
                                   rmax, rmax, rmax,
                                   xwrap, ywrap, zwrap,
                                   options->grid_refine_factors[0], options->grid_refine_factors[1], options->grid_refine_factors[2],
                                   sort_on_z,
                                   options);
        if(lattice2 == NULL) {
            free_cellarray(&lattice1);
            free_cellarray(&lattice2);
            return EXIT_FAILURE;
        }
        if( ! (lattice1->nmesh_x == lattice2->nmesh_x && lattice1->nmesh_y == lattice2->nmesh_y && lattice1->nmesh_z == lattice2->nmesh_z) ) {
            sprintf(ERRMSG,"Error: The two sets of 3-D lattices do not have identical bins. First has dims (%d, %d, %d) while second has (%d, %d, %d)\n",
                    lattice1->nmesh_x, lattice1->nmesh_y, lattice1->nmesh_z, lattice2->nmesh_x, lattice2->nmesh_y, lattice2->nmesh_z);
            free_cellarray(&lattice1);
            free_cellarray(&lattice2);
            return EXIT_FAILURE;
        }
    } else {
        lattice2 = lattice1;
    }

    // Generate the cell pairs
    int64_t num_cell_pairs = 0;
    struct cell_pair *all_cell_pairs = generate_cell_pairs(
        &num_cell_pairs,
        lattice1, lattice2,
        options->grid_refine_factors[0],
        options->grid_refine_factors[1],
        options->grid_refine_factors[2],
        xwrap, ywrap, zwrap,
        rmax, -1.0, -1.0, /*max_3D_sep, max_2D_sep, max_1D_sep*/
        enable_min_sep_opt,
        options->autocorr,
        periodic_x, periodic_y, periodic_z
    );
    if(all_cell_pairs == NULL) {
        free_cellarray(&lattice1);
        if(options->autocorr == 0) {
            free_cellarray(&lattice2);
        }
        return EXIT_FAILURE;
    }

    // Initialize the pair counters
    for (int i = 0; i < N_bin_edges - 1; i++) {
        npairs[i] = 0;
        if (options->need_avg_sep) {
            ravg[i] = 0.0; 
        }
        if (need_wavg) {
            wavg[i] = 0.0;
        }
    }

    // Initialize the progress bar
    int64_t numdone=0;
    if(options->verbose) {
        init_my_progressbar(num_cell_pairs);
    }

    uint64_t *local_npairs[options->numthreads];
    DoubleAccum *local_ravg[options->numthreads];
    DoubleAccum *local_wavg[options->numthreads];

#ifdef _OPENMP
    #pragma omp parallel num_threads(options->numthreads)
    {
        const int tid = omp_get_thread_num();
        local_npairs[tid] = (uint64_t *) my_malloc(sizeof(uint64_t), N_bin_edges - 1);
        memset(local_npairs[tid], 0, sizeof(uint64_t) * (N_bin_edges - 1));

        local_rpavg[tid] = options->need_avg_sep ? 
            (DOUBLE *) my_malloc(sizeof(DOUBLE), N_bin_edges - 1) :
            NULL;
        if (local_rpavg[tid] != NULL) {
            memset(local_rpavg[tid], 0, sizeof(DOUBLE) * (N_bin_edges - 1));
        }

        local_wavg[tid] = need_wavg ? 
            (DOUBLE *) my_malloc(sizeof(DOUBLE), N_bin_edges - 1) :
            NULL;
        if (local_wavg[tid] != NULL) {
            memset(local_wavg[tid], 0, sizeof(DOUBLE) * (N_bin_edges - 1));
        }
    }
#else
    int tid = 0;
    local_npairs[tid] = npairs;
    local_rpavg[tid] = options->need_avg_sep ? rpavg : NULL;
    local_wavg[tid] = need_wavg ? wavg : NULL;
#endif

    // For Ctrl-C handling
    int interrupted = 0;

#ifdef _OPENMP
    #pragma omp parallel for \
        schedule(dynamic) \
        shared(numdone, interrupted) \
        num_threads(options->numthreads)
#endif
    for(int64_t icellpair=0;icellpair<num_cell_pairs;icellpair++) {
        const int tid = omp_get_thread_num();

        if(interrupted) continue;
        if(options->verbose) {
            if (tid == 0){
                my_progressbar(numdone);
            }
#ifdef _OPENMP
            #pragma omp atomic
#endif
            numdone++;
        }

        if(tid == 0){
            if(PyErr_CheckSignals()){
                interrupted = 1;
                continue;
            }
        }

        struct cell_pair *this_cell_pair = &all_cell_pairs[icellpair];

        uint64_t *this_npairs = local_npairs[tid];
        DoubleAccum *this_ravg = local_ravg[tid];
        DoubleAccum *this_wavg = local_wavg[tid];

        const int64_t icell = this_cell_pair->cellindex1;
        const int64_t icell2 = this_cell_pair->cellindex2;

        int64_t first_N = lattice1->offsets[icell+1] - lattice1->offsets[icell];
        int64_t second_N = lattice2->offsets[icell2+1] - lattice2->offsets[icell2];

        int64_t i = lattice1->offsets[icell];
        int64_t j = lattice2->offsets[icell2];

        countpairs_function(
            this_npairs, this_ravg, this_wavg,
            first_N, lattice1->X + i, lattice1->Y + i, lattice1->Z + i, lattice1->W + i,
            second_N, lattice2->X + j, lattice2->Y + j, lattice2->Z + j, lattice2->W + j,
            this_cell_pair->same_cell,
            N_bin_edges, bin_edges_sqr,
            this_cell_pair->xwrap, this_cell_pair->ywrap, this_cell_pair->zwrap,
            this_cell_pair->min_dx, this_cell_pair->min_dy, this_cell_pair->min_dz,
            this_cell_pair->closest_x1, this_cell_pair->closest_y1, this_cell_pair->closest_z1,
            options->weight_method
        );
    }  // num_cell_pairs loop

    free(all_cell_pairs);
    free_cellarray(&lattice1);
    if(options->autocorr == 0) {
        free_cellarray(&lattice2);
    }

    if(interrupted){
        raise_python_exception();
        return EXIT_FAILURE;  // never reached
    }

#ifdef _OPENMP
    // Only reduce for OpenMP. Without it, the local_npairs are the same as npairs.

    // We can *almost* use OpenMP's reduction clause instead of doing the following, but
    // (1) there's not an elegant way to toggle the ravg and wavg reductions on and off;
    // (2) there's no way to parallelize the reductions (only relevant for big N_bin_edges);
    // (3) reductions live on the stack, and I'm not sure we want to rely on that for large arrays.

    #pragma omp parallel num_threads(options->numthreads)
    {
        #pragma omp for schedule(static,CACHELINE/sizeof(int64_t))
        for(int j=0;j<N_bin_edges - 1;j++) {
            for(int i=0;i<options->numthreads;i++) {
                npairs[j] += local_npairs[i][j];
                if(options->need_avg_sep) {
                    rpavg[j] += local_rpavg[i][j];
                }
                if(need_wavg) {
                    wavg[j] += local_wavg[i][j];
                }
            }
        }

        const int tid = omp_get_thread_num();
        free(local_npairs[tid]);
        if(options->need_avg_sep) {
            free(local_rpavg[tid]);
        }
        if(need_wavg) {
            free(local_wavg[tid]);
        }
    }
#endif

    if(options->verbose) {
        finish_myprogressbar();
    }

    //The code does not double count for autocorrelations
    //which means the npairs and ravg values need to be doubled;
    if(options->autocorr == 1) {
        for(int i=0;i<N_bin_edges - 1;i++) {
            npairs[i] *= 2;
            if(options->need_avg_sep) {
                ravg[i] *= (DOUBLE) 2.0;
            }
            if(need_wavg) {
                wavg[i] *= (DOUBLE) 2.0;
            }
        }

        /* Is the min. requested separation 0.0 ?*/
        /* The comparison is '<=' rather than '==' only to silence
           the compiler  */
        if(bin_edges[0] <= 0.0) {
            /* Then, add all the self-pairs. This ensures that
               a cross-correlation with two identical datasets
               produces the same result as the auto-correlation  */
            npairs[0] += ND1;

            // Increasing npairs affects ravg and wavg.
            // We don't need to add anything to ravg; all the self-pairs have 0 separation!
            // The self-pairs have non-zero weight, though.  So, fix that here.
            if(need_wavg){
                // Keep in mind this is an autocorrelation (i.e. only one particle set to consider)
                weight_func_t weight_func = get_weight_func_by_method(options->weight_method);
                for(int64_t j = 0; j < ND1; j++){
                    wavg[0] += weight_func(0., 0., 0., W1[j], W1[j]);
                }
            }
        }
    }

    for(int i=0;i<N_bin_edges - 1;i++) {
        if(npairs[i] > 0) {
            if(options->need_avg_sep) {
                ravg[i] /= (DoubleAccum) npairs[i];
            }
            if(need_wavg) {
                wavg[i] /= (DoubleAccum) npairs[i];
            }
        }
    }    

    reset_grid_refine_factors(options);
    return EXIT_SUCCESS;
}

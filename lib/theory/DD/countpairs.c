#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

#include "countpairs.h" //function proto-type
#include "kernelfuncs.h"

#include "function_precision.h"
#include "utils.h"

#include "weight_functions.h"

#include "defs.h"
#include "utils.h" //all of the utilities
#include "progressbar.h" //for the progressbar

#include "gridlink_impl.h"//function proto-type for gridlink
#include "gridlink_utils.h" //for associated helper routines

#if defined(_OPENMP)
#include <omp.h>
#endif

int interrupt_status=EXIT_SUCCESS;

void interrupt_handler_countpairs(int signo)
{
    fprintf(stderr,"Received signal = `%s' (signo = %d). Aborting \n",strsignal(signo), signo);
    interrupt_status = EXIT_FAILURE;
}

countpairs_func_ptr countpairs_driver(const config_options *options) {
    // If options.isa == FASTEST, use the fastest that both has the HAVE_<ISA> macro defined
    // and has the <isa>_available() function return true.
    // If options.isa != FASTEST, use the specified ISA and raise an error if it is not available, without falling back.

    int err_if_not_avail = options->instruction_set != FASTEST;

    countpairs_func_ptr function = NULL;

//     // Check for AVX512F support
//    // #ifdef HAVE_AVX512F
//   //  if (function == NULL) {
//    //     function = countpairs_avx512_intrinsics;
//   //  }
//   //  #endif

    switch(options->instruction_set) {
        case FASTEST:  // fallthrough
        case AVX512F:
            #ifdef HAVE_AVX512F
            if (avx512_available()) {
                function = countpairs_avx512_intrinsics;
                if(options->verbose) fprintf(stderr, "Using AVX512F kernel\n");
                break;
            } else if (err_if_not_avail) {
                fprintf(stderr, "AVX512F not available\n");
                return NULL;
            }
            #endif
            // fallthrough
        case AVX:
            #ifdef HAVE_AVX
            if (avx_available()) {
                function = countpairs_avx_intrinsics;
                if (options->verbose) fprintf(stderr, "Using AVX kernel\n");
                break;
            } else if (err_if_not_avail) {
                fprintf(stderr, "AVX not available\n");
                return NULL;
            }
            #endif
            // fallthrough
        case SSE42:
            #ifdef HAVE_SSE42
            if (sse_available()) {
                function = countpairs_sse_intrinsics;
                if (options->verbose) fprintf(stderr, "Using SSE42 kernel\n");
                break;
            } else if (err_if_not_avail) {
                fprintf(stderr, "SSE42 not available\n");
                return NULL;
            }
            #endif
            // fallthrough
        case FALLBACK:
            function = countpairs_fallback;
            if (options->verbose) fprintf(stderr, "Using fallback kernel\n");
            break;
        default:
            fprintf(stderr, "Unknown ISA\n");
            return NULL;
    }

    return function;
}

int countpairs(const int64_t ND1, DOUBLE *X1, DOUBLE *Y1, DOUBLE *Z1,
               const int64_t ND2, DOUBLE *X2, DOUBLE *Y2, DOUBLE *Z2,
               const int64_t N_bin_edges, const DOUBLE *bin_edges,
               config_options *options,
               uint64_t *npairs,
               DOUBLE *rpavg,
               DOUBLE *weighted_pairs)
{
    int need_weighted_pairs = options->weight_method != NONE;

    struct timeval t0;
    if(options->c_api_timer) {
        gettimeofday(&t0, NULL);
    }

    if(options->max_cells_per_dim == 0) {
        fprintf(stderr,"Warning: Max. cells per dimension is set to 0 - resetting to `NLATMAX' = %d\n", NLATMAX);
        options->max_cells_per_dim = NLATMAX;
    }
    
    for(int i=0;i<3;i++) {
        if(options->bin_refine_factors[i] < 1) {
            fprintf(stderr,"Warning: bin refine factor along axis = %d *must* be >=1. Instead found bin refine factor =%d\n",
                    i, options->bin_refine_factors[i]);
            reset_bin_refine_factors(options);
            break;/* all factors have been reset -> no point continuing with the loop */
        }
    }


    options->sort_on_z = 1;
    /* setup interrupt handler -> mostly useful during the python execution.
       Let's Ctrl-C abort the extension  */
    SETUP_INTERRUPT_HANDLERS(interrupt_handler_countpairs);

    /***********************
     *initializing the bins
     ************************/
    // DOUBLE rpmin = bin_edges[0];
    double rpmax = bin_edges[N_bin_edges - 1];
  
    //Find the min/max of the data
    DOUBLE xmin, xmax, ymin, ymax, zmin, zmax;
    xmin = ymin = zmin = MAX_POSITIVE_FLOAT;
    xmax = ymax = zmax = -MAX_POSITIVE_FLOAT;
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

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

    if(options->periodic && options->boxsize == BOXSIZE_NOTGIVEN){
        fprintf(stderr, "boxsize = %g must be specified with periodic wrap. Please specify a non-zero boxsize, or zero to detect the particle extent, or -1 to make a dimension non-periodic.\n",
            options->boxsize);
        return EXIT_FAILURE;
    }
    
    const double boxsize_y = options->boxsize_y == BOXSIZE_NOTGIVEN ? options->boxsize : options->boxsize_y;
    const double boxsize_z = options->boxsize_z == BOXSIZE_NOTGIVEN ? options->boxsize : options->boxsize_z;

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

    const int periodic_x = options->periodic && options->boxsize_x >= 0;
    const int periodic_y = options->periodic && boxsize_y >= 0;
    const int periodic_z = options->periodic && boxsize_z >= 0;

    // If periodic (L!=-1), use given boxsize (L>0) or auto-detect (L==0). If not, set wrap value to 0.
    const DOUBLE xwrap = periodic_x ? (options->boxsize_x > 0 ? options->boxsize_x : (xmax-xmin)) : 0.;
    const DOUBLE ywrap = periodic_y ? (boxsize_y > 0 ? boxsize_y : (ymax-ymin)) : 0.;
    const DOUBLE zwrap = periodic_z ? (boxsize_z > 0 ? boxsize_z : (zmax-zmin)) : 0.;
    const DOUBLE pimax = (DOUBLE) rpmax;
    
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
    
    if(get_bin_refine_scheme(options) == BINNING_DFL) {
        if(rpmax < 0.05*xwrap) {
            options->bin_refine_factors[0] = 1;
        }
        if(rpmax < 0.05*ywrap) {
            options->bin_refine_factors[1] = 1;
        }
        if(pimax < 0.05*zwrap) { //pimax := rpmax. Here to prevent copy-pasting bugs
            options->bin_refine_factors[2] = 1;
        }
    }

    /*---Create 3-D lattice--------------------------------------*/
    int nmesh_x=0,nmesh_y=0,nmesh_z=0;
    cellarray *lattice1 = gridlink(ND1, X1, Y1, Z1, &(options->weights0),
                                                 xmin, xmax, ymin, ymax, zmin, zmax,
                                                 rpmax, rpmax, rpmax,
                                                 xwrap, ywrap, zwrap,
                                                 options->bin_refine_factors[0],
                                                 options->bin_refine_factors[1],
                                                 options->bin_refine_factors[2],
                                                 &nmesh_x, &nmesh_y, &nmesh_z, options);
    if(lattice1 == NULL) {
        return EXIT_FAILURE;
    }

    /* If there too few cells (BOOST_CELL_THRESH is ~10), and the number of cells can be increased, then boost bin refine factor by ~1*/
    const double avg_np = ((double)ND1)/(nmesh_x*nmesh_y*nmesh_z);
    const int max_nmesh = fmax(nmesh_x, fmax(nmesh_y, nmesh_z));
    if((max_nmesh <= BOOST_CELL_THRESH || avg_np >= BOOST_NUMPART_THRESH)
       && max_nmesh < options->max_cells_per_dim) {
        if(options->verbose) {
            fprintf(stderr,"%s> gridlink seems inefficient. nmesh = (%d, %d, %d); avg_np = %.3g. ", __FUNCTION__, nmesh_x, nmesh_y, nmesh_z, avg_np);
        }
        if(get_bin_refine_scheme(options) == BINNING_DFL) {
            if(options->verbose) {
                fprintf(stderr,"Boosting bin refine factor - should lead to better performance\n");
                fprintf(stderr,"xmin = %lf xmax=%lf rpmax = %lf\n", xmin, xmax, rpmax);
            }
            free_cellarray(lattice1, nmesh_x * (int64_t) nmesh_y * nmesh_z);
            // Only boost the first two dimensions.  Prevents excessive refinement.
            for(int i=0;i<2;i++) {
                options->bin_refine_factors[i] += BOOST_BIN_REF;
            }
            lattice1 = gridlink(ND1, X1, Y1, Z1, &(options->weights0),
                                       xmin, xmax, ymin, ymax, zmin, zmax,
                                       rpmax, rpmax, rpmax,
                                       xwrap, ywrap, zwrap,
                                       options->bin_refine_factors[0], options->bin_refine_factors[1], options->bin_refine_factors[2],
                                       &nmesh_x, &nmesh_y, &nmesh_z, options);
            if(lattice1 == NULL) {
                return EXIT_FAILURE;
            }
        } else {
            if(options->verbose) {
                fprintf(stderr,"Boosting bin refine factor could have helped. However, since custom bin refine factors "
                        "= (%d, %d, %d) are being used - continuing with inefficient mesh\n", options->bin_refine_factors[0],
                        options->bin_refine_factors[1], options->bin_refine_factors[2]);
            }
        }
    }

    cellarray *lattice2 = NULL;
    if(options->autocorr==0) {
        int ngrid2_x=0,ngrid2_y=0,ngrid2_z=0;
        lattice2 = gridlink(ND2, X2, Y2, Z2, &(options->weights1),
                                   xmin, xmax, ymin, ymax, zmin, zmax,
                                   rpmax, rpmax, rpmax,
                                   xwrap, ywrap, zwrap,
                                   options->bin_refine_factors[0], options->bin_refine_factors[1], options->bin_refine_factors[2],
                                   &ngrid2_x, &ngrid2_y, &ngrid2_z, options);
        if(lattice2 == NULL) {
            return EXIT_FAILURE;
        }
        if( ! (nmesh_x == ngrid2_x && nmesh_y == ngrid2_y && nmesh_z == ngrid2_z) ) {
            fprintf(stderr,"Error: The two sets of 3-D lattices do not have identical bins. First has dims (%d, %d, %d) while second has (%d, %d, %d)\n",
                    nmesh_x, nmesh_y, nmesh_z, ngrid2_x, ngrid2_y, ngrid2_z);
            return EXIT_FAILURE;
        }
    } else {
        lattice2 = lattice1;
    }
    const int64_t totncells = (int64_t) nmesh_x * (int64_t) nmesh_y * (int64_t) nmesh_z;

    int64_t num_cell_pairs = 0;
    struct cell_pair *all_cell_pairs = generate_cell_pairs(lattice1, lattice2, totncells,
                                                                         &num_cell_pairs,
                                                                         options->bin_refine_factors[0],
                                                                         options->bin_refine_factors[1],
                                                                         options->bin_refine_factors[2],
                                                                         nmesh_x, nmesh_y, nmesh_z,
                                                                         xwrap, ywrap, zwrap,
                                                                         rpmax, -1.0, -1.0, /*max_3D_sep, max_2D_sep, max_1D_sep*/
                                                                         options->enable_min_sep_opt,
                                                                         options->autocorr,
                                                                         periodic_x, periodic_y, periodic_z);
    if(all_cell_pairs == NULL) {
        free_cellarray(lattice1, totncells);
        if(options->autocorr == 0) {
            free_cellarray(lattice2, totncells);
        }
        free((void*)bin_edges);
        return EXIT_FAILURE;
    }

    /* runtime dispatch - get the function pointer */
    countpairs_func_ptr countpairs_function = countpairs_driver(options);
    
    if(countpairs_function == NULL) {
        free_cellarray(lattice1, totncells);
        if(options->autocorr == 0) {
            free_cellarray(lattice2, totncells);
        }
        free((void *)bin_edges);
        return EXIT_FAILURE;
    }


for (int i = 0; i < N_bin_edges - 1; i++) {
        npairs[i] = 0;
        rpavg[i] = 0.0;
        weighted_pairs[i] = 0.0;
    }

#if defined(_OPENMP)
// Use appropriate data structures for OpenMP
uint64_t **all_npairs = (uint64_t **)matrix_calloc(sizeof(uint64_t), options->numthreads, N_bin_edges);

DOUBLE **all_rpavg = NULL;
if (options->need_avg_sep) {
    all_rpavg = (DOUBLE **)matrix_calloc(sizeof(DOUBLE), options->numthreads, N_bin_edges);
}

DOUBLE **all_weighted_pairs = NULL;
if (need_weighted_pairs) {
    all_weighted_pairs = (DOUBLE **)matrix_calloc(sizeof(DOUBLE), options->numthreads, N_bin_edges);
}

if (all_npairs == NULL ||
    (options->need_avg_sep && all_rpavg == NULL) ||
    (need_weighted_pairs && all_weighted_pairs == NULL)) {
    free_cellarray(lattice1, totncells);
    if (options->autocorr == 0) {
        free_cellarray(lattice2, totncells);
    }
    matrix_free((void **)all_npairs, options->numthreads);
    if (options->need_avg_sep) {
        matrix_free((void **)all_rpavg, options->numthreads);
    }
    if (need_weighted_pairs) {
        matrix_free((void **)all_weighted_pairs, options->numthreads);
    }
    free((void *)bin_edges);
    return EXIT_FAILURE;
}
#else
for (int i = 0; i < N_bin_edges - 1; i++) {
    npairs[i] = 0;
    if (options->need_avg_sep) {
        rpavg[i] = 0.0; 
    }
    if (need_weighted_pairs) {
        weighted_pairs[i] = 0.0;
    }
}
#endif

    DOUBLE bin_edges_sqr[N_bin_edges];
    for(int i=0; i < N_bin_edges;i++) {
        bin_edges_sqr[i] = bin_edges[i]*bin_edges[i];
    }

    DOUBLE sqr_rpmax=bin_edges_sqr[N_bin_edges-1];
    DOUBLE sqr_rpmin=bin_edges_sqr[0];

    int abort_status = EXIT_SUCCESS;
    int interrupted=0;
    int64_t numdone=0;
    if(options->verbose) {
        init_my_progressbar(num_cell_pairs,&interrupted);
    }
    /*---Loop-over-Data1-particles--------------------*/
#if defined(_OPENMP)
#pragma omp parallel shared(numdone, abort_status, interrupt_status) num_threads(options->numthreads)
    {
        int tid = omp_get_thread_num();
        uint64_t npairs[N_bin_edges];
        DOUBLE rpavg[N_bin_edges]; //thread-level, stored on stack
        DOUBLE weighted_pairs[N_bin_edges];

        for(int i = 0; i<N_bin_edges - 1;i++) {
            npairs[i] = 0;
            if(options->need_avg_sep) {
                rpavg[i] = 0.0;
            }
            if(need_weighted_pairs) {
                weighted_pairs[i] = 0.0;
            }
        }


#pragma omp for  schedule(dynamic) nowait
#endif//openmp
        
        for(int64_t icellpair=0;icellpair<num_cell_pairs;icellpair++) {
            

#if defined(_OPENMP)
#pragma omp flush (abort_status, interrupt_status)
#endif
            if(abort_status == EXIT_SUCCESS && interrupt_status == EXIT_SUCCESS) {
                //omp cancel was introduced in omp 4.0 - so this is my way of checking if loop needs to be cancelled
                /* If the verbose option is not enabled, avoid outputting anything unnecessary*/
                if(options->verbose) {
#if defined(_OPENMP)
                    if (omp_get_thread_num() == 0)
#endif
                        my_progressbar(numdone,&interrupted);


#if defined(_OPENMP)
#pragma omp atomic
#endif
                    numdone++;
                }

                struct cell_pair *this_cell_pair = &all_cell_pairs[icellpair];
                DOUBLE *this_rpavg = options->need_avg_sep ? rpavg:NULL;
                DOUBLE *this_weighted_pairs = need_weighted_pairs ? weighted_pairs:NULL;

                const int64_t icell = this_cell_pair->cellindex1;
                const int64_t icell2 = this_cell_pair->cellindex2;
                const cellarray *first = &lattice1[icell];
                const cellarray *second = &lattice2[icell2];

                const int status = countpairs_function(first->nelements, first->x, first->y, first->z, &(first->weights),
                                                              second->nelements, second->x, second->y, second->z, &(second->weights),
                                                              this_cell_pair->same_cell,
                                                              sqr_rpmax, sqr_rpmin, N_bin_edges, bin_edges_sqr, pimax, //pimax is simply rpmax cast to DOUBLE
                                                              this_cell_pair->xwrap, this_cell_pair->ywrap, this_cell_pair->zwrap,
                                                              this_cell_pair->min_dx, this_cell_pair->min_dy, this_cell_pair->min_dz,
                                                              this_cell_pair->closest_x1, this_cell_pair->closest_y1, this_cell_pair->closest_z1,
                                                              this_rpavg, npairs,
                                                              this_weighted_pairs, options->weight_method);
                /* This actually causes a race condition under OpenMP - but mostly
                   I care that an error occurred - rather than the exact value of
                   the error status */
                abort_status |= status;

            }//abort-status
        }//icellpair loop over num_cell_pairs

#if defined(_OPENMP)
        for(int j=0;j<N_bin_edges - 1;j++) {
            all_npairs[tid][j] = npairs[j];
            if(options->need_avg_sep) {
                all_rpavg[tid][j] = rpavg[j];
            }
            if(need_weighted_pairs) {
                all_weighted_pairs[tid][j] = weighted_pairs[j];
            }
        }
    }//close the omp parallel region
#endif

    free(all_cell_pairs);

    if(options->copy_particles == 0) {
        int64_t *original_index = lattice1[0].original_index;
        int status = reorder_particles_back_into_original_order(ND1, original_index, X1, Y1, Z1, &(options->weights0));
        if(status != EXIT_SUCCESS) {
            return status;
        }
        if(options->autocorr == 0) {
            original_index = lattice2[0].original_index;
            status = reorder_particles_back_into_original_order(ND2, original_index, X2, Y2, Z2, &(options->weights1));
            if(status != EXIT_SUCCESS) {
                return status;
            }
        }
    }

    


    free_cellarray(lattice1, totncells);
    if(options->autocorr==0) {
        free_cellarray(lattice2, totncells);
    }
    if(abort_status != EXIT_SUCCESS || interrupt_status != EXIT_SUCCESS) {
        /* Cleanup memory here if aborting */
        free((void *)bin_edges);
#if defined(_OPENMP)
        matrix_free((void **) all_npairs, options->numthreads);
        if(options->need_avg_sep) {
            matrix_free((void **) all_rpavg, options->numthreads);
        }
        if(need_weighted_pairs) {
            matrix_free((void **) all_weighted_pairs, options->numthreads);
        }
#endif
        return EXIT_FAILURE;
    }

    if(options->verbose) {
        finish_myprogressbar(&interrupted);
    }

#if defined(_OPENMP)
    uint64_t npairs[N_bin_edges];
    DOUBLE rpavg[N_bin_edges];
    DOUBLE weighted_pairs[N_bin_edges];

    for(int i=0;i<N_bin_edges - 1;i++) {
        npairs[i] = 0;
        if(options->need_avg_sep) {
            rpavg[i] = 0.0;
        }
        if(need_weighted_pairs) {
            weighted_pairs[i] = 0.0;
        }
    }

    for(int i=0;i<options->numthreads;i++) {
        for(int j=0;j<N_bin_edges - 1;j++) {
            npairs[j] += all_npairs[i][j];
            if(options->need_avg_sep) {
                rpavg[j] += all_rpavg[i][j];
            }
            if(need_weighted_pairs) {
                weighted_pairs[j] += all_weighted_pairs[i][j];
            }
        }
    }
    matrix_free((void **) all_npairs, options->numthreads);
    if(options->need_avg_sep) {
        matrix_free((void **) all_rpavg, options->numthreads);
    }
    if(need_weighted_pairs) {
        matrix_free((void **) all_weighted_pairs, options->numthreads);
    }
#endif

    //The code does not double count for autocorrelations
    //which means the npairs and rpavg values need to be doubled;
    if(options->autocorr == 1) {
        const uint64_t int_fac = 2;
        const DOUBLE dbl_fac = (DOUBLE) 2.0;

        for(int i=0;i<N_bin_edges - 1;i++) {
            npairs[i] *= int_fac;
            if(options->need_avg_sep) {
                rpavg[i] *= dbl_fac;
            }
            if(need_weighted_pairs) {
                weighted_pairs[i] *= dbl_fac;
            }
        }

        /* Is the min. requested separation 0.0 ?*/
        /* The comparison is '<=' rather than '==' only to silence
           the compiler  */
        if(bin_edges[0] <= 0.0) {
            /* Then, add all the self-pairs. This ensures that
               a cross-correlation with two identical datasets
               produces the same result as the auto-correlation  */
            npairs[1] += ND1; //npairs[1] contains the first valid bin.

            // Increasing npairs affects rpavg and weighted_pairs.
            // We don't need to add anything to rpavg; all the self-pairs have 0 separation!
            // The self-pairs have non-zero weight, though.  So, fix that here.
            if(need_weighted_pairs){
                // Keep in mind this is an autocorrelation (i.e. only one particle set to consider)
                weight_func_t weight_func = get_weight_func_by_method(options->weight_method);
                pair_struct pair = {.num_weights = options->weights0.num_weights,
                                           .dx.d=0., .dy.d=0., .dz.d=0.,  // always 0 separation
                                           .parx.d=0., .pary.d=0., .parz.d=0.};
                for(int64_t j = 0; j < ND1; j++){
                    for(int w = 0; w < pair.num_weights; w++){
                        pair.weights0[w].d = ((DOUBLE *) options->weights0.weights[w])[j];
                        pair.weights1[w].d = ((DOUBLE *) options->weights0.weights[w])[j];
                    }
                    weighted_pairs[1] += weight_func(&pair);
                }
            }
        }
    }


    for(int i=0;i<N_bin_edges - 1;i++) {
        if(npairs[i] > 0) {
            if(options->need_avg_sep) {
                rpavg[i] /= (DOUBLE) npairs[i] ;
            }
            if(need_weighted_pairs) {
                weighted_pairs[i] /= (DOUBLE) npairs[i];
            }
        }
    }

    /* reset interrupt handlers to default */
    RESET_INTERRUPT_HANDLERS();
    reset_bin_refine_factors(options);

    if(options->c_api_timer) {
        struct timeval t1;
        gettimeofday(&t1, NULL);
        options->c_api_time = ADD_DIFF_TIME(t0, t1);
    }
    return EXIT_SUCCESS;

}

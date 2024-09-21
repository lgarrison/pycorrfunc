#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#include "defs.h"
#include "function_precision.h"
#include "utils.h"

#include "gridlink_utils.h"
#include "gridlink_impl.h"

#include "cpp_utils.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

#define CONVERT_3D_INDEX_TO_LINEAR(ix, iy, iz, nx, ny, nz) (ix*ny*nz + iy*nz + iz)


int gridlink(
    cellarray *lattice,
    const int64_t NPART,
    DOUBLE *X, DOUBLE *Y, DOUBLE *Z, DOUBLE *W,
    const DOUBLE xmin, const DOUBLE xmax,
    const DOUBLE ymin, const DOUBLE ymax,
    const DOUBLE zmin, const DOUBLE zmax,
    const DOUBLE max_x_size,
    const DOUBLE max_y_size,
    const DOUBLE max_z_size,
    const DOUBLE xwrap,
    const DOUBLE ywrap,
    const DOUBLE zwrap,
    const int xbin_refine_factor,
    const int ybin_refine_factor,
    const int zbin_refine_factor,
    const int sort_on_z,
    const config_options *options
    ){

    int nmesh_x=0,nmesh_y=0,nmesh_z=0;

    struct timeval t0;
    if(options->verbose) {
        gettimeofday(&t0,NULL);
    }

    DOUBLE xbinsize=ZERO, ybinsize=ZERO, zbinsize=ZERO;

    const int xstatus = get_binsize(xmax-xmin, xwrap, max_x_size, xbin_refine_factor, options->max_cells_per_dim, &xbinsize, &nmesh_x);
    const int ystatus = get_binsize(ymax-ymin, ywrap, max_y_size, ybin_refine_factor, options->max_cells_per_dim, &ybinsize, &nmesh_y);
    const int zstatus = get_binsize(zmax-zmin, zwrap, max_z_size, zbin_refine_factor, options->max_cells_per_dim, &zbinsize, &nmesh_z);
    if(xstatus != EXIT_SUCCESS || ystatus != EXIT_SUCCESS || zstatus != EXIT_SUCCESS) {
      fprintf(stderr,"Received xstatus = %d ystatus = %d zstatus = %d. Error\n", xstatus, ystatus, zstatus);
      return EXIT_FAILURE;
    }

    if(options->verbose) {
      fprintf(stderr,"In %s> Running with [nmesh_x, nmesh_y, nmesh_z]  = %d,%d,%d. ",__FUNCTION__,nmesh_x,nmesh_y,nmesh_z);
    }

    if(allocate_cellarray(lattice, NPART, nmesh_x, nmesh_y, nmesh_z, W != NULL) != EXIT_SUCCESS) {
        fprintf(stderr,"Error: Could not allocate memory for cellarray\n");
        return EXIT_FAILURE;
    }
    int64_t tot_ncells = lattice->tot_ncells;

    int64_t *all_cell_indices = (int64_t *) my_malloc(sizeof(*all_cell_indices), NPART);
    if(all_cell_indices == NULL) {
        free_cellarray(lattice);
        fprintf(stderr,"Error: In %s> Error allocating cell indicies\n", __FUNCTION__);
        return EXIT_FAILURE;
    }

    // "binsize > 0" guards against all particles falling in a plane (or worse),
    // in which case we set the inv to 0 to assign all particles to the first cell
    const DOUBLE xinv = xbinsize > 0 ? 1.0/xbinsize : 0.;
    const DOUBLE yinv = ybinsize > 0 ? 1.0/ybinsize : 0.;
    const DOUBLE zinv = zbinsize > 0 ? 1.0/zbinsize : 0.;

    // First pass over the particles: compute cell indices
    int64_t out_of_bounds = 0;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) reduction(+:out_of_bounds) num_threads(options->numthreads)
#endif
    for (int64_t i=0;i<NPART;i++)  {
        int ix=(int)((X[i]-xmin)*xinv) ;
        int iy=(int)((Y[i]-ymin)*yinv) ;
        int iz=(int)((Z[i]-zmin)*zinv) ;

        if (ix>nmesh_x-1)  ix--;    /* this shouldn't happen, but . . . */
        if (iy>nmesh_y-1)  iy--;
        if (iz>nmesh_z-1)  iz--;
    
        out_of_bounds += ix < 0 || ix >= nmesh_x || iy < 0 || iy >= nmesh_y || iz < 0 || iz >= nmesh_z;

        const int64_t icell = CONVERT_3D_INDEX_TO_LINEAR(ix, iy, iz, nmesh_x, nmesh_y, nmesh_z);
        all_cell_indices[i] = icell;
    }
    
    if(out_of_bounds != 0){
        fprintf(stderr,"Error: %"PRId64" particles are out of bounds. Check periodic wrapping?\n", out_of_bounds);
        free(all_cell_indices);
        free_cellarray(lattice);
        return EXIT_FAILURE;
    } 
    
    // Now determine the number of particles per cell so each cell knows its global starting index
    // This information is used for both the copy and in-place versions
    int nthreads = omp_get_max_threads();  // this is the upper bound, we will find the actual value (usually the same) below
    int64_t *cell_occupation[nthreads];

#if defined(_OPENMP)
    #pragma omp parallel num_threads(options->numthreads)
#endif
    {
        int tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();  // reduce to actual number of threads, will be the same in almost all cases
        cell_occupation[tid] = my_calloc(sizeof(*(cell_occupation[0])), tot_ncells);
        
        #if defined(_OPENMP)
        #pragma omp for schedule(static)
        #endif
        for(int64_t i = 0; i < NPART; i++){
            cell_occupation[tid][all_cell_indices[i]]++;
        }
    }
    
    int64_t maxcell = 0;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(options->numthreads) reduction(max:maxcell)
#endif
    for(int64_t c = 0; c < tot_ncells; c++){
        for(int t = 1; t < nthreads; t++){
            cell_occupation[0][c] += cell_occupation[t][c];
        }
        maxcell = MAX(maxcell, cell_occupation[0][c]);
    }

    // Fill cellstarts with the cumulative sum of the cell histogram
    parallel_cumsum(cell_occupation[0], tot_ncells, lattice->offsets);
    lattice->offsets[tot_ncells] = NPART;

#if defined(_OPENMP)
    #pragma omp parallel num_threads(options->numthreads)
#endif
    {
        int tid = omp_get_thread_num();
        free(cell_occupation[tid]);
    }

    int64_t *nwritten = my_malloc(sizeof(*nwritten), tot_ncells);
    if(nwritten == NULL) {
        free(all_cell_indices);
        free_cellarray(lattice);
        fprintf(stderr,"Error: Could not allocate memory for storing the number of particles per cell\n");
        return EXIT_FAILURE;
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(options->numthreads)
#endif
    for(int64_t c = 0; c < tot_ncells; c++){
        nwritten[c] = 0;
    }

    // Now we come to the final writes of the particles into their cells
#if defined(_OPENMP)
    #pragma omp parallel num_threads(options->numthreads)
#endif
    {
        int tid = omp_get_thread_num();
        int64_t cstart = tot_ncells*tid/nthreads;
        int64_t cend = tot_ncells*(tid+1)/nthreads;
        // Second loop over particles: having computed the starts of each cell, fill the cells.
        // Each thread is going to loop over all particles, but only process those in its cell domain.
        for (int64_t i=0;i<NPART;i++) {
            const int64_t icell = all_cell_indices[i];
            if(icell < cstart || icell >= cend){
                continue;
            }
            const int64_t j = lattice->offsets[icell] + nwritten[icell]++;
            lattice->X[j] = X[i];
            lattice->Y[j] = Y[i];
            lattice->Z[j] = Z[i];
            if(W != NULL) lattice->W[j] = W[i];

            // Store the particle bounds
            lattice->xbounds[0][icell] = MIN(lattice->xbounds[0][icell], X[i]);
            lattice->ybounds[0][icell] = MIN(lattice->ybounds[0][icell], Y[i]);
            lattice->zbounds[0][icell] = MIN(lattice->zbounds[0][icell], Z[i]);

            lattice->xbounds[1][icell] = MAX(lattice->xbounds[1][icell], X[i]);
            lattice->ybounds[1][icell] = MAX(lattice->ybounds[1][icell], Y[i]);
            lattice->zbounds[1][icell] = MAX(lattice->zbounds[1][icell], Z[i]);
        }
    }  // end parallel region

    // check that all particles have been written
    for(int64_t c = 0; c < tot_ncells; c++){
        if(nwritten[c] != lattice->offsets[c+1] - lattice->offsets[c]){
            fprintf(stderr,"Error: Cell %ld has %ld particles, but should have %ld particles\n", c, nwritten[c], lattice->offsets[c+1] - lattice->offsets[c]);
            free(all_cell_indices);
            free(nwritten);
            free_cellarray(lattice);
            return EXIT_FAILURE;
        }
    }
    
    free(all_cell_indices);  //Done with re-ordering the particles
    free(nwritten);

    /* Do we need to sort the particles in Z ? */
    if(0 && sort_on_z) {
        #ifdef _OPENMP
        #pragma omp parallel num_threads(options->numthreads)
        #endif
        {
            int64_t *iord = my_malloc(sizeof(*iord), maxcell);
            
            #ifdef _OPENMP
            #pragma omp for schedule(dynamic) 
            #endif
            for(int64_t icell=0;icell<tot_ncells;icell++) {
                DOUBLE *Zcell = lattice->Z + lattice->offsets[icell];
                int64_t ncell = lattice->offsets[icell+1] - lattice->offsets[icell];
                argsort(iord, Zcell, ncell);
                
                DOUBLE *Xcell = lattice->X + lattice->offsets[icell];
                DOUBLE *Ycell = lattice->Y + lattice->offsets[icell];
                DOUBLE *Wcell = lattice->W + lattice->offsets[icell];

                for(int64_t i=0;i<ncell;i++) {
                    int64_t j = iord[i];
                    Xcell[i] = lattice->X[lattice->offsets[icell] + j];
                    Ycell[i] = lattice->Y[lattice->offsets[icell] + j];
                    if(W != NULL) Wcell[i] = lattice->W[lattice->offsets[icell] + j];
                }
            }

            free(iord);
        }
    }
    
    if(options->verbose) {
      struct timeval t1;
      gettimeofday(&t1,NULL);
      fprintf(stderr," Time taken = %7.3lf sec\n",ADD_DIFF_TIME(t0,t1));
    }

    return EXIT_SUCCESS;
}


struct cell_pair *generate_cell_pairs(
    int64_t *ncell_pairs,
    const cellarray *lattice1,
    const cellarray *lattice2,
    const int xbin_refine_factor, const int ybin_refine_factor, const int zbin_refine_factor,
    const DOUBLE xwrap, const DOUBLE ywrap, const DOUBLE zwrap,
    const DOUBLE max_3D_sep, const DOUBLE max_2D_sep, const DOUBLE max_1D_sep,
    const int enable_min_sep_opt,
    const int autocorr,
    const int periodic_x, const int periodic_y, const int periodic_z
    ){

    const int64_t nmesh_x = lattice1->nmesh_x;
    const int64_t nmesh_y = lattice1->nmesh_y;
    const int64_t nmesh_z = lattice1->nmesh_z;

    const int64_t nx_ngb = 2*xbin_refine_factor + 1;
    const int64_t ny_ngb = 2*ybin_refine_factor + 1;
    const int64_t nz_ngb = 2*zbin_refine_factor + 1;
    const int64_t max_ngb_cells = nx_ngb * ny_ngb * nz_ngb - 1; // -1 for self
    
    const int any_periodic = periodic_x || periodic_y || periodic_z;

    const int64_t num_self_pairs = lattice1->tot_ncells;
    const int64_t num_nonself_pairs = lattice1->tot_ncells * max_ngb_cells;

    const int64_t max_num_cell_pairs = num_self_pairs + num_nonself_pairs;
    int64_t num_cell_pairs = 0;
    struct cell_pair *all_cell_pairs = my_malloc(sizeof(*all_cell_pairs), max_num_cell_pairs);
    XRETURN(all_cell_pairs != NULL, NULL,
            "Error: Could not allocate memory for storing all the cell pairs. "
            "Reducing bin refine factors might help. Requested for %"PRId64" elements "
            "with each element of size %zu bytes\n", max_num_cell_pairs, sizeof(*all_cell_pairs));


    /* Under periodic boundary conditions + small nmesh_x/y/z, the periodic wrapping would cause
       the same cell to be included as an neighbour cell from both the left and the right side (i.e.,
       when including cells with -bin_refine_factor, and when including cells up to +bin_refine_factor)

       Previously this would throw an error, but we can simply not add the duplicate cells.

       Raised in issue# 192 (https://github.com/manodeep/Corrfunc/issues/192)

       MS: 23/8/2019
     */
    const int check_for_duplicate_ngb_cells = ( any_periodic &&
                                                (nmesh_x < (2*xbin_refine_factor + 1) ||
                                                 nmesh_y < (2*ybin_refine_factor + 1) ||
                                                 nmesh_z < (2*zbin_refine_factor + 1))  ) ? 1:0;
    for(int64_t icell=0;icell<lattice1->tot_ncells;icell++) {
        int64_t first_N = lattice1->offsets[icell+1] - lattice1->offsets[icell];
        if(first_N == 0) continue;
        const int iz = icell % nmesh_z;
        const int ix = icell / (nmesh_y * nmesh_z );
        const int iy = (icell - iz - ix*nmesh_z*nmesh_y)/nmesh_z;
        XRETURN(icell == (ix * nmesh_y * nmesh_z + iy * nmesh_z + (int64_t) iz), NULL,
            ANSI_COLOR_RED"BUG: Index reconstruction is wrong. icell = %"PRId64" reconstructed index = %"PRId64  ANSI_COLOR_RESET "\n",
                icell, (ix * nmesh_y * nmesh_z + iy * nmesh_z + (int64_t) iz));

        int64_t num_ngb_this_cell = 0;
        for(int iix=-xbin_refine_factor;iix<=xbin_refine_factor;iix++){
            const int periodic_ix = (ix + iix + nmesh_x) % nmesh_x;
            const int non_periodic_ix = ix + iix;
            const int iiix = (periodic_x == 1) ? periodic_ix:non_periodic_ix;
            if(iiix < 0 || iiix >= nmesh_x) continue;
            const DOUBLE off_xwrap = ((ix + iix) >= 0) && ((ix + iix) < nmesh_x) ? 0.0: ((ix+iix) < 0 ? xwrap:-xwrap);

            for(int iiy=-ybin_refine_factor;iiy<=ybin_refine_factor;iiy++) {
                const int periodic_iy = (iy + iiy + nmesh_y) % nmesh_y;
                const int non_periodic_iy = iy + iiy;
                const int iiiy = (periodic_y == 1) ? periodic_iy:non_periodic_iy;
                if(iiiy < 0 || iiiy >= nmesh_y) continue;
                const DOUBLE off_ywrap = ((iy + iiy) >= 0) && ((iy + iiy) < nmesh_y) ? 0.0: ((iy+iiy) < 0 ? ywrap:-ywrap);
                for(int64_t iiz=-zbin_refine_factor;iiz<=zbin_refine_factor;iiz++){
                    const int periodic_iz = (iz + iiz + nmesh_z) % nmesh_z;
                    const int non_periodic_iz = iz + iiz;
                    const int iiiz = (periodic_z == 1) ? periodic_iz:non_periodic_iz;
                    if(iiiz < 0 || iiiz >= nmesh_z) continue;

                    const DOUBLE off_zwrap = ((iz + iiz) >= 0) && ((iz + iiz) < nmesh_z) ? 0.0: ((iz+iiz) < 0 ? zwrap:-zwrap);
                    const int64_t icell2 = iiiz + (int64_t) nmesh_z*iiiy + nmesh_z*nmesh_y*iiix;

                    //Since we are creating a giant array with all possible cell-pairs, we need
                    //to account for cases where an auto-correlation is occurring within the same cell.
                    //To do so, means 'same_cell', 'min_dx/dy/dz', and 'closest_x1/y1/z1' must all be
                    //set here. Also, if the second cell has no particles, then just skip it
                    int64_t second_N = lattice2->offsets[icell2+1] - lattice2->offsets[icell2];
                    if((autocorr == 1 && icell2 > icell) || second_N == 0) {
                        continue;
                    }

                    //Check if the ngb-cell has already been added - can happen under periodic boundary
                    //conditions, with small value of nmesh_x/y/z (ie large Rmax relative to BoxSize)
                    if(check_for_duplicate_ngb_cells) {
                        CHECK_AND_CONTINUE_FOR_DUPLICATE_NGB_CELLS(icell, icell2, off_xwrap, off_ywrap, off_zwrap, num_cell_pairs, num_ngb_this_cell, all_cell_pairs);
                    }

                    DOUBLE closest_x1 = ZERO, closest_y1 = ZERO, closest_z1 = ZERO;
                    DOUBLE min_dx = ZERO, min_dy = ZERO, min_dz = ZERO;
                    if(enable_min_sep_opt) {
                        /* Adjust for periodic boundary conditions */
                        const DOUBLE x_low = lattice1->xbounds[0][icell] + off_xwrap, x_hi = lattice1->xbounds[1][icell] + off_xwrap;
                        const DOUBLE y_low = lattice1->ybounds[0][icell] + off_ywrap, y_hi = lattice1->ybounds[1][icell] + off_ywrap;
                        const DOUBLE z_low = lattice1->zbounds[0][icell] + off_zwrap, z_hi = lattice1->zbounds[1][icell] + off_zwrap;

                        closest_x1 = iix < 0 ? x_low:(iix > 0 ? x_hi:ZERO);
                        closest_y1 = iiy < 0 ? y_low:(iiy > 0 ? y_hi:ZERO);
                        closest_z1 = iiz < 0 ? z_low:(iiz > 0 ? z_hi:ZERO);

                        //chose lower bound if secondary cell is to the left (smaller x), else upper
                        const DOUBLE first_x  = iix < 0 ? x_low:x_hi;//second condition also contains iix==0
                        //chose upper bound if primary cell is to the right (larger x), else lower
                        const DOUBLE second_x = iix < 0 ? lattice2->xbounds[1][icell2]:lattice2->xbounds[0][icell2];//second condition also contains iix==0
                        min_dx = iix != 0 ? (first_x - second_x):ZERO;//ensure min_dx == 0 for iix ==0

                        //repeat for min_dy
                        const DOUBLE first_y  = iiy < 0 ? y_low:y_hi;//second condition also contains iix==0
                        const DOUBLE second_y = iiy < 0 ? lattice2->ybounds[1][icell2]:lattice2->ybounds[0][icell2];//second condition also contains iix==0
                        min_dy = iiy != 0  ? (first_y - second_y):ZERO;//ensure min_dy == 0 for iiy ==0

                        //repeat for min_dz
                        const DOUBLE first_z  = iiz < 0 ? z_low:z_hi;//second condition also contains iix==0
                        const DOUBLE second_z = iiz < 0 ? lattice2->zbounds[1][icell2]:lattice2->zbounds[0][icell2];//second condition also contains iix==0
                        min_dz = iiz != 0 ? (first_z - second_z):ZERO;//ensure min_dz == 0 for iiz ==0

                        if(max_3D_sep > 0) {
                            const DOUBLE sqr_min_sep_cells = min_dx*min_dx + min_dy*min_dy + min_dz*min_dz;
                            if(sqr_min_sep_cells >= max_3D_sep*max_3D_sep) {
                                continue;
                            }
                        }/* end of condition for max_3D_sep */


                        /* in a separate if (rather than an else with the previous if for max_3D_sep)
                           so that conditions can be chained */
                        if(max_2D_sep > 0) {
                            const DOUBLE sqr_min_sep_cells = min_dx*min_dx + min_dy*min_dy;
                            if(sqr_min_sep_cells >= max_2D_sep*max_2D_sep) {
                                continue;
                            }
                        } /* end of if condition for max_2D_sep */

                        /* in a separate if (rather than an else with the previous if for max_3D_sep)
                           so that conditions can be chained */
                        if(max_1D_sep > 0 && iiz != 0) {
                            const DOUBLE sqr_min_sep_cells = min_dz*min_dz;
                            if(sqr_min_sep_cells >= max_1D_sep*max_1D_sep) {
                                continue;
                            }
                        } /* end of if condition for max_1D_sep */
                    } /* end of if condition for enable_min_sep_opt*/


                    XRETURN(num_cell_pairs < max_num_cell_pairs, NULL,
                            "Error: Assigning this existing cell-pair would require accessing invalid memory.\n"
                            "Expected that the total number of cell pairs can be at most %"PRId64" but "
                            "currently have number of cell pairs = %"PRId64"\n", max_num_cell_pairs, num_cell_pairs);
                    //If we have reached here, then this cell *MIGHT* have a pair. We
                    //need to add a cell-pair to the array of all cell-pairs
                    struct cell_pair *this_cell_pair = &all_cell_pairs[num_cell_pairs];
                    this_cell_pair->cellindex1 = icell;
                    this_cell_pair->cellindex2 = icell2;

                    this_cell_pair->xwrap = off_xwrap;
                    this_cell_pair->ywrap = off_ywrap;
                    this_cell_pair->zwrap = off_zwrap;

                    this_cell_pair->min_dx = min_dx;
                    this_cell_pair->min_dy = min_dy;
                    this_cell_pair->min_dz = min_dz;

                    this_cell_pair->closest_x1 = closest_x1;
                    this_cell_pair->closest_y1 = closest_y1;
                    this_cell_pair->closest_z1 = closest_z1;

                    this_cell_pair->same_cell = (autocorr == 1 && icell2 == icell) ? 1:0;

                    num_cell_pairs++;
                    num_ngb_this_cell++;
                } //looping over neighbours in Z
            } //looping over neighbours along Y
        }//looping over neighbours along X
    }

    *ncell_pairs = num_cell_pairs;
    return all_cell_pairs;
}

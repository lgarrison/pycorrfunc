#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "function_precision.h"
#include "utils.h"

#include "gridlink_utils.h"
#include "gridlink_mocks_impl.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef CONVERT_3D_INDEX_TO_LINEAR
#define CONVERT_3D_INDEX_TO_LINEAR(ix, iy, iz, nx, ny, nz)           {ix*ny*nz + iy*nz + iz}
#endif

void free_cellarray_mocks(cellarray_mocks *lattice, const int64_t totncells)
{
    //In the case where we have not requested to copy particles.
    //the memory address for the 'original_indices' from gridlink will be
    //at the first cell. We need to free this memory to avoid a leak
    if(lattice[0].owns_memory == 0) {
        free(lattice[0].original_index);
    } else {
        for(int64_t i=0;i<totncells;i++){
            if(lattice[i].owns_memory) {
                free(lattice[i].x);
                free(lattice[i].y);
                free(lattice[i].z);
                free(lattice[i].original_index);
                for(int w = 0; w < lattice[i].weights.num_weights; w++){
                    free(lattice[i].weights.weights[w]);
                }
            }
        }
    }
    free(lattice);
}

cellarray_mocks * gridlink_mocks(const int64_t NPART,
                                               DOUBLE *X, DOUBLE *Y, DOUBLE *Z,
                                               const weight_struct *WEIGHTS,
                                               const DOUBLE xmin, const DOUBLE xmax,
                                               const DOUBLE ymin, const DOUBLE ymax,
                                               const DOUBLE zmin, const DOUBLE zmax,
                                               const DOUBLE max_x_size,
                                               const DOUBLE max_y_size,
                                               const DOUBLE max_z_size,
                                               const int xbin_refine_factor,
                                               const int ybin_refine_factor,
                                               const int zbin_refine_factor,
                                               const int sort_on_z,
                                               int *nlattice_x,
                                               int *nlattice_y,
                                               int *nlattice_z,
                                               const config_options *options)
{
    int nmesh_x=0,nmesh_y=0,nmesh_z=0;
    struct timeval t0;
    if(options->verbose) {
      gettimeofday(&t0,NULL);
    }

    DOUBLE xbinsize=ZERO, ybinsize=ZERO, zbinsize=ZERO;
    DOUBLE wrap=ZERO;
    const int xstatus = get_binsize(xmax - xmin, wrap, max_x_size, xbin_refine_factor, options->max_cells_per_dim, &xbinsize, &nmesh_x);
    const int ystatus = get_binsize(ymax - ymin, wrap, max_y_size, ybin_refine_factor, options->max_cells_per_dim, &ybinsize, &nmesh_y);
    const int zstatus = get_binsize(zmax - zmin, wrap, max_z_size, zbin_refine_factor, options->max_cells_per_dim, &zbinsize, &nmesh_z);
    if(xstatus != EXIT_SUCCESS || ystatus != EXIT_SUCCESS || zstatus != EXIT_SUCCESS) {
      fprintf(stderr,"Received xstatus = %d ystatus = %d zstatus = %d. Error\n", xstatus, ystatus, zstatus);
      return NULL;
    }

    const int64_t totncells = (int64_t) nmesh_x * (int64_t) nmesh_y * (int64_t) nmesh_z;

    const DOUBLE xdiff = xmax-xmin;
    const DOUBLE ydiff = ymax-ymin;
    const DOUBLE zdiff = zmax-zmin;

    const DOUBLE cell_volume=xbinsize*ybinsize*zbinsize;
    const DOUBLE box_volume=xdiff*ydiff*zdiff;
    int64_t expected_n=(int64_t)(NPART*cell_volume/box_volume*MEMORY_INCREASE_FAC);
    expected_n=expected_n < 2 ? 2:expected_n;

    if(options->verbose) {
      fprintf(stderr,"In %s> Running with [nmesh_x, nmesh_y, nmesh_z]  = %d,%d,%d. ",__FUNCTION__,nmesh_x,nmesh_y,nmesh_z);
    }

    cellarray_mocks *lattice  = (cellarray_mocks *) my_malloc(sizeof(*lattice), totncells);
    int64_t *all_cell_indices = NULL;
    int64_t *original_indices =  NULL;
    int64_t *nallocated = NULL;//to keep track of how many particles have been allocated per cell (only when creating a copy of particle positions)
    if(options->copy_particles) {
        nallocated = (int64_t *) my_calloc(sizeof(*nallocated), totncells);
    } else {
        all_cell_indices = (int64_t *) my_malloc(sizeof(*all_cell_indices), NPART);
        original_indices = (int64_t *) my_malloc(sizeof(*original_indices), NPART);
    }
    if(lattice == NULL ||
       (options->copy_particles == 0 && all_cell_indices == NULL) ||
       (options->copy_particles == 0 && original_indices == NULL) ||
       (options->copy_particles && nallocated == NULL)) {

        free(lattice);free(nallocated);free(all_cell_indices);free(original_indices);
        fprintf(stderr,"Error: In %s> Could not allocate memory for creating the lattice and associated arrays\n", __FUNCTION__);
        return NULL;
    }

    for (int64_t icell=0;icell<totncells;icell++) {
        lattice[icell].nelements=0;
        lattice[icell].owns_memory = 0;
        lattice[icell].weights.num_weights = (WEIGHTS == NULL) ? 0 : WEIGHTS->num_weights;
        lattice[icell].xbounds[0] = MAX_POSITIVE_FLOAT;
        lattice[icell].xbounds[1] = -MAX_POSITIVE_FLOAT;
        lattice[icell].ybounds[0] = MAX_POSITIVE_FLOAT;
        lattice[icell].ybounds[1] = -MAX_POSITIVE_FLOAT;
        lattice[icell].zbounds[0] = MAX_POSITIVE_FLOAT;
        lattice[icell].zbounds[1] = -MAX_POSITIVE_FLOAT;

        if(options->copy_particles) {
            lattice[icell].owns_memory = 1;
            lattice[icell].original_index = my_malloc(sizeof(lattice[icell].original_index[0]), expected_n);
            XRETURN(lattice[icell].original_index != NULL, NULL, "Error in %s> Could not allocate memory for holding the original "
                    "indices for particles. (allocating for %"PRId64" particles \n", __FUNCTION__, expected_n);

            const size_t memsize=sizeof(DOUBLE);
            lattice[icell].x = my_malloc(memsize,expected_n);//This allocates extra and is wasteful
            lattice[icell].y = my_malloc(memsize,expected_n);//This allocates extra and is wasteful
            lattice[icell].z = my_malloc(memsize,expected_n);//This allocates extra and is wasteful

            // Now do the same for the weights
            int w_alloc_status = EXIT_SUCCESS;
            for(int w = 0; w < lattice[icell].weights.num_weights; w++){
                lattice[icell].weights.weights[w] = (DOUBLE *) my_malloc(memsize, expected_n);
                if(lattice[icell].weights.weights[w] == NULL){
                    w_alloc_status = EXIT_FAILURE;
                }
            }

            if(lattice[icell].x == NULL || lattice[icell].y == NULL || lattice[icell].z == NULL || w_alloc_status == EXIT_FAILURE) {
                for(int64_t j=icell;j>=0;j--) {
                    free(lattice[j].x);free(lattice[j].y);free(lattice[j].z);free(lattice[j].original_index);
                    for(int w = 0; w < lattice[icell].weights.num_weights; w++){
                        free(lattice[icell].weights.weights[w]);
                    }
                }
                free(nallocated);free(lattice);
                return NULL;
            }
            nallocated[icell] = expected_n;
        }//if condition when creating a copy of the particle positions
    }//end of loop over totncells

    const DOUBLE xinv=1.0/xbinsize;
    const DOUBLE yinv=1.0/ybinsize;
    const DOUBLE zinv=1.0/zbinsize;

    for (int64_t i=0;i<NPART;i++)  {
        int ix=(int)((X[i]-xmin)*xinv) ;
        int iy=(int)((Y[i]-ymin)*yinv) ;
        int iz=(int)((Z[i]-zmin)*zinv) ;

        if (ix>nmesh_x-1)  ix--;    /* this shouldn't happen, but . . . */
        if (iy>nmesh_y-1)  iy--;
        if (iz>nmesh_z-1)  iz--;
        XRETURN(X[i] >= xmin && X[i] <= xmax, NULL,
               "x[%"PRId64"] = %"REAL_FORMAT" must be within [%"REAL_FORMAT",%"REAL_FORMAT"]\n",
               i, X[i], xmin, xmax);
        XRETURN(Y[i] >= ymin && Y[i] <= ymax, NULL,
               "y[%"PRId64"] = %"REAL_FORMAT" must be within [%"REAL_FORMAT",%"REAL_FORMAT"]\n",
               i, Y[i], ymin, ymax);
        XRETURN(Z[i] >= zmin && Z[i] <= zmax, NULL,
               "z[%"PRId64"] = %"REAL_FORMAT" must be within [%"REAL_FORMAT",%"REAL_FORMAT"]\n",
               i, Z[i], zmin, zmax);

        XRETURN(ix >= 0 && ix < nmesh_x, NULL, "ix=%d must be within [0,%d)\n", ix, nmesh_x);
        XRETURN(iy >= 0 && iy < nmesh_y, NULL, "iy=%d must be within [0,%d)\n", iy, nmesh_y);
        XRETURN(iz >= 0 && iz < nmesh_z, NULL, "iz=%d must be within [0,%d)\n", iz, nmesh_z);

        const int64_t icell = CONVERT_3D_INDEX_TO_LINEAR(ix, iy, iz, nmesh_x, nmesh_y, nmesh_z);
        if(options->copy_particles == 0) {
            all_cell_indices[i] = icell;
            original_indices[i] = i;
        } else {

            //check if we need to reallocate
            if(lattice[icell].nelements == nallocated[icell]) {

                expected_n = nallocated[icell]*MEMORY_INCREASE_FAC;

                //In case expected_n is 1 or MEMORY_INCREASE_FAC is 1.
                //This way, we only increase by a very few particles
                // at a time. Smaller memory footprint
                while(expected_n <= nallocated[icell])
                    expected_n++;

                const size_t memsize=sizeof(DOUBLE);
                DOUBLE *posx=NULL, *posy=NULL, *posz=NULL;
                int64_t *orig_index = NULL;
                int w_alloc_status;
                do{
                    posx = my_realloc(lattice[icell].x ,memsize,expected_n,"lattice.x");
                    posy = my_realloc(lattice[icell].y ,memsize,expected_n,"lattice.y");
                    posz = my_realloc(lattice[icell].z ,memsize,expected_n,"lattice.z");
                    orig_index = my_realloc(lattice[icell].original_index , sizeof(*(lattice[icell].original_index)), expected_n,"lattice.original_index");

                    lattice[icell].x  = (posx == NULL)  ? lattice[icell].x:posx;
                    lattice[icell].y  = (posy == NULL)  ? lattice[icell].y:posy;
                    lattice[icell].z  = (posz == NULL)  ? lattice[icell].z:posz;
                    lattice[icell].original_index = (orig_index == NULL) ? lattice[icell].original_index:orig_index;

                    w_alloc_status = EXIT_SUCCESS;
                    for(int w = 0; w < lattice[icell].weights.num_weights; w++){
                        DOUBLE *newweights = (DOUBLE *) my_realloc(lattice[icell].weights.weights[w], memsize, expected_n, "lattice.weights");
                        if(newweights == NULL){
                            w_alloc_status = EXIT_FAILURE;
                        } else {
                            lattice[icell].weights.weights[w] = newweights;
                        }
                    }

                    if(posx == NULL || posy == NULL || posz == NULL || orig_index == NULL || w_alloc_status == EXIT_FAILURE) {
                        expected_n--;
                    }
                } while(expected_n > nallocated[icell] && (posx == NULL ||
                                                           posy == NULL ||
                                                           posz == NULL ||
                                                           orig_index == NULL ||
                                                           w_alloc_status == EXIT_FAILURE));

                if(expected_n == nallocated[icell]) {
                    /*realloc failed. free memory and return */
                    fprintf(stderr,"In %s> Reallocation failed,  randomly subsampling the input particle set (currently at %"PRId64" particles) might help\n",
                            __FUNCTION__, NPART);
                    fprintf(stderr,"posx = %p posy = %p posz = %p \n", posx, posy, posz);
                    free_cellarray_mocks(lattice, totncells);
                    free(nallocated);
                    return NULL;
                }
                nallocated[icell] = expected_n;
            }
            XRETURN(lattice[icell].nelements < nallocated[icell], NULL,
                    ANSI_COLOR_RED"BUG: lattice[%"PRId64"].nelements = %"PRId64" must be less than allocated memory = %"PRId64  ANSI_COLOR_RESET "\n",
                    icell, lattice[icell].nelements, nallocated[icell]);

            const int64_t ipos = lattice[icell].nelements;
            lattice[icell].x[ipos] = X[i];
            lattice[icell].y[ipos] = Y[i];
            lattice[icell].z[ipos] = Z[i];
            lattice[icell].original_index[ipos] = i;
            for(int w = 0; w < lattice[icell].weights.num_weights; w++){
                lattice[icell].weights.weights[w][ipos] = ((DOUBLE *) (WEIGHTS->weights[w]))[i];
            }
        }
        lattice[icell].xbounds[0] = X[i] < lattice[icell].xbounds[0] ? X[i]:lattice[icell].xbounds[0];
        lattice[icell].ybounds[0] = Y[i] < lattice[icell].ybounds[0] ? Y[i]:lattice[icell].ybounds[0];
        lattice[icell].zbounds[0] = Z[i] < lattice[icell].zbounds[0] ? Z[i]:lattice[icell].zbounds[0];

        lattice[icell].xbounds[1] = X[i] > lattice[icell].xbounds[1] ? X[i]:lattice[icell].xbounds[1];
        lattice[icell].ybounds[1] = Y[i] > lattice[icell].ybounds[1] ? Y[i]:lattice[icell].ybounds[1];
        lattice[icell].zbounds[1] = Z[i] > lattice[icell].zbounds[1] ? Z[i]:lattice[icell].zbounds[1];

        lattice[icell].nelements++;
    }

    if(options->copy_particles) {
        //All the particle positions have already been copied -> do not need to re-allocate any more
        //You can free the extra memory reserved by the mallocs by looping over totncells
        //and doing a realloc(lattice[cellindex].x,sizeof(DOUBLE),lattice[cellindex].nelements,"lattice.x")
        free(nallocated);
    } else {
        // We have been told to work with the particle positions in-place i.e., not create a copy
        // of the particle positions within the lattice. Therefore, now we have to sort the
        // input particle positions to get them to be contiguous in their respective 3D cell
        if(sizeof(*(lattice->original_index)) != sizeof(*original_indices)) {
            fprintf(stderr, "Error: In %s> The array to track the indices of input particle positions "
                    "should be the same size as the indices themselves\n", __FUNCTION__);
            fprintf(stderr,"Perhaps check that these two variables are the same type\n");
            fprintf(stderr,"'original_index' within the 'cellarray', defined in 'cellarray.h' and \n");
            fprintf(stderr,"'original_indices' defined within function '%s' in file '%s'\n", __FUNCTION__, __FILE__);
            return NULL;
        }

        // First sort all particles into their respective cell-indices
        // also simultaneously swap the other associated array
#define MULTIPLE_ARRAY_EXCHANGER(type,a,i,j) {                          \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, X, i, j);            \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, Y, i, j);            \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, Z, i, j);            \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(int64_t, original_indices, i, j); \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(int64_t, all_cell_indices, i, j); \
            for(int w = 0; w < WEIGHTS->num_weights; w++) {             \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, ((DOUBLE *) WEIGHTS->weights[w]), i, j); \
            }                                                           \
        }

        //If the input-array is sorted exactly, then the quicksort will become a very-slow O(N^2)
        //Try to protect the user.
        int64_t num_sorted = 1;//an array containing exactly one element is always sorted
        for(int64_t ii=0;ii<NPART-1;ii++) {
            //increment by 1 if the next element is greater than or equal to current
            //decrement by 1 if the next element is smaller
            num_sorted += (all_cell_indices[ii+1] >= all_cell_indices[ii]) ? +1:-1;
        }

        //Check if there is any sorting to do
        //If the input array is already sorted, then all_cell_indices will also be sorted
        //which would result in num_sorted == NPART
        if(num_sorted < NPART) {
            //Since the particles might be coming from an already sorted array - quicksort might degenerate to
            //a O(N^2) process -- heap-sort might be safer.
            if(options->use_heap_sort || num_sorted > FRACTION_SORTED_REQD_TO_HEAP_SORT * NPART) {
                SGLIB_ARRAY_HEAP_SORT(int64_t, all_cell_indices, NPART, SGLIB_NUMERIC_COMPARATOR, MULTIPLE_ARRAY_EXCHANGER);
            } else {
                SGLIB_ARRAY_QUICK_SORT(int64_t, all_cell_indices, NPART, SGLIB_NUMERIC_COMPARATOR, MULTIPLE_ARRAY_EXCHANGER);
            }
        }

        //Now the particles are sorted contiguously according to their respective cellindex
        //We need to fix up the x/y/z pointers at the beginning of each cell to point to the right places
#undef MULTIPLE_ARRAY_EXCHANGER

        free(all_cell_indices);//Done with re-ordering the particles

        int64_t nelements_so_far = 0;
        for(int64_t icell=0;icell<totncells;icell++) {
            cellarray_mocks *first=&(lattice[icell]);
            first->x = X + nelements_so_far;//take the base pointer address and add however many particles that have appeared summed across all previous cells
            first->y = Y + nelements_so_far;
            first->z = Z + nelements_so_far;
            first->original_index = original_indices + nelements_so_far;
            for(int w = 0; w < WEIGHTS->num_weights; w++) {
                first->weights.weights[w] = ((DOUBLE *) WEIGHTS->weights[w]) + nelements_so_far;
            }
            nelements_so_far += first->nelements;
        }
        XRETURN(nelements_so_far == NPART, NULL,
                "Error in %s> Expected to assign all particles = %"PRId64" into cells but only seem "
                "to have assigned %"PRId64". Perhaps, there are some edge cases with floating point accuracy\n",
                __FUNCTION__, NPART, nelements_so_far);

    }//end of options->copy_particles == 0

    //Now the cells are all setup correctly irrespective of
    //copy_particles setting.


    /* Do we need to sort the particles in Z ? */
    if(sort_on_z) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic) num_threads(options->numthreads)
#endif
        for(int64_t icell=0;icell<totncells;icell++) {
            const cellarray_mocks *first=&(lattice[icell]);
            if(first->nelements == 0) continue;

#define MULTIPLE_ARRAY_EXCHANGER(type,a,i,j) {                          \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->x, i, j); \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->y, i, j); \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->z, i, j); \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(int64_t, first->original_index, i, j); \
                for(int w = 0; w < first->weights.num_weights; w++){    \
                    SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->weights.weights[w],i,j); \
                }                                                       \
            }

            //If the input-array is sorted exactly, then the quicksort will become a very-slow O(N^2)
            //Try to protect the user.
            int64_t num_sorted = 1;//an array containing exactly one element is always sorted
            for(int64_t ii=0;ii<first->nelements-1;ii++) {
                //increment by 1 if the next element is greater than or equal to current
                //decrement by 1 if the next element is smaller
                num_sorted += (first->z[ii+1] >= first->z[ii]) ? +1:-1;
            }

            //Check if there is any sorting to do
            //If the input array is already sorted, then all_cell_indices will also be sorted
            //which would result in num_sorted == first->nelements
            if(num_sorted < first->nelements) {
                //Since the particles might be coming from an already sorted array - quicksort might degenerate to
                //a O(N^2) process -- heap-sort might be safer.
                if(options->use_heap_sort || num_sorted > FRACTION_SORTED_REQD_TO_HEAP_SORT * first->nelements) {
                    SGLIB_ARRAY_HEAP_SORT(DOUBLE, first->z, first->nelements, SGLIB_NUMERIC_COMPARATOR , MULTIPLE_ARRAY_EXCHANGER);
                } else {
                    SGLIB_ARRAY_QUICK_SORT(DOUBLE, first->z, first->nelements, SGLIB_NUMERIC_COMPARATOR , MULTIPLE_ARRAY_EXCHANGER);
                }
            }//if sorting is required
#undef MULTIPLE_ARRAY_EXCHANGER
        }//loop over cells
    }

    //You can free the extra memory reserved by the mallocs by looping over totncells and doing a realloc(lattice[icell].x,sizeof(DOUBLE),lattice[icell].nelements,"lattice.x")
    *nlattice_x=nmesh_x;
    *nlattice_y=nmesh_y;
    *nlattice_z=nmesh_z;
    if(options->verbose) {
      struct timeval t1;
      gettimeofday(&t1,NULL);
      fprintf(stderr," Time taken = %7.3lf sec\n",ADD_DIFF_TIME(t0,t1));
    }

    return lattice;
}


struct cell_pair * generate_cell_pairs_mocks(cellarray_mocks *lattice1,
                                                           cellarray_mocks *lattice2,
                                                           const int64_t totncells,
                                                           int64_t *ncell_pairs,
                                                           const int xbin_refine_factor, const int ybin_refine_factor, const int zbin_refine_factor,
                                                           const int nmesh_x, const int nmesh_y, const int nmesh_z,
                                                           const DOUBLE max_3D_sep,
                                                           const int enable_min_sep_opt,
                                                           const int autocorr)
{
    const int64_t nx_ngb = 2*xbin_refine_factor + 1;
    const int64_t ny_ngb = 2*ybin_refine_factor + 1;
    const int64_t nz_ngb = 2*zbin_refine_factor + 1;
    const int64_t max_ngb_cells = nx_ngb * ny_ngb * nz_ngb - 1; // -1 for self

    if( ! (autocorr == 0 || autocorr == 1) ) {
        fprintf(stderr,"Error: Strange value of autocorr = %d. Expected to receive either 1 (auto-correlations) or 0 (cross-correlations)\n", autocorr);
        return NULL;
    }
    const int64_t num_self_pairs = totncells;
    const int64_t num_nonself_pairs = totncells * max_ngb_cells / (autocorr + 1);

    const int64_t max_num_cell_pairs = num_self_pairs + num_nonself_pairs;
    int64_t num_cell_pairs = 0;
    struct cell_pair *all_cell_pairs = my_malloc(sizeof(*all_cell_pairs), max_num_cell_pairs);
    XRETURN(all_cell_pairs != NULL, NULL,
            "Error: Could not allocate memory for storing all the cell pairs. "
            "Reducing bin refine factors might help. Requested for %"PRId64" elements "
            "with each element of size %zu bytes\n", max_num_cell_pairs, sizeof(*all_cell_pairs));

    for(int64_t icell=0;icell<totncells;icell++) {
        cellarray_mocks *first = &(lattice1[icell]);
        if(first->nelements == 0) continue;
        const int iz = icell % nmesh_z;
        const int ix = icell / (nmesh_y * nmesh_z );
        const int iy = (icell - iz - ix*nmesh_z*nmesh_y)/nmesh_z;
        XRETURN(icell == (ix * nmesh_y * nmesh_z + iy * nmesh_z + (int64_t) iz), NULL,
                ANSI_COLOR_RED"BUG: Index reconstruction is wrong. icell = %"PRId64" reconstructed index = %"PRId64  ANSI_COLOR_RESET "\n",
                icell, (ix * nmesh_y * nmesh_z + iy * nmesh_z + (int64_t) iz));

        for(int iix=-xbin_refine_factor;iix<=xbin_refine_factor;iix++){
            const int iiix = ix + iix;
            if(iiix < 0 || iiix >= nmesh_x) continue;

            for(int iiy=-ybin_refine_factor;iiy<=ybin_refine_factor;iiy++) {
                const int iiiy = iy + iiy;
                if(iiiy < 0 || iiiy >= nmesh_y) continue;

                for(int64_t iiz=-zbin_refine_factor;iiz<=zbin_refine_factor;iiz++){
                    const int iiiz = iz + iiz;
                    if(iiiz < 0 || iiiz >= nmesh_z) continue;

                    const int64_t icell2 = iiiz + (int64_t) nmesh_z*iiiy + nmesh_z*nmesh_y*iiix;

                    //Since we are creating a giant array with all possible cell-pairs, we need
                    //to account for cases where an auto-correlation is occurring within the same cell.
                    //To do so, means 'same_cell', 'min_dx/dy/dz', and 'closest_x1/y1/z1' must all be
                    //set here. Also, if the second cell has no particles, then just skip it
                    if((autocorr == 1 && icell2 > icell) || lattice2[icell2].nelements == 0) {
                        continue;
                    }

                    cellarray_mocks *second = &(lattice2[icell2]);
                    DOUBLE closest_x1 = ZERO, closest_y1 = ZERO, closest_z1 = ZERO;
                    DOUBLE min_dx = ZERO, min_dy = ZERO, min_dz = ZERO;

                    if(enable_min_sep_opt && max_3D_sep > 0) {
                        /* Check for periodic boundary conditions....etc*/
                        const DOUBLE x_low = first->xbounds[0], x_hi = first->xbounds[1];
                        const DOUBLE y_low = first->ybounds[0], y_hi = first->ybounds[1];
                        const DOUBLE z_low = first->zbounds[0], z_hi = first->zbounds[1];

                        closest_x1 = iix < 0 ? x_low:(iix > 0 ? x_hi:ZERO);
                        closest_y1 = iiy < 0 ? y_low:(iiy > 0 ? y_hi:ZERO);
                        closest_z1 = iiz < 0 ? z_low:(iiz > 0 ? z_hi:ZERO);

                        //chose lower bound if secondary cell is to the left (smaller x), else upper
                        const DOUBLE first_x  = iix < 0 ? x_low:x_hi;//second condition also contains iix==0
                        //chose upper bound if primary cell is to the right (larger x), else lower
                        const DOUBLE second_x = iix < 0 ? second->xbounds[1]:second->xbounds[0];//second condition also contains iix==0
                        min_dx = iix != 0 ? (first_x - second_x):ZERO;//ensure min_dx == 0 for iix ==0

                        //repeat for min_dy
                        const DOUBLE first_y  = iiy < 0 ? y_low:y_hi;//second condition also contains iiy==0
                        const DOUBLE second_y = iiy < 0 ? second->ybounds[1]:second->ybounds[0];//second condition also contains iiy==0
                        min_dy = iiy != 0  ? (first_y - second_y):ZERO;//ensure min_dy == 0 for iiy ==0

                        //repeat for min_dz
                        const DOUBLE first_z  = iiz < 0 ? z_low:z_hi;//second condition also contains iiz==0
                        const DOUBLE second_z = iiz < 0 ? second->zbounds[1]:second->zbounds[0];//second condition also contains iiz==0
                        min_dz = iiz != 0 ? (first_z - second_z):ZERO;//ensure min_dz == 0 for iiz ==0

                        const DOUBLE sqr_min_sep_cells = min_dx*min_dx + min_dy*min_dy + min_dz*min_dz;
                        if(sqr_min_sep_cells >= max_3D_sep*max_3D_sep) {
                            continue;
                        }
                    }/* end of if condition for enable_min_sep_opt and max_3D_sep > 0 */

                    //If we have reached here, then this cell *MIGHT* have a pair. We
                    //need to add a cell-pair to the array of all cell-pairs
                    XRETURN(num_cell_pairs < max_num_cell_pairs, NULL,
                            "Error: Assigning this existing cell-pair would require accessing invalid memory.\n"
                            "Expected that the total number of cell pairs can be at most %"PRId64" but "
                            "currently have number of cell pairs = %"PRId64"\n", max_num_cell_pairs, num_cell_pairs);
                    struct cell_pair *this_cell_pair = &all_cell_pairs[num_cell_pairs];
                    this_cell_pair->cellindex1 = icell;
                    this_cell_pair->cellindex2 = icell2;

                    this_cell_pair->min_dx = min_dx;
                    this_cell_pair->min_dy = min_dy;
                    this_cell_pair->min_dz = min_dz;

                    this_cell_pair->closest_x1 = closest_x1;
                    this_cell_pair->closest_y1 = closest_y1;
                    this_cell_pair->closest_z1 = closest_z1;

                    this_cell_pair->same_cell = (autocorr == 1 && icell2 == icell) ? 1:0;

                    num_cell_pairs++;
                }
            }
        }
    }

    *ncell_pairs = num_cell_pairs;
    return all_cell_pairs;
}



/* Create the lattice simply based on declination */
cellarray_mocks * gridlink_mocks_theta_dec(const int64_t NPART,
                                                         DOUBLE *RA, DOUBLE *DEC,
                                                         DOUBLE *X, DOUBLE *Y, DOUBLE *Z, weight_struct *WEIGHTS,
                                                         const DOUBLE dec_min,const DOUBLE dec_max,
                                                         const DOUBLE max_dec_size,
                                                         const int dec_refine_factor,
                                                         const int sort_on_z,
                                                         const DOUBLE thetamax,
                                                         int64_t *totncells,
                                                         const config_options *options)
{
    int64_t expected_n;
    size_t totnbytes=0;
    const DOUBLE dec_diff = dec_max-dec_min;
    const DOUBLE inv_dec_diff = 1.0/dec_diff;

    struct timeval t0;
    if(options->verbose) {
        gettimeofday(&t0,NULL);
    }

    /* Input validation */
    XRETURN(thetamax > 0.0, NULL, "Minimum angular separation = %"REAL_FORMAT" must be positive\n", thetamax);
    XRETURN(dec_diff > 0.0, NULL, "All of the points can not be at the same declination. Declination difference = %"REAL_FORMAT" must be non-zero\n", dec_diff);
    XRETURN(NPART > 0, NULL, "Number of points =%"PRId64" must be >0\n", NPART);
    XRETURN(RA != NULL, NULL, "RA must be a valid array \n");
    XRETURN(DEC != NULL, NULL, "DEC must be a valid array \n");
    XRETURN(X != NULL, NULL, "X must be a valid array \n");
    XRETURN(Y != NULL, NULL, "Y must be a valid array \n");
    XRETURN(Z != NULL, NULL, "Z must be a valid array \n");
    XRETURN(dec_refine_factor >= 1, NULL, "DEC refine factor must be at least 1\n");
    XRETURN(totncells != NULL, NULL, "Pointer to return the total number of cells must be a valid address\n");
    XRETURN(options != NULL, NULL, "Structure containing code options must be a valid address\n");

    /* Protect against accidental changes to compile-time (macro) constant */
    XRETURN(MEMORY_INCREASE_FAC >= 1.0, NULL, "Memory increase factor = %lf must be >=1 \n",MEMORY_INCREASE_FAC);

    /* Find the max. number of declination cells that can be */
    const DOUBLE this_ngrid_dec = (dec_diff/thetamax < 1) ? 1:dec_diff/thetamax;
    const int this_ngrid_dec_int = ((int) this_ngrid_dec) * dec_refine_factor;
    int ngrid_dec = this_ngrid_dec_int > max_dec_size ? max_dec_size:this_ngrid_dec_int;
    XRETURN(ngrid_dec >= dec_refine_factor, NULL, "Number of grid cells = %d must be at least the declination refinement factor = %d\n",
            ngrid_dec, dec_refine_factor);

    *totncells = ngrid_dec;

    expected_n=(int64_t)( (NPART/(DOUBLE) (ngrid_dec)) *MEMORY_INCREASE_FAC);
    expected_n = expected_n < 2 ? 2:expected_n;//at least allocate 2 particles

    /*---Allocate-and-initialize-grid-arrays----------*/
    cellarray_mocks *lattice = (cellarray_mocks *) my_calloc(sizeof(*lattice),ngrid_dec);
    totnbytes += sizeof(*lattice)*ngrid_dec;

    int64_t *all_cell_indices = NULL;
    int64_t *original_indices =  NULL;
    int64_t *nallocated = NULL;//to keep track of how many particles have been allocated per cell (only when creating a copy of particle positions)

    if(options->copy_particles) {
        nallocated = (int64_t *) my_calloc(sizeof(*nallocated), *totncells);
        totnbytes += sizeof(*nallocated)*ngrid_dec;
    } else {
        all_cell_indices = (int64_t *) my_malloc(sizeof(*all_cell_indices), NPART);
        original_indices = (int64_t *) my_malloc(sizeof(*original_indices), NPART);
        totnbytes += sizeof(*all_cell_indices)*NPART + sizeof(*original_indices)*NPART;
    }
    if(lattice == NULL ||
       (options->copy_particles == 0 && all_cell_indices == NULL) ||
       (options->copy_particles == 0 && original_indices == NULL) ||
       (options->copy_particles && nallocated == NULL)) {

        free(lattice);free(nallocated);free(all_cell_indices);free(original_indices);
        fprintf(stderr,"Error: In %s> Could not allocate memory for creating the lattice and associated arrays\n", __FUNCTION__);
        return NULL;
    }

    for(int icell=0;icell<ngrid_dec;icell++) {
        lattice[icell].nelements = 0;
        lattice[icell].owns_memory = 0;
        lattice[icell].weights.num_weights = (WEIGHTS == NULL) ? 0 : WEIGHTS->num_weights;
        lattice[icell].xbounds[0] = MAX_POSITIVE_FLOAT;
        lattice[icell].xbounds[1] = -MAX_POSITIVE_FLOAT;
        lattice[icell].ybounds[0] = MAX_POSITIVE_FLOAT;
        lattice[icell].ybounds[1] = -MAX_POSITIVE_FLOAT;
        lattice[icell].zbounds[0] = MAX_POSITIVE_FLOAT;
        lattice[icell].zbounds[1] = -MAX_POSITIVE_FLOAT;
        lattice[icell].dec_bounds[0]=MAX_POSITIVE_FLOAT;
        lattice[icell].dec_bounds[1]=-MAX_POSITIVE_FLOAT;
        lattice[icell].ra_bounds[0]=MAX_POSITIVE_FLOAT;
        lattice[icell].ra_bounds[1]=-MAX_POSITIVE_FLOAT;

        if(options->copy_particles) {
            lattice[icell].owns_memory = 1;
            const size_t memsize = sizeof(DOUBLE);
            lattice[icell].x = my_malloc(memsize,expected_n);
            lattice[icell].y = my_malloc(memsize,expected_n);
            lattice[icell].z = my_malloc(memsize,expected_n);
            lattice[icell].original_index = my_malloc(sizeof(*(lattice[icell].original_index)), expected_n);

            int w_alloc_status = EXIT_SUCCESS;
            for(int w = 0; w < lattice[icell].weights.num_weights; w++){
                lattice[icell].weights.weights[w] = (DOUBLE *) my_malloc(memsize, expected_n);
                if(lattice[icell].weights.weights[w] == NULL){
                    w_alloc_status = EXIT_FAILURE;
                }
            }

            if(lattice[icell].x == NULL || lattice[icell].y == NULL ||
               lattice[icell].z == NULL || lattice[icell].original_index == NULL ||
               w_alloc_status == EXIT_FAILURE) {
                for(int k=icell;k>=0;k--) {
                    free(lattice[k].x);free(lattice[k].y);free(lattice[k].z);free(lattice[k].original_index);
                    for(int w = 0; w < lattice[k].weights.num_weights; w++){
                        free(lattice[k].weights.weights[w]);
                    }
                }
                free(lattice);
                return NULL;
            }

            nallocated[icell] = expected_n;
            totnbytes += 3 * memsize * expected_n + sizeof(int64_t) * expected_n; //assumes original_index is 64 bit
        }
    }

    /*---Loop-over-particles-and-build-grid-arrays----*/
    for(int64_t i=0;i<NPART;i++) {
        int idec = (int)(ngrid_dec*(DEC[i]-dec_min)*inv_dec_diff);
        if(idec >=ngrid_dec) idec--;
        XRETURN(idec >= 0 && idec < ngrid_dec, NULL,
                "idec (dec bin index) = %d must be within [0, %d)", idec, ngrid_dec);

        const int64_t icell = idec;
        if(options->copy_particles == 0) {
            all_cell_indices[i] = icell;
            original_indices[i] = i;
        } else {

            if(lattice[icell].nelements == nallocated[icell]) {
                expected_n = nallocated[icell]*MEMORY_INCREASE_FAC;
                while(expected_n <= lattice[icell].nelements) {
                    expected_n++;
                }

                const size_t memsize = sizeof(DOUBLE);
                DOUBLE *posx=NULL, *posy=NULL, *posz=NULL;
                int64_t *orig_index=NULL;
                int w_alloc_status;
                do {
                    posx = my_realloc(lattice[icell].x,memsize,expected_n,"lattice.x");
                    posy = my_realloc(lattice[icell].y,memsize,expected_n,"lattice.y");
                    posz = my_realloc(lattice[icell].z,memsize,expected_n,"lattice.z");
                    orig_index = my_realloc(lattice[icell].original_index , sizeof(*(lattice[icell].original_index)), expected_n,"lattice.original_index");

                    lattice[icell].x = (posx == NULL) ? lattice[icell].x:posx;
                    lattice[icell].y = (posy == NULL) ? lattice[icell].y:posy;
                    lattice[icell].z = (posz == NULL) ? lattice[icell].z:posz;
                    lattice[icell].original_index = (orig_index == NULL) ? lattice[icell].original_index:orig_index;

                    w_alloc_status = EXIT_SUCCESS;
                    for(int w = 0; w < lattice[icell].weights.num_weights; w++){
                        DOUBLE *newweights = (DOUBLE *) my_realloc(lattice[icell].weights.weights[w], memsize, expected_n, "lattice.weights");
                        if(newweights == NULL){
                            w_alloc_status = EXIT_FAILURE;
                        } else {
                            lattice[icell].weights.weights[w] = newweights;
                        }
                    }

                    if(posx == NULL || posy == NULL || posz == NULL || orig_index == NULL || w_alloc_status == EXIT_FAILURE) {
                        expected_n--;
                    }
                } while(expected_n > nallocated[icell] && (posx == NULL ||
                                                          posy == NULL ||
                                                          posz == NULL ||
                                                          orig_index == NULL ||
                                                          w_alloc_status == EXIT_FAILURE));

                if(expected_n == nallocated[icell]) {
                    /*realloc failed. free memory and return */
                    fprintf(stderr,"In %s> Reallocation failed,  randomly subsampling the input particle set (currently at %"PRId64" particles) might help\n",
                            __FUNCTION__, NPART);
                    fprintf(stderr,"posx = %p posy = %p posz = %p\n", posx, posy, posz);
                    free_cellarray_mocks(lattice, *totncells);
                    free(nallocated);
                    return NULL;
                }
                nallocated[icell] = expected_n;
            }

            XRETURN(lattice[icell].nelements < nallocated[icell],NULL,
                    ANSI_COLOR_RED"BUG: lattice[%d].nelements = %"PRId64" must be less than allocated memory = %"PRId64  ANSI_COLOR_RESET "\n",
                    idec, lattice[icell].nelements, nallocated[icell]);

            const int64_t ipos=lattice[icell].nelements;
            lattice[icell].x[ipos]  = X[i];
            lattice[icell].y[ipos]  = Y[i];
            lattice[icell].z[ipos]  = Z[i];
            lattice[icell].original_index[ipos] = i;
            for(int w = 0; w < lattice[icell].weights.num_weights; w++){
                lattice[icell].weights.weights[w][ipos] = ((DOUBLE *) WEIGHTS->weights[w])[i];
            }
        }

        //store particle bounds now
        lattice[icell].dec_bounds[0] = DEC[i] < lattice[icell].dec_bounds[0] ? DEC[i]:lattice[icell].dec_bounds[0];
        lattice[icell].dec_bounds[1] = DEC[i] > lattice[icell].dec_bounds[1] ? DEC[i]:lattice[icell].dec_bounds[1];

        lattice[icell].ra_bounds[0] = RA[i] < lattice[icell].ra_bounds[0] ? RA[i]:lattice[icell].ra_bounds[0];
        lattice[icell].ra_bounds[1] = RA[i] > lattice[icell].ra_bounds[1] ? RA[i]:lattice[icell].ra_bounds[1];

        lattice[icell].xbounds[0] = X[i] < lattice[icell].xbounds[0] ? X[i]:lattice[icell].xbounds[0];
        lattice[icell].ybounds[0] = Y[i] < lattice[icell].ybounds[0] ? Y[i]:lattice[icell].ybounds[0];
        lattice[icell].zbounds[0] = Z[i] < lattice[icell].zbounds[0] ? Z[i]:lattice[icell].zbounds[0];

        lattice[icell].xbounds[1] = X[i] > lattice[icell].xbounds[1] ? X[i]:lattice[icell].xbounds[1];
        lattice[icell].ybounds[1] = Y[i] > lattice[icell].ybounds[1] ? Y[i]:lattice[icell].ybounds[1];
        lattice[icell].zbounds[1] = Z[i] > lattice[icell].zbounds[1] ? Z[i]:lattice[icell].zbounds[1];

        lattice[icell].nelements++;
    }

    if(options->copy_particles) {
        //All the particle positions have already been copied -> do not need to re-allocate any more
        //You can free the extra memory reserved by the mallocs by looping over totncells
        //and doing a realloc(lattice[cellindex].x,sizeof(DOUBLE),lattice[cellindex].nelements,"lattice.x")
        free(nallocated);
    } else {
        // We have been told to work with the particle positions in-place i.e., not create a copy
        // of the particle positions within the lattice. Therefore, now we have to sort the
        // input particle positions to get them to be contiguous in their respective 3D cell
        if(sizeof(*(lattice->original_index)) != sizeof(*original_indices)) {
            fprintf(stderr, "Error: In %s> The array to track the indices of input particle positions "
                    "should be the same size as the indices themselves\n", __FUNCTION__);
            fprintf(stderr,"Perhaps check that these two variables are the same type\n");
            fprintf(stderr,"'original_index' within the 'cellarray', defined in 'cellarray.h' and \n");
            fprintf(stderr,"'original_indices' defined within function '%s' in file '%s'\n", __FUNCTION__, __FILE__);
            return NULL;
        }

        // First sort all particles into their respective cell-indices
        // also simultaneously swap the other associated array
#define MULTIPLE_ARRAY_EXCHANGER(type,a,i,j) {                          \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, X, i, j);            \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, Y, i, j);            \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, Z, i, j);            \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(int64_t, original_indices, i, j); \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(int64_t, all_cell_indices, i, j); \
            for(int w = 0; w < WEIGHTS->num_weights; w++) {             \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, ((DOUBLE *) WEIGHTS->weights[w]), i, j); \
            }                                                           \
        }


        //If the input-array is sorted exactly, then the quicksort will become a very-slow O(N^2)
        //Try to protect the user.
        int64_t num_sorted = 1;//an array containing exactly one element is always sorted
        for(int64_t ii=0;ii<NPART-1;ii++) {
            //increment by 1 if the next element is greater than or equal to current
            //decrement by 1 if the next element is smaller
            num_sorted += (all_cell_indices[ii+1] >= all_cell_indices[ii]) ? +1:-1;
        }

        //Check if there is any sorting to do
        //If the input array is already sorted, then all_cell_indices will also be sorted
        //which would result in num_sorted == NPART
        if(num_sorted < NPART) {
            //Since the particles might be coming from an already sorted array - quicksort might degenerate to
            //a O(N^2) process -- heap-sort might be safer.
            if(options->use_heap_sort || num_sorted > FRACTION_SORTED_REQD_TO_HEAP_SORT * NPART) {
                SGLIB_ARRAY_HEAP_SORT(int64_t, all_cell_indices, NPART, SGLIB_NUMERIC_COMPARATOR, MULTIPLE_ARRAY_EXCHANGER);
            } else {
                SGLIB_ARRAY_QUICK_SORT(int64_t, all_cell_indices, NPART, SGLIB_NUMERIC_COMPARATOR, MULTIPLE_ARRAY_EXCHANGER);
            }
        }

        //Now the particles are sorted contiguously according to their respective cellindex
        //We need to fix up the x/y/z pointers at the beginning of each cell to point to the right places
#undef MULTIPLE_ARRAY_EXCHANGER

        free(all_cell_indices);//Done with re-ordering the particles

        int64_t nelements_so_far = 0;
        for(int64_t icell=0;icell<*totncells;icell++) {
            cellarray_mocks *first=&(lattice[icell]);
            first->x = X + nelements_so_far;//take the base pointer address and add however many particles that have appeared summed across all previous cells
            first->y = Y + nelements_so_far;
            first->z = Z + nelements_so_far;
            first->original_index = original_indices + nelements_so_far;
            for(int w = 0; w < WEIGHTS->num_weights; w++) {
                first->weights.weights[w] = ((DOUBLE *) WEIGHTS->weights[w]) + nelements_so_far;
            }
            nelements_so_far += first->nelements;
        }
        XRETURN(nelements_so_far == NPART, NULL,
                "Error in %s> Expected to assign all particles = %"PRId64" into cells but only seem "
                "to have assigned %"PRId64". Perhaps, there are some edge cases with floating point accuracy\n",
                __FUNCTION__, NPART, nelements_so_far);

    }//end of options->copy_particles == 0

    if(sort_on_z) {
        for(int64_t icell=0;icell<ngrid_dec;icell++) {
            cellarray_mocks *first = &lattice[icell];
            if(first->nelements == 0) continue;

#define MULTIPLE_ARRAY_EXCHANGER(type,a,i,j) {                          \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->x, i, j); \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->y, i, j); \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->z, i, j); \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(int64_t, first->original_index,i, j); \
                for(int w = 0; w < first->weights.num_weights; w++){    \
                    SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->weights.weights[w],i,j); \
                }                                                       \
            }

            //If the input-array is sorted exactly, then the quicksort will become a very-slow O(N^2)
            //Try to protect the user.
            int64_t num_sorted = 1;//an array containing exactly one element is always sorted
            for(int64_t ii=0;ii<first->nelements-1;ii++) {
                //increment by 1 if the next element is greater than or equal to current
                //decrement by 1 if the next element is smaller
                num_sorted += (first->z[ii+1] >= first->z[ii]) ? +1:-1;
            }

            //Check if there is any sorting to do
            //If the input array is already sorted, then all_cell_indices will also be sorted
            //which would result in num_sorted == first->nelements
            if(num_sorted < first->nelements) {
                //Since the particles might be coming from an already sorted array - quicksort might degenerate to
                //a O(N^2) process -- heap-sort might be safer.
                if(options->use_heap_sort || num_sorted > FRACTION_SORTED_REQD_TO_HEAP_SORT * first->nelements) {
                    SGLIB_ARRAY_HEAP_SORT(DOUBLE, first->z, first->nelements, SGLIB_NUMERIC_COMPARATOR , MULTIPLE_ARRAY_EXCHANGER);
                } else {
                    SGLIB_ARRAY_QUICK_SORT(DOUBLE, first->z, first->nelements, SGLIB_NUMERIC_COMPARATOR , MULTIPLE_ARRAY_EXCHANGER);
                }
            }//if sorting is required
#undef MULTIPLE_ARRAY_EXCHANGER
        }//loop over cells
    }

    if(options->verbose) {
        struct timeval t1;
        gettimeofday(&t1,NULL);
        fprintf(stderr,"%s> expected_n = %"PRId64" ngrid (declination) = %d np=%"PRId64" Memory required = %0.2lf MB. Time taken = %7.3lf sec \n",
                __FUNCTION__, expected_n, ngrid_dec, NPART, totnbytes/1024.0/1024., ADD_DIFF_TIME(t0,t1));
    }

    return lattice;
}

struct cell_pair * generate_cell_pairs_mocks_theta_dec(cellarray_mocks *lattice1,
                                                                     cellarray_mocks *lattice2,
                                                                     const int64_t totncells,
                                                                     int64_t *ncell_pairs,
                                                                     const DOUBLE thetamax,
                                                                     const int dec_refine_factor,
                                                                     const int enable_min_sep_opt,
                                                                     const int autocorr)
{
    /* const DOUBLE max_chord_sep = 2.0*SIND(0.5*thetamax); */
    /*    C = 2.0 * SIN(thetamax/2)
       -> C^2 = 4.0 * SIN^2 (thetamax/2.0)
       -> C^2 = 2.0 * (2 * SIN^2(thetamax/2.0))
       -> C^2 = 2.0 * (1 - COS(thetamax))
     */
    const DOUBLE sqr_max_chord_sep = 2.0 * (1.0 - COSD(thetamax));

    const int64_t max_ngb_cells = 2*dec_refine_factor;  // -1 for self

    if( ! (autocorr == 0 || autocorr == 1) ) {
        fprintf(stderr,"Error: Strange value of autocorr = %d. Expected to receive either 1 (auto-correlations) or 0 (cross-correlations)\n", autocorr);
        return NULL;
    }
    const int64_t num_self_pairs = totncells;
    const int64_t num_nonself_pairs = totncells * max_ngb_cells / (autocorr + 1);

    const int64_t max_num_cell_pairs = num_self_pairs + num_nonself_pairs;

    int64_t num_cell_pairs = 0;
    struct cell_pair *all_cell_pairs = my_malloc(sizeof(*all_cell_pairs), max_num_cell_pairs);
    XRETURN(all_cell_pairs != NULL, NULL,
            "Error: Could not allocate memory for storing all the cell pairs. "
            "Reducing bin refine factors might help. Requested for %"PRId64" elements "
            "with each element of size %zu bytes\n", max_num_cell_pairs, sizeof(*all_cell_pairs));

    /* This ngb is a trivial function. Loop over +- idec from every cell. And that's a neighbour cell */
    for(int64_t icell=0;icell<totncells;icell++) {
        cellarray_mocks *first = &(lattice1[icell]);
        if(first->nelements == 0) continue;

        for(int idec=-dec_refine_factor;idec<=dec_refine_factor;idec++) {
            const int64_t icell2 = icell + idec;

            if(icell2 < 0 || icell2 >= totncells) continue;

            //Since we are creating a giant array with all possible cell-pairs, we need
            //to account for cases where an auto-correlation is occurring within the same cell.
            //To do so, means 'same_cell', 'min_dx/dy/dz', and 'closest_x1/y1/z1' must all be
            //set here. Also, if the second cell has no particles, then just skip it
            if((autocorr == 1 && icell2 > icell) || lattice2[icell2].nelements == 0) {
                continue;
            }

            cellarray_mocks *second = &(lattice2[icell2]);
            const DOUBLE closest_x1 = ZERO, closest_y1 = ZERO;
            DOUBLE closest_z1 = ZERO;
            const DOUBLE min_dx = ZERO, min_dy = ZERO;
            DOUBLE min_dz = ZERO;
            if(enable_min_sep_opt && idec != 0) {
                const DOUBLE first_z_low = first->zbounds[0], first_z_hi = first->zbounds[1];
                const DOUBLE second_z_low = second->zbounds[0], second_z_hi = second->zbounds[1];

                closest_z1 = idec < 0 ? first_z_low:first_z_hi;

                const DOUBLE first_z  = idec < 0 ? first_z_low:first_z_hi;
                const DOUBLE second_z = idec < 0 ? second_z_hi:second_z_low;
                min_dz = (first_z - second_z);
                const DOUBLE sqr_min_sep_cells = min_dx*min_dx + min_dy*min_dy + min_dz*min_dz;
                if(sqr_min_sep_cells >= sqr_max_chord_sep) {
                    continue;
                }
            }


            XRETURN(num_cell_pairs < max_num_cell_pairs, NULL,
                    "Error: Assigning this existing cell-pair would require accessing invalid memory.\n"
                    "Expected that the total number of cell pairs can be at most %"PRId64" but "
                    "currently have number of cell pairs = %"PRId64"\n", max_num_cell_pairs, num_cell_pairs);
            //If we have reached here, then this cell *MIGHT* have a pair. We
            //need to add a cell-pair to the array of all cell-pairs
            struct cell_pair *this_cell_pair = &all_cell_pairs[num_cell_pairs];
            this_cell_pair->cellindex1 = icell;
            this_cell_pair->cellindex2 = icell2;

            this_cell_pair->min_dx = min_dx;
            this_cell_pair->min_dy = min_dy;
            this_cell_pair->min_dz = min_dz;

            this_cell_pair->closest_x1 = closest_x1;
            this_cell_pair->closest_y1 = closest_y1;
            this_cell_pair->closest_z1 = closest_z1;

            this_cell_pair->same_cell = (autocorr == 1 && icell2 == icell) ? 1:0;

            num_cell_pairs++;
        }
    }

    *ncell_pairs = num_cell_pairs;
    return all_cell_pairs;
}


/* Create the lattice based on declination first, and then on right ascension */
cellarray_mocks * gridlink_mocks_theta_ra_dec(const int64_t NPART,
                                                            DOUBLE *RA, DOUBLE *DEC,
                                                            DOUBLE *X, DOUBLE *Y,
                                                            DOUBLE *Z, weight_struct *WEIGHTS,
                                                            const DOUBLE ra_min,const DOUBLE ra_max,
                                                            const DOUBLE dec_min, const DOUBLE dec_max,
                                                            const int max_ra_size,
                                                            const int max_dec_size,
                                                            const int ra_refine_factor,
                                                            const int dec_refine_factor,
                                                            const int sort_on_z,
                                                            const DOUBLE thetamax,
                                                            int64_t *ncells,
                                                            int *ngrid_declination,
                                                            int *max_nmesh_ra,//not really required - serves as additional checking mechanism
                                                            int **ngrid_phi,//ngrid in ra, updates on caller -> hence the pointer to pointer
                                                            const config_options *options)
{
    int64_t expected_n;
    size_t totnbytes=0;
    const DOUBLE dec_diff = dec_max - dec_min;
    const DOUBLE ra_diff = ra_max - ra_min;

    /* Input validation */
    XRETURN(thetamax > 0.0, NULL, "Minimum angular separation = %"REAL_FORMAT" must be positive\n", thetamax);
    XRETURN(NPART > 0, NULL, "Number of points =%"PRId64" must be >0\n", NPART);
    XRETURN(RA != NULL, NULL, "RA must be a valid array \n");
    XRETURN(DEC != NULL, NULL, "DEC must be a valid array \n");
    XRETURN(X != NULL, NULL, "X must be a valid array \n");
    XRETURN(Y != NULL, NULL, "Y must be a valid array \n");
    XRETURN(Z != NULL, NULL, "Z must be a valid array \n");
    XRETURN(ra_refine_factor >= 1, NULL, "RA refine factor must be at least 1\n");
    XRETURN(dec_refine_factor >= 1, NULL, "DEC refine factor must be at least 1\n");
    XRETURN(ncells != NULL, NULL, "Pointer to return the total number of cells must be a valid address\n");
    XRETURN(options != NULL, NULL, "Structure containing code options must be a valid address\n");
    XRETURN(ngrid_declination != NULL, NULL, "Address to return the number of DEC cells must be valid (ngrid_declination is NULL)\n");

    /* Validate the point distribution requested */
    XRETURN(dec_diff > 0.0, NULL,
            "All of the points can not be at the same declination. Declination difference = %"REAL_FORMAT" must be non-zero\n",
            dec_diff);
    XRETURN(ra_diff > 0.0, NULL,
            "All of the points can not be at the same RA. RA difference = %"REAL_FORMAT" must be non-zero\n",
            ra_diff);

    /* Protect against accidental edits */
    XRETURN(MEMORY_INCREASE_FAC >= 1.0, NULL, "Memory increase factor = %lf must be >=1 \n",MEMORY_INCREASE_FAC);

    const DOUBLE inv_dec_diff = 1.0/dec_diff;
    const DOUBLE inv_ra_diff = 1.0/ra_diff;

    struct timeval t0;
    if(options->verbose) {
        gettimeofday(&t0,NULL);
    }

    const DOUBLE this_ngrid_dec = (dec_diff/thetamax < 1) ? 1:dec_diff/thetamax;
    const int this_ngrid_dec_int = ((int) this_ngrid_dec) * dec_refine_factor;
    int ngrid_dec = this_ngrid_dec_int > max_dec_size ? max_dec_size:this_ngrid_dec_int;
    ngrid_dec = ngrid_dec < 1 ? 1:ngrid_dec;
    *ngrid_declination=ngrid_dec;
    DOUBLE dec_binsize=dec_diff/ngrid_dec;

    *ngrid_phi = my_malloc(sizeof(**ngrid_phi), ngrid_dec);
    if(*ngrid_phi == NULL) {
        return NULL;
    }
    totnbytes += sizeof(**ngrid_phi)*ngrid_dec;

    /* Use a local pointer. Only one level of dereferencing */
    int *ngrid_ra = *ngrid_phi;

    const DOUBLE costhetamax=COSD(thetamax);
    const DOUBLE sin_half_thetamax=SIND(0.5*thetamax);

    //Just make sure that the assign_ngb does not keep adding the same cell over and over
    int max_nmesh_phi = 1;//at least one cell (at the poles, there might be only one cell).
    const DOUBLE max_phi_cell = ra_diff;//must have at least one cell -> max ra binsize := ra_diff/1.0;
    for(int idec=0;idec<ngrid_dec;idec++) {
        DOUBLE this_min_dec;
        const DOUBLE dec_lower = dec_min + idec*dec_binsize;
        const DOUBLE dec_upper = dec_lower + dec_binsize;
        const DOUBLE cos_dec_upper = COSD(dec_upper);
        const DOUBLE cos_dec_lower = COSD(dec_lower);
        DOUBLE cos_min_dec;
        if(cos_dec_lower < cos_dec_upper) {
            this_min_dec = dec_lower;
            cos_min_dec = cos_dec_lower;
        } else {
            this_min_dec = dec_upper;
            cos_min_dec = cos_dec_upper;
        }


        //Use the Haversine approx. (https://en.wikipedia.org/wiki/Great-circle_distance)
        // \delta\sigma = 2 asin ( sqrt ( sin^2( 1/2 (phi1 - phi2)) + cos(phi1) * cos(phi2) * sin^2( 1/2 max-ra-diff)) )
        // Since phi1 == phi2 = this_min_dec; cos(this_min_dec) := cos_min_dec
        // => \thetamax = 2 asin (  sqrt( cos_min_dec^2 * sin^2( 1/2 max-ra-diff)) )
        // => \thetamax = 2 asin( cos_min_dec * sin(1/2 max-ra-diff)
        // => sin (1/2 \thetamax) = cos_min_dec * sin(1/2 max-ra-diff)
        // => sin(1/2 max-ra-diff) = sin (1/2 \thetamax)/cos_min_dec
        // => max-ra-diff = 2 * asin( sin(1/2 \thetamax)/cos_min_dec )

        DOUBLE phi_cell = max_phi_cell;
        if( (90.0 - ABS(this_min_dec) ) > 1.0) { //make sure min_dec is not close to the pole (within 1 degree)-> divide by zero happens the cosine term
            //Haversine formula but numerical precision issues might cause
            //sin_half_thetamax/cos_min_dec > 1.0 or sin_half_thetamax/cos_min_dec < -1.0
            //at which point arcsin() will barf. Therefore, a two-step process to limit
            //the possible values.
            //Sorry about the pythonic naming convention + the names themselves - MS 3rd April, 2017
            const DOUBLE _tmp = sin_half_thetamax/cos_min_dec;// sin(1/2 thetamax) > 0, cos(min_dec) >= 0 ( -90 <= declination <= 90, thetamax > 0)
            const DOUBLE _tmp1 = _tmp < ZERO ? ZERO:(_tmp > 1.0 ? 1.0:_tmp);
            phi_cell = 2.0 * ASIN( _tmp1 ) * INV_PI_OVER_180;

            //I don't think this condition can trigger any more, now that the
            //previous phi_cell calculation is split up into two steps - MS 3rd April, 2017
            if(phi_cell <= ZERO) {
                phi_cell = max_phi_cell;
            }

            //NAN's might trigger this condition ?
            if( ! (phi_cell > ZERO) ) {
                fprintf(stderr,"Error: Encountered invalid binsize for RA bins = %"REAL_FORMAT" for declination bin = %d\n", phi_cell, idec);
                fprintf(stderr,"min(declination in bin) = %"REAL_FORMAT" (min_dec)\n", this_min_dec);
                fprintf(stderr,"costhetamax = %"REAL_FORMAT" sin(thetamax/2.0) = %"REAL_FORMAT" cos(min_dec) = %"REAL_FORMAT" \n",
                        costhetamax, sin_half_thetamax, cos_min_dec);
            }
        }
        XRETURN(phi_cell > ZERO, NULL, "Please remove LINK_IN_RA from compile options "
                "(or from the code config options)\n");

        phi_cell = phi_cell > max_phi_cell ? max_phi_cell:phi_cell;
        const DOUBLE this_nmesh_ra = (ra_diff/phi_cell < 1) ? 1:ra_diff/phi_cell; //should be >= 1 since phi_cell is <= ra_diff
        const int this_nmesh_ra_int = ((int) this_nmesh_ra) * ra_refine_factor;
        int nmesh_ra = this_nmesh_ra_int > max_ra_size ? max_ra_size:this_nmesh_ra_int;

        //at least one ra bin
        if(nmesh_ra < 1) {
            nmesh_ra = 1;
        }

        if(nmesh_ra > max_nmesh_phi) max_nmesh_phi = nmesh_ra;
        ngrid_ra[idec] = nmesh_ra;
    }
    *max_nmesh_ra = max_nmesh_phi;
    expected_n=(int64_t)( (NPART/(DOUBLE) (ngrid_dec*max_nmesh_phi)) *MEMORY_INCREASE_FAC);
    expected_n = expected_n < 2 ? 2:expected_n;

    int64_t *ra_offset_for_dec = my_malloc(sizeof(*ra_offset_for_dec), ngrid_dec);
    int64_t offset = 0;
    for(int idec=0;idec<ngrid_dec;idec++) {
        ra_offset_for_dec[idec] = offset;
        offset += ngrid_ra[idec];
    }

    /*---Allocate-and-initialize-grid-arrays----------*/
    const int64_t totncells = offset;
    *ncells = totncells;
    cellarray_mocks *lattice = my_calloc(sizeof(*lattice), totncells);
    totnbytes += sizeof(*lattice) * totncells;
    int64_t *all_cell_indices = NULL;
    int64_t *original_indices =  NULL;
    int64_t *nallocated = NULL;//to keep track of how many particles have been allocated per cell (only when creating a copy of particle positions)
    if(options->copy_particles) {
        nallocated = (int64_t *) my_calloc(sizeof(*nallocated), totncells);
        totnbytes += sizeof(*nallocated) * totncells;
    } else {
        all_cell_indices = (int64_t *) my_malloc(sizeof(*all_cell_indices), NPART);
        original_indices = (int64_t *) my_malloc(sizeof(*original_indices), NPART);
        totnbytes += sizeof(*all_cell_indices)*NPART + sizeof(*original_indices)*NPART;
    }
    if(lattice == NULL ||
       (options->copy_particles == 0 && all_cell_indices == NULL) ||
       (options->copy_particles == 0 && original_indices == NULL) ||
       (options->copy_particles && nallocated == NULL)) {

        free(lattice);free(nallocated);free(all_cell_indices);free(original_indices);
        fprintf(stderr,"Error: In %s> Could not allocate memory for creating the lattice and associated arrays\n", __FUNCTION__);
        return NULL;
    }

    for(int64_t icell=0;icell<totncells;icell++) {
        lattice[icell].nelements=0;
        lattice[icell].x = NULL;
        lattice[icell].y = NULL;
        lattice[icell].z = NULL;
        lattice[icell].original_index = NULL;
        lattice[icell].owns_memory = 0;

        // Now do the same for the weights
        lattice[icell].weights.num_weights = (WEIGHTS == NULL) ? 0 : WEIGHTS->num_weights;

        lattice[icell].dec_bounds[0]=MAX_POSITIVE_FLOAT;
        lattice[icell].dec_bounds[1]=-MAX_POSITIVE_FLOAT;
        lattice[icell].ra_bounds[0]=MAX_POSITIVE_FLOAT;
        lattice[icell].ra_bounds[1]=-MAX_POSITIVE_FLOAT;
        lattice[icell].xbounds[0] = MAX_POSITIVE_FLOAT;
        lattice[icell].xbounds[1] = -MAX_POSITIVE_FLOAT;
        lattice[icell].ybounds[0] = MAX_POSITIVE_FLOAT;
        lattice[icell].ybounds[1] = -MAX_POSITIVE_FLOAT;
        lattice[icell].zbounds[0] = MAX_POSITIVE_FLOAT;
        lattice[icell].zbounds[1] = -MAX_POSITIVE_FLOAT;
    }

    for(int idec=0;idec<ngrid_dec;idec++) {
        const int nmesh_ra = ngrid_ra[idec];
        for(int ira=0;ira<nmesh_ra;ira++) {
            const int64_t ra_base = ra_offset_for_dec[idec];
            const int64_t icell = ra_base + ira;
            if(options->copy_particles) {
                lattice[icell].owns_memory = 1;
                const size_t memsize=sizeof(DOUBLE);

                lattice[icell].x = my_malloc(memsize,expected_n);
                lattice[icell].y = my_malloc(memsize,expected_n);
                lattice[icell].z = my_malloc(memsize,expected_n);
                lattice[icell].original_index = my_malloc(sizeof(*(lattice[icell].original_index)), expected_n);

                int w_alloc_status = EXIT_SUCCESS;
                for(int w = 0; w < lattice[icell].weights.num_weights; w++){
                    lattice[icell].weights.weights[w] = (DOUBLE *) my_malloc(memsize, expected_n);
                    if(lattice[icell].weights.weights[w] == NULL){
                        w_alloc_status = EXIT_FAILURE;
                    }
                }

                if(lattice[icell].x == NULL || lattice[icell].y == NULL ||
                   lattice[icell].z == NULL || lattice[icell].original_index == NULL ||
                   w_alloc_status == EXIT_FAILURE) {
                    /* Since all the x/y/z are initialized to NULL,
                       I can call the helper routine directly */
                    free_cellarray_mocks(lattice, totncells);
                    free(lattice);
                    free(nallocated);
                    return NULL;
                }
                nallocated[icell]=expected_n;
                totnbytes += 3 * memsize * expected_n + sizeof(int64_t) * expected_n;
            }
        }
    }

    /*---Loop-over-particles-and-build-grid-arrays----*/
    for(int64_t i=0;i<NPART;i++) {
        int idec = (int)(ngrid_dec*(DEC[i]-dec_min)*inv_dec_diff);
        if(idec >= ngrid_dec) idec--;

        XRETURN(idec >=0 && idec < ngrid_dec, NULL,
                "Declination index for particle position = %d must be within [0, %d)\n",
                idec, ngrid_dec);

        int ira  = (int)(ngrid_ra[idec]*(RA[i]-ra_min)*inv_ra_diff);
        if(ira >=ngrid_ra[idec]) ira--;
        XRETURN(ira >=0 && ira < ngrid_ra[idec],NULL,
                "RA index for particle position = %d must be within [0, %d) for declination bin = %d\n",
                idec, ngrid_ra[idec], idec);

        const int64_t ra_base = ra_offset_for_dec[idec];
        const int64_t icell = ra_base + ira;

        if(options->copy_particles == 0) {
            all_cell_indices[i] = icell;
            original_indices[i] = i;
        } else {

            if(lattice[icell].nelements == nallocated[icell]) {
                expected_n = nallocated[icell]*MEMORY_INCREASE_FAC;
                while(expected_n <= lattice[icell].nelements){
                    expected_n++;
                }

                const size_t memsize=sizeof(DOUBLE);
                DOUBLE *posx=NULL, *posy=NULL, *posz = NULL;
                int64_t *orig_index=NULL;
                int w_alloc_status;
                do {
                    posx = my_realloc(lattice[icell].x ,memsize,expected_n,"lattice.x");
                    posy = my_realloc(lattice[icell].y ,memsize,expected_n,"lattice.y");
                    posz = my_realloc(lattice[icell].z ,memsize,expected_n,"lattice.z");
                    orig_index = my_realloc(lattice[icell].original_index, sizeof(*(lattice[icell].original_index)), expected_n,"lattice.original_index");

                    lattice[icell].x = (posx == NULL) ? lattice[icell].x:posx;
                    lattice[icell].y = (posy == NULL) ? lattice[icell].y:posy;
                    lattice[icell].z = (posz == NULL) ? lattice[icell].z:posz;
                    lattice[icell].original_index = (orig_index == NULL) ? lattice[icell].original_index:orig_index;

                    w_alloc_status = EXIT_SUCCESS;
                    for(int w = 0; w < lattice[icell].weights.num_weights; w++){
                        DOUBLE *newweights = (DOUBLE *) my_realloc(lattice[icell].weights.weights[w], memsize, expected_n, "lattice.weights");
                        if(newweights == NULL){
                            w_alloc_status = EXIT_FAILURE;
                        } else {
                            lattice[icell].weights.weights[w] = newweights;
                        }
                    }

                    if(posx == NULL || posy == NULL || posz == NULL || orig_index == NULL ||  w_alloc_status == EXIT_FAILURE) {
                        expected_n--;
                    }
                } while(expected_n > nallocated[icell] && (posx == NULL ||
                                                           posy == NULL ||
                                                           posz == NULL ||
                                                           orig_index == NULL ||
                                                           w_alloc_status == EXIT_FAILURE));

                if(expected_n == nallocated[icell]) {
                    /*realloc failed. free memory and return */
                    fprintf(stderr,"In %s> Reallocation failed,  randomly subsampling the input particle set (currently at %"PRId64" particles) might help\n",
                            __FUNCTION__, NPART);
                    fprintf(stderr,"posx = %p posy = %p posz = %p\n", posx, posy, posz);
                    free_cellarray_mocks(lattice, totncells);
                    free(nallocated);
                    return NULL;
                }
                totnbytes += 3 * (expected_n - nallocated[icell]) * memsize + (expected_n - nallocated[icell]) * sizeof(int64_t);
                nallocated[icell] = expected_n;
            }
            XRETURN(lattice[icell].nelements < nallocated[icell],NULL,
                    ANSI_COLOR_RED"BUG: lattice[%"PRId64"].nelements = %"PRId64" must be less than allocated memory = %"PRId64  ANSI_COLOR_RESET "\n",
                    icell, lattice[icell].nelements, nallocated[icell]);

            const int64_t ipos=lattice[icell].nelements;
            lattice[icell].x[ipos]  = X[i];
            lattice[icell].y[ipos]  = Y[i];
            lattice[icell].z[ipos]  = Z[i];
            lattice[icell].original_index[ipos] = i;
            for(int w = 0; w < lattice[icell].weights.num_weights; w++){
                lattice[icell].weights.weights[w][ipos] = ((DOUBLE *)WEIGHTS->weights[w])[i];
            }
        }

        lattice[icell].dec_bounds[0] = DEC[i] < lattice[icell].dec_bounds[0] ? DEC[i]:lattice[icell].dec_bounds[0];
        lattice[icell].dec_bounds[1] = DEC[i] > lattice[icell].dec_bounds[1] ? DEC[i]:lattice[icell].dec_bounds[1];

        lattice[icell].ra_bounds[0] = RA[i] < lattice[icell].ra_bounds[0] ? RA[i]:lattice[icell].ra_bounds[0];
        lattice[icell].ra_bounds[1] = RA[i] > lattice[icell].ra_bounds[1] ? RA[i]:lattice[icell].ra_bounds[1];

        lattice[icell].xbounds[0] = X[i] < lattice[icell].xbounds[0] ? X[i]:lattice[icell].xbounds[0];
        lattice[icell].ybounds[0] = Y[i] < lattice[icell].ybounds[0] ? Y[i]:lattice[icell].ybounds[0];
        lattice[icell].zbounds[0] = Z[i] < lattice[icell].zbounds[0] ? Z[i]:lattice[icell].zbounds[0];

        lattice[icell].xbounds[1] = X[i] > lattice[icell].xbounds[1] ? X[i]:lattice[icell].xbounds[1];
        lattice[icell].ybounds[1] = Y[i] > lattice[icell].ybounds[1] ? Y[i]:lattice[icell].ybounds[1];
        lattice[icell].zbounds[1] = Z[i] > lattice[icell].zbounds[1] ? Z[i]:lattice[icell].zbounds[1];

        lattice[icell].nelements++;
    }
    free(ra_offset_for_dec);

    if(options->copy_particles) {
        //All the particle positions have already been copied -> do not need to re-allocate any more
        //You can free the extra memory reserved by the mallocs by looping over totncells
        //and doing a realloc(lattice[cellindex].x,sizeof(DOUBLE),lattice[cellindex].nelements,"lattice.x")
        free(nallocated);
    } else {
        // We have been told to work with the particle positions in-place i.e., not create a copy
        // of the particle positions within the lattice. Therefore, now we have to sort the
        // input particle positions to get them to be contiguous in their respective 3D cell
        if(sizeof(*(lattice->original_index)) != sizeof(*original_indices)) {
            fprintf(stderr, "Error: In %s> The array to track the indices of input particle positions "
                    "should be the same size as the indices themselves\n", __FUNCTION__);
            fprintf(stderr,"Perhaps check that these two variables are the same type\n");
            fprintf(stderr,"'original_index' within the 'cellarray', defined in 'cellarray.h' and \n");
            fprintf(stderr,"'original_indices' defined within function '%s' in file '%s'\n", __FUNCTION__, __FILE__);
            return NULL;
        }

        // First sort all particles into their respective cell-indices
        // also simultaneously swap the other associated array
#define MULTIPLE_ARRAY_EXCHANGER(type,a,i,j) {                          \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, X, i, j);            \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, Y, i, j);            \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, Z, i, j);            \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(int64_t, original_indices, i, j); \
            SGLIB_ARRAY_ELEMENTS_EXCHANGER(int64_t, all_cell_indices, i, j); \
            for(int w = 0; w < WEIGHTS->num_weights; w++) {             \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, ((DOUBLE *) WEIGHTS->weights[w]), i, j); \
            }                                                           \
        }

        //If the input-array is sorted exactly, then the quicksort will become a very-slow O(N^2)
        //Try to protect the user.
        int64_t num_sorted = 1;
        for(int64_t ii=0;ii<NPART-1;ii++) {
            //increment by 1 if the next element is greater than or equal to current
            //decrement by 1 if the next element is smaller
            num_sorted += (all_cell_indices[ii+1] >= all_cell_indices[ii]) ? +1:-1;
        }

        //Do we need to actually sort? (perhaps the input arrays are already sorted)
        if(num_sorted < NPART) {
            //Since the particles might be coming from an already sorted array - quicksort might degenerate to
            //a O(N^2) process -- heap-sort might be safer.
            if(options->use_heap_sort || num_sorted >  FRACTION_SORTED_REQD_TO_HEAP_SORT * NPART) {
                SGLIB_ARRAY_HEAP_SORT(int64_t, all_cell_indices, NPART, SGLIB_NUMERIC_COMPARATOR, MULTIPLE_ARRAY_EXCHANGER);
            } else {
                SGLIB_ARRAY_QUICK_SORT(int64_t, all_cell_indices, NPART, SGLIB_NUMERIC_COMPARATOR, MULTIPLE_ARRAY_EXCHANGER);
            }
        }

        //Now the particles are sorted contiguously according to their respective cellindex
        //We need to fix up the x/y/z pointers at the beginning of each cell to point to the right places
#undef MULTIPLE_ARRAY_EXCHANGER

        free(all_cell_indices);//Done with re-ordering the particles

        int64_t nelements_so_far = 0;
        for(int64_t icell=0;icell<totncells;icell++) {
            cellarray_mocks *first=&(lattice[icell]);
            first->x = X + nelements_so_far;//take the base pointer address and add however many particles that have appeared summed across all previous cells
            first->y = Y + nelements_so_far;
            first->z = Z + nelements_so_far;
            first->original_index = original_indices + nelements_so_far;
            for(int w = 0; w < WEIGHTS->num_weights; w++) {
                first->weights.weights[w] = ((DOUBLE *) WEIGHTS->weights[w]) + nelements_so_far;
            }
            nelements_so_far += first->nelements;
        }
        XRETURN(nelements_so_far == NPART, NULL,
                "Error in %s> Expected to assign all particles = %"PRId64" into cells but only seem "
                "to have assigned %"PRId64". Perhaps, there are some edge cases with floating point accuracy\n",
                __FUNCTION__, NPART, nelements_so_far);

    }//end of options->copy_particles == 0

    if(sort_on_z) {
        for(int64_t icell=0;icell<totncells;icell++) {
            cellarray_mocks *first = &lattice[icell];
            if(first->nelements == 0) continue;
#define MULTIPLE_ARRAY_EXCHANGER(type,a,i,j) {                          \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->x, i, j);        \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->y, i, j);        \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->z, i, j);        \
                SGLIB_ARRAY_ELEMENTS_EXCHANGER(int64_t, first->original_index, i, j); \
                for(int w = 0; w < first->weights.num_weights; w++){ \
                    SGLIB_ARRAY_ELEMENTS_EXCHANGER(DOUBLE, first->weights.weights[w],i,j);\
                }\
            }

            //If the input-array is sorted exactly, then the quicksort will become a very-slow O(N^2)
            //Try to protect the user.
            int64_t num_sorted = 1;//an array containing exactly one element is always sorted
            for(int64_t ii=0;ii<first->nelements-1;ii++) {
                //increment by 1 if the next element is greater than or equal to current
                //decrement by 1 if the next element is smaller
                num_sorted += (first->z[ii+1] >= first->z[ii]) ? +1:-1;
            }

            //Check if there is any sorting to do
            //If the input array is already sorted, then all_cell_indices will also be sorted
            //which would result in num_sorted == first->nelements
            if(num_sorted < first->nelements) {
                //Since the particles might be coming from an already sorted array - quicksort might degenerate to
                //a O(N^2) process -- heap-sort might be safer.

                //Sorting on z -> equivalent to sorting on declination
                //(since z := sin(dec) is a monotonic mapping in -90 <= dec <= 90, the domain for dec)
                if(options->use_heap_sort || num_sorted > FRACTION_SORTED_REQD_TO_HEAP_SORT * first->nelements) {
                    SGLIB_ARRAY_HEAP_SORT(DOUBLE, first->z, first->nelements, SGLIB_NUMERIC_COMPARATOR , MULTIPLE_ARRAY_EXCHANGER);
                } else {
                    SGLIB_ARRAY_QUICK_SORT(DOUBLE, first->z, first->nelements, SGLIB_NUMERIC_COMPARATOR , MULTIPLE_ARRAY_EXCHANGER);
                }
            }//if sorting is required
#undef MULTIPLE_ARRAY_EXCHANGER
        }//loop over cells
    }

    if(options->verbose) {
        struct timeval t1;
        gettimeofday(&t1,NULL);
        fprintf(stderr,"%s> Max. points in cell <= %"PRId64" ngrid (declination) = %d max ngrid (ra) = %d. Number of points = %"PRId64 " Memory required = %0.2g MB."
                " Time taken = %7.3lf sec.\n", __FUNCTION__,expected_n,ngrid_dec, max_nmesh_phi, NPART, totnbytes/1024./1024.,ADD_DIFF_TIME(t0,t1));
    }
    return lattice;
}

struct cell_pair * generate_cell_pairs_mocks_theta_ra_dec(cellarray_mocks *lattice1,
                                                                        cellarray_mocks *lattice2,
                                                                        const int64_t totncells,
                                                                        int64_t *ncell_pairs,
                                                                        const DOUBLE thetamax,
                                                                        const int ra_refine_factor, const int dec_refine_factor,
                                                                        const int ngrid_dec, const int max_ngrid_ra,
                                                                        const DOUBLE ra_min, const DOUBLE ra_max,
                                                                        const int *ngrid_ra,
                                                                        const int enable_min_sep_opt,
                                                                        const int autocorr)
{
    /* const DOUBLE max_chord_sep = 2.0*SIND(0.5*thetamax); */
    /*    C = 2.0 * SIN(thetamax/2)
       -> C^2 = 4.0 * SIN^2 (thetamax/2.0)
       -> C^2 = 2.0 * (2 * SIN^2(thetamax/2.0))
       -> C^2 = 2.0 * (1 - COS(thetamax))
     */
    const DOUBLE sqr_max_chord_sep = 2.0 * (1.0 - COSD(thetamax));

    const int max_dec_ngb = 2*dec_refine_factor + 1;
    const int max_ra_ngb  = 2*ra_refine_factor + 3;
    /* There might be additional cells on either end for cases where RA
       position changes between first and second cells */
    const int max_ngb_cells = max_dec_ngb * max_ra_ngb;

    const int64_t max_num_cell_pairs = totncells * max_ngb_cells;
    int64_t num_cell_pairs = 0;
    struct cell_pair *all_cell_pairs = my_malloc(sizeof(*all_cell_pairs), max_num_cell_pairs);
    XRETURN(all_cell_pairs != NULL, NULL,
            "Error: Could not allocate memory for storing all the cell pairs. "
            "Reducing bin refine factors might help. Requested for %"PRId64" elements "
            "with each element of size %zu bytes\n", max_num_cell_pairs, sizeof(*all_cell_pairs));

    const DOUBLE ra_diff = ra_max - ra_min;
    const DOUBLE inv_ra_diff = 1.0/ra_diff;
    (void) max_ngrid_ra;
    XRETURN( totncells <= max_ngrid_ra*ngrid_dec, NULL,
             "Total number of cells = %"PRId64" can be at most the product of max RA cells = %d and the number of DEC cells = %d\n",
             totncells, max_ngrid_ra, ngrid_dec);

    /* Since lattice1 is a single dimension array containing dec + ra bins, where the number
       of ra bins per dec bin varies, I need to have the offset (array index, **NOT* pointer offset)
       for the starting ra cell for each dec bin.
       Note: lattice2 *must* be of the exact same shape and there the same offsets apply to both lattices
    */
    int64_t *ra_offset_for_dec = my_malloc(sizeof(*ra_offset_for_dec), ngrid_dec);
    if(ra_offset_for_dec == NULL) {
        return NULL;
    }

    int64_t offset = 0;
    for(int idec=0;idec<ngrid_dec;idec++) {
        ra_offset_for_dec[idec] = offset;
        offset += ngrid_ra[idec];
    }
    XRETURN( totncells == offset, NULL,
             "Total number of cells = %"PRId64" must be exactly equal to the sum of number of RA cells over all declinations = %"PRId64"\n",
             totncells, offset);

    for(int idec=0;idec<ngrid_dec;idec++) {
        for(int ira=0;ira<ngrid_ra[idec];ira++) {
            const int64_t ra_base = ra_offset_for_dec[idec];
            const int64_t icell = ra_base + ira;
            cellarray_mocks *first = &(lattice1[icell]);
            if(first->nelements == 0) continue;

            int64_t num_ngb_this_cell = 0;
            for(int dec_refine=-dec_refine_factor;dec_refine<=dec_refine_factor;dec_refine++){
                const int this_dec = idec + dec_refine;
                if(this_dec < 0 || this_dec >= ngrid_dec) continue;

                /* Figure out what is the min and max RA cell that any particle in "first" cell could be in
                   if the "first" cell particle had a declination for this_dec (rather than idec) */
                //include one additional cell -> will get pruned if the separation is too large *and* never be duplicated
                //brute force approach would be to simply loop from 0 to ngrid_ra[this_dec] and prune away.
                const int min_ra_this_dec = (int) (ngrid_ra[this_dec] * (first->ra_bounds[0] - ra_min) * inv_ra_diff) - 1;
                const int max_ra_this_dec = (int) (ngrid_ra[this_dec] * (first->ra_bounds[1] - ra_min) * inv_ra_diff) + 1;

                for(int iira=min_ra_this_dec-ra_refine_factor;iira<=max_ra_this_dec+ra_refine_factor;iira++) {
                    int this_ra = iira + ngrid_ra[this_dec];
                    while(this_ra < 0) {
                        this_ra += ngrid_ra[this_dec];
                    }
                    this_ra = this_ra % ngrid_ra[this_dec];

                    XRETURN(this_ra >= 0 && this_ra < ngrid_ra[this_dec], NULL,
                            "Error: Cell index = %d for neighbour cell must be within [0, %d) "
                            "min/max ra index computed = [%d, %d]\n"
                            "Please reduce ra/dec refine factors\n",
                            this_ra, ngrid_ra[this_dec], min_ra_this_dec, max_ra_this_dec);

                    const int64_t this_ra_base = ra_offset_for_dec[this_dec];
                    const int64_t icell2 = this_ra_base + this_ra;
                    XRETURN(icell2 < totncells, NULL,
                            "index for ngb cell = %"PRId64" should be less total number of cells = %"PRId64"\n",
                            icell2, totncells);

                    cellarray_mocks *second = &(lattice2[icell2]);
                    //For cases where we are not double-counting (i.e., auto-corrs), the same-cell
                    //must always be evaluated. In all other cases, (i.e., where double-counting is occurring)
                    //is used, include that in the ngb_cells! The interface is a lot cleaner in the double-counting
                    //kernels in that case!
                    //The second condition essentially halves the number of cell-pairs in auto-corr calculations
                    if(second->nelements==0 || (autocorr==1 && icell2 > icell)) {
                        continue;
                    }

                    // No periodicity with mocks; wrap value is always zero.
                    const DOUBLE xwrap = ZERO;
                    const DOUBLE ywrap = ZERO;
                    const DOUBLE zwrap = ZERO;

                    CHECK_AND_CONTINUE_FOR_DUPLICATE_NGB_CELLS(icell, icell2, xwrap, ywrap, zwrap, num_cell_pairs, num_ngb_this_cell, all_cell_pairs);

                    XRETURN(num_cell_pairs < max_num_cell_pairs, NULL,
                            "Error: Assigning this existing cell-pair would require accessing invalid memory.\n"
                            "Expected that the total number of cell pairs can be at most %"PRId64" but "
                            "currently have number of cell pairs = %"PRId64"\n", max_num_cell_pairs, num_cell_pairs);

                    DOUBLE closest_x1 = ZERO, closest_y1 = ZERO, closest_z1 = ZERO;
                    DOUBLE min_dx = ZERO, min_dy = ZERO, min_dz = ZERO;

                    if(enable_min_sep_opt) {
                        DOUBLE closest_pos0;
                        min_dx = find_closest_pos(first->xbounds, second->xbounds, &closest_pos0);
                        closest_x1 = closest_pos0;

                        min_dy = find_closest_pos(first->ybounds, second->ybounds, &closest_pos0);
                        closest_y1 = closest_pos0;

                        min_dz = find_closest_pos(first->zbounds, second->zbounds, &closest_pos0);
                        closest_z1 = closest_pos0;

                        const DOUBLE sqr_min_sep_cells = min_dx*min_dx + min_dy*min_dy + min_dz*min_dz;
                        if(sqr_min_sep_cells >= sqr_max_chord_sep) {
                            continue;
                        }
                    }/* end of if condition for enable_min_sep_opt*/

                    //If we have reached here, then this cell *MIGHT* have a pair. We
                    //need to add a cell-pair to the array of all cell-pairs
                    struct cell_pair *this_cell_pair = &all_cell_pairs[num_cell_pairs];
                    this_cell_pair->cellindex1 = icell;
                    this_cell_pair->cellindex2 = icell2;

                    this_cell_pair->min_dx = min_dx;
                    this_cell_pair->min_dy = min_dy;
                    this_cell_pair->min_dz = min_dz;

                    this_cell_pair->closest_x1 = closest_x1;
                    this_cell_pair->closest_y1 = closest_y1;
                    this_cell_pair->closest_z1 = closest_z1;

                    this_cell_pair->xwrap = xwrap;
                    this_cell_pair->ywrap = ywrap;
                    this_cell_pair->zwrap = zwrap;

                    this_cell_pair->same_cell = (autocorr == 1 && icell2 == icell) ? 1:0;

                    num_cell_pairs++;
                    num_ngb_this_cell++;
                }//loop over possible range of RA values in this dec bin for the original RA bin in first
            }//loop over neighbouring DEC cells
        }//loop over all RA cells contained in this DEC bin
    }//loop over all DEC cells
    free(ra_offset_for_dec);

    *ncell_pairs = num_cell_pairs;
    return all_cell_pairs;
}
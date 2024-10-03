#include <stddef.h>

#include "utils.h"
#include "function_precision.h"

#include "cellarray.h"

cellarray *allocate_cellarray(
    const int64_t np,
    const int nx,
    const int ny,
    const int nz,
    const int with_weights
    ){

    cellarray *lattice = (cellarray *) my_malloc(sizeof(cellarray), 1);
    if (lattice == NULL) {
        return NULL;
    }

    int64_t totncells = (int64_t) nx * ny * nz;

    lattice->offsets = (int64_t *) my_malloc(sizeof(*lattice->offsets), totncells + 1);
    lattice->X = (DOUBLE *) my_malloc(sizeof(*lattice->X), np);
    lattice->Y = (DOUBLE *) my_malloc(sizeof(*lattice->Y), np);
    lattice->Z = (DOUBLE *) my_malloc(sizeof(*lattice->Z), np);
    if(with_weights) {
        lattice->W = (DOUBLE *) my_malloc(sizeof(*lattice->W), np);
        lattice->have_weights = 1;
    }
    else {
        lattice->W = NULL;
        lattice->have_weights = 0;
    }

    lattice->xbounds[0] = (DOUBLE *) my_malloc(sizeof(*lattice->xbounds[0]), totncells);
    lattice->xbounds[1] = (DOUBLE *) my_malloc(sizeof(*lattice->xbounds[1]), totncells);
    lattice->ybounds[0] = (DOUBLE *) my_malloc(sizeof(*lattice->ybounds[0]), totncells);
    lattice->ybounds[1] = (DOUBLE *) my_malloc(sizeof(*lattice->ybounds[1]), totncells);
    lattice->zbounds[0] = (DOUBLE *) my_malloc(sizeof(*lattice->zbounds[0]), totncells);
    lattice->zbounds[1] = (DOUBLE *) my_malloc(sizeof(*lattice->zbounds[1]), totncells);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int64_t icell=0;icell<totncells;icell++){
        lattice->xbounds[0][icell] = MAX_DOUBLE;
        lattice->xbounds[1][icell] = -MAX_DOUBLE;
        lattice->ybounds[0][icell] = MAX_DOUBLE;
        lattice->ybounds[1][icell] = -MAX_DOUBLE;
        lattice->zbounds[0][icell] = MAX_DOUBLE;
        lattice->zbounds[1][icell] = -MAX_DOUBLE;
    }

    lattice->nmesh_x = nx;
    lattice->nmesh_y = ny;
    lattice->nmesh_z = nz;
    lattice->tot_ncells = totncells;

    if(validate_cellarray(lattice) != EXIT_SUCCESS){
        free_cellarray(&lattice);
        return NULL;
    }

    return lattice;
}

int validate_cellarray(const cellarray *lattice){
    if(lattice->offsets == NULL || lattice->X == NULL || lattice->Y == NULL || lattice->Z == NULL){
        return EXIT_FAILURE;
    }
    if(lattice->xbounds[0] == NULL || lattice->xbounds[1] == NULL || lattice->ybounds[0] == NULL || lattice->ybounds[1] == NULL || lattice->zbounds[0] == NULL || lattice->zbounds[1] == NULL){
        return EXIT_FAILURE;
    }
    if(lattice->have_weights && lattice->W == NULL){
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void free_cellarray(cellarray **lattice_ptr){
    return;
    if(lattice_ptr == NULL) return;

    cellarray *lattice = *lattice_ptr;
    *lattice_ptr = NULL;
    if(lattice == NULL) return;

    free(lattice->X); lattice->X = NULL;
    free(lattice->Y); lattice->Y = NULL;
    free(lattice->Z); lattice->Z = NULL;
    if(lattice->have_weights){
        free(lattice->W); lattice->W = NULL;
    }
    free(lattice->offsets); lattice->offsets = NULL;
    free(lattice->xbounds[0]); lattice->xbounds[0] = NULL;
    free(lattice->xbounds[1]); lattice->xbounds[1] = NULL;
    free(lattice->ybounds[0]); lattice->ybounds[0] = NULL;
    free(lattice->ybounds[1]); lattice->ybounds[1] = NULL;
    free(lattice->zbounds[0]); lattice->zbounds[0] = NULL;
    free(lattice->zbounds[1]); lattice->zbounds[1] = NULL;

    free(lattice);
}

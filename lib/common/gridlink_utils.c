#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "function_precision.h"
#include "utils.h"

#include "gridlink_utils.h"

#ifndef CONVERT_3D_INDEX_TO_LINEAR
#define CONVERT_3D_INDEX_TO_LINEAR(ix, iy, iz, nx, ny, nz)           {ix*ny*nz + iy*nz + iz}
#endif


int get_gridsize(DOUBLE *xgridsize, int *nlattice, const DOUBLE xdiff, const DOUBLE xwrap, const DOUBLE rmax, const int refine_factor, const int max_ncells)
{
    int nmesh=(int)(refine_factor*xdiff/rmax);
    nmesh = nmesh < 1 ? 1:nmesh;
    
    if(xwrap > 0 && rmax >= xwrap/2){
        sprintf(ERRMSG,"%s> ERROR: rmax=%f must be less than half of periodic boxize=%f to avoid double-counting particles\n",
            __FILE__,rmax,xwrap);
        return EXIT_FAILURE;
    }

    if (nmesh>max_ncells)  nmesh=max_ncells;
    if (nmesh<2)  nmesh=2;  // to avoid forming duplicate cell pairs
    *xgridsize = xdiff/nmesh;
    *nlattice = nmesh;

    return EXIT_SUCCESS;
}


void get_max_min(const int64_t ND1, const DOUBLE * restrict X1, const DOUBLE * restrict Y1, const DOUBLE * restrict Z1,
                        DOUBLE *min_x, DOUBLE *min_y, DOUBLE *min_z, DOUBLE *max_x, DOUBLE *max_y, DOUBLE *max_z)
{
    DOUBLE xmin, ymin, zmin;
    DOUBLE xmax, ymax, zmax;

    xmin = ymin = zmin = MAX_DOUBLE;
    xmax = ymax = zmax = -MAX_DOUBLE;

#ifdef _OPENMP
    #pragma omp parallel for reduction(min: xmin, ymin, zmin) reduction(max: xmax, ymax, zmax)
#endif
    for(int64_t i=0;i<ND1;i++) {
        if(X1[i] < xmin) xmin=X1[i];
        if(Y1[i] < ymin) ymin=Y1[i];
        if(Z1[i] < zmin) zmin=Z1[i];


        if(X1[i] > xmax) xmax=X1[i];
        if(Y1[i] > ymax) ymax=Y1[i];
        if(Z1[i] > zmax) zmax=Z1[i];
    }
    
    *min_x=xmin;*min_y=ymin;*min_z=zmin;
    *max_x=xmax;*max_y=ymax;*max_z=zmax;
}



void get_max_min_ra_dec(const int64_t ND1, const DOUBLE *RA, const DOUBLE *DEC,
                               DOUBLE *ra_min, DOUBLE *dec_min, DOUBLE *ra_max, DOUBLE *dec_max)
{
    DOUBLE xmin = *ra_min, ymin = *dec_min;
    DOUBLE xmax = *ra_max, ymax = *dec_max;

    for(int64_t i=0;i<ND1;i++) {
        if(RA[i]  < xmin) xmin=RA[i];
        if(DEC[i] < ymin) ymin=DEC[i];

        if(RA[i] > xmax) xmax=RA[i];
        if(DEC[i] > ymax) ymax=DEC[i];
    }
    *ra_min=xmin;*dec_min=ymin;
    *ra_max=xmax;*dec_max=ymax;
}

DOUBLE find_closest_pos(const DOUBLE first_xbounds[2], const DOUBLE second_xbounds[2], DOUBLE *closest_pos0)
{
    *closest_pos0 = ZERO;
    /* if the limits are overlapping then the minimum possible separation is 0 */
    if(first_xbounds[0] <= second_xbounds[1]
       && second_xbounds[0] <= first_xbounds[1]) {
        return ZERO;
    }

    DOUBLE min_dx = FABS(first_xbounds[0] - second_xbounds[0]);
    *closest_pos0 = first_xbounds[0];
    for(int i=0;i<2;i++) {
        for(int j=0;j<2;j++) {
            const DOUBLE dx = FABS(first_xbounds[i] - second_xbounds[j]);
            if(dx < min_dx) {
                *closest_pos0 = first_xbounds[i];
                min_dx = dx;
            }
        }
    }

    return min_dx;
}


void find_min_and_max_sqr_sep_between_cell_pairs(const DOUBLE first_xbounds[2], const DOUBLE first_ybounds[2], const DOUBLE first_zbounds[2],
                                                        const DOUBLE second_xbounds[2], const DOUBLE second_ybounds[2], const DOUBLE second_zbounds[2],
                                                        DOUBLE *sqr_sep_min, DOUBLE *sqr_sep_max)
{
    DOUBLE min_sqr_sep = ZERO;

    if (first_xbounds[0] > second_xbounds[1]) min_sqr_sep += (first_xbounds[0] - second_xbounds[1])*(first_xbounds[0] - second_xbounds[1]);
    if (second_xbounds[0] > first_xbounds[1]) min_sqr_sep += (second_xbounds[0] - first_xbounds[1])*(second_xbounds[0] - first_xbounds[1]);

    if (first_ybounds[0] > second_ybounds[1]) min_sqr_sep += (first_ybounds[0] - second_ybounds[1])*(first_ybounds[0] - second_ybounds[1]);
    if (second_ybounds[0] > first_ybounds[1]) min_sqr_sep += (second_ybounds[0] - first_ybounds[1])*(second_ybounds[0] - first_ybounds[1]);

    if (first_zbounds[0] > second_zbounds[1]) min_sqr_sep += (first_zbounds[0] - second_zbounds[1])*(first_zbounds[0] - second_zbounds[1]);
    if (second_zbounds[0] > first_zbounds[1]) min_sqr_sep += (second_zbounds[0] - first_zbounds[1])*(second_zbounds[0] - first_zbounds[1]);

    const DOUBLE xmin = FMIN(first_xbounds[0], second_xbounds[0]);
    const DOUBLE xmax = FMAX(first_xbounds[1], second_xbounds[1]);

    const DOUBLE ymin = FMIN(first_ybounds[0], second_ybounds[0]);
    const DOUBLE ymax = FMAX(first_ybounds[1], second_ybounds[1]);

    const DOUBLE zmin = FMIN(first_zbounds[0], second_zbounds[0]);
    const DOUBLE zmax = FMAX(first_zbounds[1], second_zbounds[1]);

    const DOUBLE max_sqr_sep = (xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin) + (zmax - zmin) * (zmax - zmin);

    *sqr_sep_min = min_sqr_sep;
    *sqr_sep_max = max_sqr_sep;

}

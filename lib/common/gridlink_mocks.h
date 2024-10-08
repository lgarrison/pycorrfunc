#pragma once

#include <stdint.h>

#include "cell_pair.h"
#include "cellarray_mocks.h"
#include "defs.h"

/* Functions related to DDrppi_mocks/DDsmu_mocks */
extern cellarray_mocks *gridlink_mocks(const int64_t np,
                                       DOUBLE *x,
                                       DOUBLE *y,
                                       DOUBLE *z,
                                       const weight_struct *weights,
                                       const DOUBLE xmin,
                                       const DOUBLE xmax,
                                       const DOUBLE ymin,
                                       const DOUBLE ymax,
                                       const DOUBLE zmin,
                                       const DOUBLE zmax,
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
   __attribute__((warn_unused_result));

extern struct cell_pair *generate_cell_pairs_mocks(cellarray_mocks *lattice1,
                                                   cellarray_mocks *lattice2,
                                                   const int64_t totncells,
                                                   int64_t *ncell_pairs,
                                                   const int xbin_refine_factor,
                                                   const int ybin_refine_factor,
                                                   const int zbin_refine_factor,
                                                   const int nmesh_x,
                                                   const int nmesh_y,
                                                   const int nmesh_z,
                                                   const DOUBLE max_3D_sep,
                                                   const int enable_min_sep_opt,
                                                   const int autocorr)
   __attribute__((warn_unused_result));

extern void free_cellarray_mocks(cellarray_mocks *lattice, const int64_t totncells);
/* End of functions related to DDrppi_mocks */

/* Function declarations for DDtheta_mocks */
extern cellarray_mocks *gridlink_mocks_theta_dec(const int64_t NPART,
                                                 DOUBLE *RA,
                                                 DOUBLE *DEC,
                                                 DOUBLE *X,
                                                 DOUBLE *Y,
                                                 DOUBLE *Z,
                                                 weight_struct *WEIGHTS,
                                                 const DOUBLE dec_min,
                                                 const DOUBLE dec_max,
                                                 const DOUBLE max_dec_size,
                                                 const int dec_refine_factor,
                                                 const int sort_on_z,
                                                 const DOUBLE thetamax,
                                                 int64_t *totncells,
                                                 const config_options *options)
   __attribute__((warn_unused_result));

extern struct cell_pair *
generate_cell_pairs_mocks_theta_dec(cellarray_mocks *lattice1,
                                    cellarray_mocks *lattice2,
                                    const int64_t totncells,
                                    int64_t *ncell_pairs,
                                    const DOUBLE thetamax,
                                    const int dec_refine_factor,
                                    const int enable_min_sep_opt,
                                    const int autocorr)
   __attribute__((warn_unused_result));

extern cellarray_mocks *gridlink_mocks_theta_ra_dec(
   const int64_t NPART,
   DOUBLE *RA,
   DOUBLE *DEC,
   DOUBLE *X,
   DOUBLE *Y,
   DOUBLE *Z,
   weight_struct *WEIGHTS,
   const DOUBLE ra_min,
   const DOUBLE ra_max,
   const DOUBLE dec_min,
   const DOUBLE dec_max,
   const int max_ra_size,
   const int max_dec_size,
   const int ra_refine_factor,
   const int dec_refine_factor,
   const int sort_on_z,
   const DOUBLE thetamax,
   int64_t *ncells,
   int *ngrid_declination,
   int *max_nmesh_ra,  // not really required - serves as additional checking mechanism
   int **ngrid_phi,    // ngrid in ra, updates on caller -> hence the pointer to pointer
   const config_options *options) __attribute__((warn_unused_result));

extern struct cell_pair *
generate_cell_pairs_mocks_theta_ra_dec(cellarray_mocks *lattice1,
                                       cellarray_mocks *lattice2,
                                       const int64_t totncells,
                                       int64_t *ncell_pairs,
                                       const DOUBLE thetamax,
                                       const int ra_refine_factor,
                                       const int dec_refine_factor,
                                       const int nmesh_dec,
                                       const int max_nmesh_ra,
                                       const DOUBLE ra_min,
                                       const DOUBLE ra_max,
                                       const int *nmesh_grid_ra,
                                       const int enable_min_sep_opt,
                                       const int autocorr)
   __attribute__((warn_unused_result));

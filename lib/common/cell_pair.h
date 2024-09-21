#pragma once

#include <stdint.h>

struct cell_pair{
    int64_t cellindex1;
    int64_t cellindex2;

    DOUBLE xwrap;
    DOUBLE ywrap;
    DOUBLE zwrap;

    DOUBLE min_dx;
    DOUBLE min_dy;
    DOUBLE min_dz;

    DOUBLE closest_x1;
    DOUBLE closest_y1;
    DOUBLE closest_z1;

    int8_t same_cell;
};

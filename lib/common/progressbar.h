#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void init_my_progressbar(const int64_t N);
void my_progressbar(const int64_t curr_index);
void finish_myprogressbar();

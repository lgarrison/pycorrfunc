#include "avx512_calls.h"

#ifdef PYCORRFUNC_USE_DOUBLE
const int64_t bits_set_in_avx512_mask[] = { B8(0) };
const uint8_t masks_per_misalignment_value[] = {0b11111111, 
                                                0b00000001,
                                                0b00000011,
                                                0b00000111,
                                                0b00001111,
                                                0b00011111,
                                                0b00111111,
                                                0b01111111};
#else
const int64_t bits_set_in_avx512_mask[] = { B16(0) };
const uint16_t masks_per_misalignment_value[] = {0b1111111111111111,
                                                    0b0000000000000001,
                                                    0b0000000000000011,
                                                    0b0000000000000111,
                                                    0b0000000000001111,
                                                    0b0000000000011111,
                                                    0b0000000000111111,
                                                    0b0000000001111111,
                                                    0b0000000011111111,
                                                    0b0000000111111111,
                                                    0b0000001111111111,
                                                    0b0000011111111111,
                                                    0b0000111111111111,
                                                    0b0001111111111111,
                                                    0b0011111111111111,
                                                    0b0111111111111111};
#endif

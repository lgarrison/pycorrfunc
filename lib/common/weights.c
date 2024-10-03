#include "weights.h"

/* Gives the number of weight arrays required by the given weighting method
 */
// Currently unused
inline int get_num_weights_by_method(const weight_method_t method){
    switch(method){
        case PAIR_PRODUCT:
            return 1;
        default:
        case NONE:
            return 0;
    }
}

/* Maps a name to weighting method
   `method` will be set on return.
 */
inline int get_weight_method_by_name(const char *name, weight_method_t *method){
    if(name == NULL || strcmp(name, "") == 0){
        *method = NONE;
        return EXIT_SUCCESS;
    }
    // These should not be strncmp because we want the implicit length comparison of strcmp.
    // It is still safe because one of the args is a string literal.
    if(strcmp(name, "pair_product") == 0 || strcmp(name, "p") == 0){
        *method = PAIR_PRODUCT;
        return EXIT_SUCCESS;
    }

    return EXIT_FAILURE;
}


/* Gives a pointer to the weight function for the given weighting method
 * and instruction set.
 */
weight_func_t get_weight_func_by_method(const weight_method_t method){
    switch(method){
        case PAIR_PRODUCT:
            return &pair_product;
        default:
        case NONE:
            return NULL;
    }
}

#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>

#ifndef UPTK_CHECK
#define UPTK_CHECK(error) \
    do { \
        UPTKresult result = (error); \
        if (result != UPTK_SUCCESS) { \
            fprintf(stderr, "UPTK error at %s:%d - Code: %d\n", __FILE__, __LINE__, result); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif

int main(){
    unsigned int num_options = 2;
    unsigned int threads_per_block = 256;
    long wall_time = 1000;
    UPTKjit_option valid_options_ptr[] = {UPTK_JIT_THREADS_PER_BLOCK, UPTK_JIT_WALL_TIME};
    void *valid_options_vals_pptr[] = {(void *)(uintptr_t)threads_per_block, &wall_time};
    UPTKlinkState UPTK_link_state_ptr = nullptr;
    UPTKjitInputType input_type = UPTK_JIT_INPUT_PTX;

    const char name[] = "init";
    size_t num_elements = 5;
    size_t element_size = sizeof(int);
    size_t image_size = num_elements * element_size;
    void *image = malloc(image_size);
    int data[] = {1, 2, 3, 4, 5};
    memcpy(image, data, image_size);

    input_type = UPTK_JIT_INPUT_PTX;
    const char nul_name[] = "";
    UPTK_CHECK(UPTKLinkCreate(num_options, valid_options_ptr, valid_options_vals_pptr, &UPTK_link_state_ptr));
    UPTK_CHECK(UPTKLinkAddData(UPTK_link_state_ptr, input_type, image, image_size, nul_name, num_options, valid_options_ptr, valid_options_vals_pptr));
    UPTK_CHECK(UPTKLinkDestroy(UPTK_link_state_ptr));
    free(image);
    return 0;
}

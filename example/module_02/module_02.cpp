#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>
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

    void *bin_out = nullptr;
    size_t size_out = 0;
    UPTK_CHECK(UPTKLinkCreate(num_options, valid_options_ptr, valid_options_vals_pptr, &UPTK_link_state_ptr));
    UPTK_CHECK(UPTKLinkComplete(UPTK_link_state_ptr, &bin_out, &size_out));
    UPTK_CHECK(UPTKLinkDestroy(UPTK_link_state_ptr));

    return 0;
}

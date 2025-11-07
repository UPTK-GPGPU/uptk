#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>

extern "C" {

// 向量加法内核
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 包含全局变量的内核
__device__ int global_counter = 100;

__global__ void incrementAndGet(int* result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int old_val = atomicAdd(&global_counter, 1);
        result[idx] = old_val + 1;  // 返回递增后的值
    }
}

}

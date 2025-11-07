#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>
#include <stdio.h>

// 核函数，计算数组元素的平方
__global__ void squareKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}

int main() {
    int N = 100;
    const int blockSize = 32;
    const int gridSize = (N + blockSize - 1) / blockSize;
    
    float *h_data, *d_data;
    size_t size = N * sizeof(float);
    
    // 分配主机和设备内存
    h_data = (float*)malloc(size);
    UPTKMalloc(&d_data, size);
    
    // 初始化主机数据
    for (int i = 0; i < N; i++) {
        h_data[i] = i + 1.0f;  // 1.0, 2.0, 3.0, ...
    }
    
    // 复制数据到设备
    UPTKMemcpy(d_data, h_data, size, UPTKMemcpyHostToDevice);
    
    // 1. 使用 UPTKFuncGetAttributes 获取函数属性
    UPTKFuncAttributes attr;
    UPTKError_t result = UPTKFuncGetAttributes(&attr, (const void*)squareKernel);
    if (result != UPTKSuccess) {
        printf("UPTKFuncGetAttributes failed: %s\n", UPTKGetErrorString(result));
        return -1;
    }
    
    printf("Function attributes:\n");
    printf("  Max threads per block: %d\n", attr.maxThreadsPerBlock);
    printf("  Num registers: %d\n", attr.numRegs);
    printf("  PTX version: %d\n", attr.ptxVersion);
    printf("  Binary version: %d\n", attr.binaryVersion);
    
    // 2. 使用 UPTKFuncSetAttribute 设置函数属性
    printf("Setting function attributes with UPTKFuncSetAttribute...\n");
    
    // 设置最大动态共享内存大小
    result = UPTKFuncSetAttribute((const void*)squareKernel, UPTKFuncAttributeMaxDynamicSharedMemorySize, 1024);
    if (result != UPTKSuccess) {
        printf("UPTKFuncSetAttribute (MaxDynamicSharedMemorySize) failed: %s\n", UPTKGetErrorString(result));
    }
    
    // 设置偏好共享内存缓存
    result = UPTKFuncSetAttribute((const void*)squareKernel, UPTKFuncAttributePreferredSharedMemoryCarveout, UPTKSharedmemCarveoutMaxShared);
    if (result != UPTKSuccess) {
        printf("UPTKFuncSetAttribute (PreferredSharedMemoryCarveout) failed: %s\n", UPTKGetErrorString(result));
    }
    
    // 3. 使用 UPTKLaunchKernel 启动核函数
    printf("Launching kernel with UPTKLaunchKernel...\n");
    void* args[] = {&d_data, &N};
    result = UPTKLaunchKernel((const void*)squareKernel, gridSize, blockSize, args, 0, 0);
    if (result != UPTKSuccess) {
        printf("UPTKLaunchKernel failed: %s\n", UPTKGetErrorString(result));
        return -1;
    }
    UPTKDeviceSynchronize();
    
    // 复制结果回主机
    UPTKMemcpy(h_data, d_data, size, UPTKMemcpyDeviceToHost);
    
    // 验证结果
    printf("Verifying results (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("  data[%d] = %.1f (expected %.1f)\n", i, h_data[i], (i+1.0f)*(i+1.0f));
    }
    
    // 清理
    free(h_data);
    UPTKFree(d_data);
    
    printf("Program completed successfully!\n");
    return 0;
}

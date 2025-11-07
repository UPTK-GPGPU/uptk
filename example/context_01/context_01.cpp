#include <stdio.h>
#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>

// 向量加法内核函数
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    UPTKresult result;
    UPTKdevice device;
    UPTKcontext context;
    size_t limit_value;
    const int N = 256; 
    size_t size = N * sizeof(float);
    
    // 初始化UPTK驱动API
    result = UPTKInit(0);
    if (result != UPTKSuccess) {
        printf("UPTKInit failed: %d\n", result);
        return -1;
    }
    
    // 1. 创建上下文 - UPTKCtxCreate
    result = UPTKCtxCreate(&context, 0, device);
    if (result != UPTKSuccess) {
        printf("UPTKCtxCreate failed: %d\n", result);
        return -1;
    }
    printf("Context created successfully\n");
    
    // 2. 设置当前上下文 - UPTKCtxSetCurrent
    result = UPTKCtxSetCurrent(context);
    if (result != UPTKSuccess) {
        printf("UPTKCtxSetCurrent failed: %d\n", result);
        return -1;
    }
    printf("Context set as UPTKCurrent\n");
    
    // 3. 获取上下文限制 - UPTKCtxGetLimit
    result = UPTKCtxGetLimit(&limit_value, UPTK_LIMIT_STACK_SIZE);
    if (result != UPTKSuccess) {
        printf("UPTKCtxGetLimit failed: %d\n", result);
        return -1;
    }
    printf("UPTKCurrent stack size limit: %zu bytes\n", limit_value);
    
    // 4. 设置上下文限制 - UPTKCtxSetLimit
    size_t new_limit = 4096; // 4KB
    result = UPTKCtxSetLimit(UPTK_LIMIT_STACK_SIZE, new_limit);
    if (result != UPTKSuccess) {
        printf("UPTKCtxSetLimit failed: %d\n", result);
        return -1;
    }
    printf("Stack size limit set to: %zu bytes\n", new_limit);
    
    // 验证设置是否生效
    result = UPTKCtxGetLimit(&limit_value, UPTK_LIMIT_STACK_SIZE);
    if (result != UPTKSuccess) {
        printf("UPTKCtxGetLimit failed: %d\n", result);
        return -1;
    }
    printf("Verified stack size limit: %zu bytes\n", limit_value);
    
    float *h_a, *h_b, *h_c;  // 主机内存
    float *d_a, *d_b, *d_c;  // 设备内存
    
    // 分配主机内存
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    // 初始化主机数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 分配设备内存
    UPTKMalloc(&d_a, size);
    UPTKMalloc(&d_b, size);
    UPTKMalloc(&d_c, size);
    
    // 拷贝数据到设备
    UPTKMemcpy(d_a, h_a, size, UPTKMemcpyHostToDevice);
    UPTKMemcpy(d_b, h_b, size, UPTKMemcpyHostToDevice);
    
    // 启动内核
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    
    // 5. 同步上下文 - UPTKCtxSynchronize
    result = UPTKCtxSynchronize();
    if (result != UPTKSuccess) {
        printf("UPTKCtxSynchronize failed: %d\n", result);
        return -1;
    }
    printf("Context synchronized successfully\n");
    
    // 拷贝结果回主机
    UPTKMemcpy(h_c, d_c, size, UPTKMemcpyDeviceToHost);
    
    // 验证结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Kernel execution successful! Vector addition verified.\n");
    } else {
        printf("Kernel execution failed! Results incorrect.\n");
    }
    
    // 打印前5个结果作为示例
    printf("First 5 results: ");
    for (int i = 0; i < 5 && i < N; i++) {
        printf("%.1f + %.1f = %.1f, ", h_a[i], h_b[i], h_c[i]);
    }
    printf("\n");
    
    // 清理资源
    free(h_a);
    free(h_b);
    free(h_c);
    UPTKFree(d_a);
    UPTKFree(d_b);
    UPTKFree(d_c);
    
    // 6. 销毁上下文 - UPTKCtxDestroy
    result = UPTKCtxDestroy(context);
    if (result != UPTKSuccess) {
        printf("UPTKCtxDestroy failed: %d\n", result);
        return -1;
    }
    printf("Context destroyed successfully\n");
    
    printf("All UPTK context operations completed successfully!\n");
    return 0;
}

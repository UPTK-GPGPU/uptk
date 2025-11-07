#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define BLOCK_SIZE 256

// 向量加法内核
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    
    // 1. 创建默认事件
    UPTKEvent_t event_default;
    UPTKEventCreate(&event_default);
    printf("使用UPTKEventCreate创建默认事件\n");
    
    // 2. 创建带标志的事件
    UPTKEvent_t event_flags;
    UPTKEventCreateWithFlags(&event_flags, UPTKEventDisableTiming);
    printf("使用UPTKEventCreateWithFlags创建禁用计时的事件\n");
    
    // 3. 创建用于计时的事件
    UPTKEvent_t start, stop;
    UPTKEventCreate(&start);
    UPTKEventCreate(&stop);
    printf("创建开始和停止计时事件\n");
    
    // 分配内存
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);
    
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    UPTKMalloc(&d_a, size);
    UPTKMalloc(&d_b, size);
    UPTKMalloc(&d_c, size);
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // 记录开始时间
    UPTKEventRecord(start, 0);
    printf("使用UPTKEventRecord记录开始事件\n");
    
    // 数据传输到设备
    UPTKMemcpy(d_a, h_a, size, UPTKMemcpyHostToDevice);
    UPTKMemcpy(d_b, h_b, size, UPTKMemcpyHostToDevice);
    
    // 执行内核
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAdd<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    
    // 记录默认事件
    UPTKEventRecord(event_default, 0);
    
    // 记录禁用计时的事件
    UPTKEventRecord(event_flags, 0);
    
    // 数据传输回主机
    UPTKMemcpy(h_c, d_c, size, UPTKMemcpyDeviceToHost);
    
    // 记录停止时间
    UPTKEventRecord(stop, 0);
    printf("使用UPTKEventRecord记录停止事件\n");
    
    // 4. 使用UPTKEventSynchronize等待事件完成
    UPTKEventSynchronize(stop);
    printf("使用UPTKEventSynchronize等待所有操作完成\n");
    
    // 5. 使用UPTKEventQuery检查事件状态
    UPTKError_t query_result = UPTKEventQuery(event_default);
    if (query_result == UPTKSuccess) {
        printf("使用UPTKEventQuery检查事件状态: 事件已完成\n");
    } else {
        printf("事件尚未完成\n");
    }
    
    // 6. 使用UPTKEventElapsedTime计算执行时间
    float milliseconds = 0;
    UPTKEventElapsedTime(&milliseconds, start, stop);
    printf("使用UPTKEventElapsedTime计算执行时间: %.3f ms\n", milliseconds);
    
    // 验证结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            success = false;
            break;
        }
    }
    printf("计算结果: %s\n", success ? "正确" : "错误");
    
    // 7. 使用UPTKEventDestroy销毁所有事件
    UPTKEventDestroy(event_default);
    UPTKEventDestroy(event_flags);
    UPTKEventDestroy(start);
    UPTKEventDestroy(stop);
    printf("使用UPTKEventDestroy销毁所有事件\n");
    
    // 清理内存
    free(h_a);
    free(h_b);
    free(h_c);
    UPTKFree(d_a);
    UPTKFree(d_b);
    UPTKFree(d_c);
    
    printf("程序结束\n");
    return 0;
}

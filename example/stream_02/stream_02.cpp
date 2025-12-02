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
    
    // 创建流
    UPTKStream_t stream;
    UPTKStreamCreate(&stream);
    
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
    
    // 1. 使用UPTKStreamIsCapturing检查初始状态
    UPTKStreamCaptureStatus status;
    UPTKStreamIsCapturing(stream, &status);
    printf("开始前状态: %s\n", status == 0 ? "未捕获" : "捕获中");
    
    // 2. 使用UPTKStreamBeginCapture开始捕获
    UPTKGraph_t graph;
    UPTKStreamBeginCapture(stream, UPTKStreamCaptureModeGlobal);
    printf("开始流捕获\n");
    
    // 检查捕获中状态
    UPTKStreamIsCapturing(stream, &status);
    printf("捕获中状态: %s\n", status == 0 ? "未捕获" : "捕获中");
    
    // 在捕获期间执行简单操作
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    UPTKMemcpyAsync(d_a, h_a, size, UPTKMemcpyHostToDevice, stream);
    UPTKMemcpyAsync(d_b, h_b, size, UPTKMemcpyHostToDevice, stream);
    vectorAdd<<<numBlocks, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_c, N);
    UPTKMemcpyAsync(h_c, d_c, size, UPTKMemcpyDeviceToHost, stream);
    
    // 3. 使用UPTKStreamEndCapture结束捕获
    UPTKStreamEndCapture(stream, &graph);
    printf("结束流捕获\n");
    
    // 检查结束后的状态
    UPTKStreamIsCapturing(stream, &status);
    printf("结束后状态: %s\n", status == 0 ? "未捕获" : "捕获中");
    
    // 实例化并执行图
    UPTKGraphExec_t graphExec;
    UPTKGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    UPTKGraphLaunch(graphExec, stream);
    UPTKStreamSynchronize(stream);
    printf("图执行完成\n");
    
    // 清理资源
    UPTKGraphExecDestroy(graphExec);
    UPTKGraphDestroy(graph);
    UPTKStreamDestroy(stream);
    
    free(h_a);
    free(h_b);
    free(h_c);
    UPTKFree(d_a);
    UPTKFree(d_b);
    UPTKFree(d_c);
    
    return 0;
}

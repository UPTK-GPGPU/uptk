#include <stdio.h>
#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>



#define N 256

// 向量加法核函数
__global__ void vectorAdd(int *a, int *b, int *c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // 主机数据
    int h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 设备内存
    int *d_a, *d_b, *d_c;
    UPTKMalloc(&d_a, N * sizeof(int));
    UPTKMalloc(&d_b, N * sizeof(int));
    UPTKMalloc(&d_c, N * sizeof(int));
    
    // 1. 创建异步流
    UPTKStream_t stream1;
    UPTKStreamCreate(&stream1);
    
    // 2. 创建带标志的流（非阻塞）
    UPTKStream_t stream2;
    UPTKStreamCreateWithFlags(&stream2, UPTKStreamNonBlocking);
    
    // 3. 创建带优先级的流，dcu流优先级为范围[-1,1]
    UPTKStream_t stream3;
    int high_prio = -1;
    UPTKStreamCreateWithPriority(&stream3, UPTKStreamDefault, high_prio);
    
    // 4. 获取流关联的设备
    int device1, device2, device3;
    UPTKStreamGetDevice(stream1, &device1);
    UPTKStreamGetDevice(stream2, &device2);
    UPTKStreamGetDevice(stream3, &device3);
    
    printf("Stream devices: %d, %d, %d\n", device1, device2, device3);
    
    // 创建事件
    UPTKEvent_t event;
    UPTKEventCreate(&event);
    
    // 流1操作：数据传输和计算
    UPTKMemcpyAsync(d_a, h_a, N * sizeof(int), UPTKMemcpyHostToDevice, stream1);
    UPTKMemcpyAsync(d_b, h_b, N * sizeof(int), UPTKMemcpyHostToDevice, stream1);
    vectorAdd<<<1, N, 0, stream1>>>(d_a, d_b, d_c);
    UPTKEventRecord(event, stream1);
    
    // 5. 流2等待流1的事件
    UPTKStreamWaitEvent(stream2, event, 0);
    
    // 流2操作：验证性计算
    int *d_temp;
    UPTKMalloc(&d_temp, N * sizeof(int));
    vectorAdd<<<1, N, 0, stream2>>>(d_c, d_b, d_temp);
    
    // 流3操作：另一种计算
    vectorAdd<<<1, N, 0, stream3>>>(d_a, d_c, d_temp);
    
    // 6. 同步所有流
    UPTKStreamSynchronize(stream1);
    UPTKStreamSynchronize(stream2);
    UPTKStreamSynchronize(stream3);
    
    // 复制结果回主机
    UPTKMemcpyAsync(h_c, d_c, N * sizeof(int), UPTKMemcpyDeviceToHost, stream1);
    UPTKStreamSynchronize(stream1);
    
    // 验证结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            break;
        }
    }
    printf("Calculation %s\n", success ? "SUCCESS" : "FAILED");
    
    // 7. 销毁所有流
    UPTKStreamDestroy(stream1);
    UPTKStreamDestroy(stream2);
    UPTKStreamDestroy(stream3);
    
    // 清理
    UPTKEventDestroy(event);
    UPTKFree(d_a);
    UPTKFree(d_b);
    UPTKFree(d_c);
    UPTKFree(d_temp);
    
    return 0;
}

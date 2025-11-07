#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

// 向量加法内核
__global__ void vectorAdd(const int* a, const int* b, int* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("UPTK设备管理接口演示\n\n");
    
    // 1. 使用UPTKGetDeviceCount获取设备数量
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    printf("✓ UPTKGetDeviceCount: 系统中GPU数量 = %d\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("未找到可用的UPTK设备\n");
        return 1;
    }
    
    // 显示所有设备信息
    printf("\n=== 所有设备信息 ===\n");
    for (int i = 0; i < deviceCount; i++) {
        // 2. 使用UPTKDeviceGetName获取设备名称
        char deviceName[256];
        UPTKDeviceGetName(deviceName, 256, i);
        
        // 3. 使用UPTKDeviceGetAttribute获取设备属性
        int major, minor, multiProcessorCount, warpSize;
        UPTKDeviceGetAttribute(&major, UPTKDevAttrComputeCapabilityMajor, i);
        UPTKDeviceGetAttribute(&minor, UPTKDevAttrComputeCapabilityMinor, i);
        UPTKDeviceGetAttribute(&multiProcessorCount, UPTKDevAttrMultiProcessorCount, i);
        UPTKDeviceGetAttribute(&warpSize, UPTKDevAttrWarpSize, i);
        
        printf("设备 %d: %s\n", i, deviceName);
        printf("  多处理器数量: %d\n", multiProcessorCount);
        printf("  Warp大小: %d\n", warpSize);
    }
    
    // 4. 使用UPTKChooseDevice自动选择最佳设备
    printf("\n=== 自动设备选择 ===\n");
    UPTKDeviceProp prop;
    prop.computeMode = UPTKComputeModeDefault;
    
    int chosenDevice;
    UPTKError_t chooseResult = UPTKChooseDevice(&chosenDevice, &prop);
    
    if (chooseResult == UPTKSuccess) {
        printf("✓ UPTKChooseDevice: 自动选择设备 %d\n", chosenDevice);
    } else {
        printf("UPTKChooseDevice失败，使用设备0作为默认\n");
        chosenDevice = 0;
    }
    
    // 5. 使用UPTKSetDevice设置当前设备
    UPTKSetDevice(chosenDevice);
    printf("✓ UPTKSetDevice: 已设置当前设备为 %d\n", chosenDevice);
    
    // 6. 使用UPTKGetDevice验证当前设备
    int currentDevice;
    UPTKGetDevice(&currentDevice);
    printf("✓ UPTKGetDevice: 当前设备ID = %d\n", currentDevice);
    
    // 验证设备设置是否正确
    if (currentDevice != chosenDevice) {
        printf("警告: 当前设备与选择设备不匹配\n");
    }
    
    // 分配和初始化内存
    printf("\n=== 内存分配和初始化 ===\n");
    size_t size = N * sizeof(int);
    
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);
    
    int *d_a, *d_b, *d_c;
    UPTKMalloc(&d_a, size);
    UPTKMalloc(&d_b, size);
    UPTKMalloc(&d_c, size);
    
    // 初始化主机数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    printf("✓ 主机和设备内存分配完成\n");
    
    // 数据传输到设备
    UPTKMemcpy(d_a, h_a, size, UPTKMemcpyHostToDevice);
    UPTKMemcpy(d_b, h_b, size, UPTKMemcpyHostToDevice);
    
    // 执行内核
    printf("\n=== 内核执行 ===\n");
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    printf("✓ 内核启动完成\n");
    
    // 7. 使用UPTKDeviceSynchronize等待设备完成
    UPTKDeviceSynchronize();
    printf("✓ UPTKDeviceSynchronize: 设备操作已完成\n");
    
    // 将结果拷贝回主机
    UPTKMemcpy(h_c, d_c, size, UPTKMemcpyDeviceToHost);
    
    // 验证结果
    printf("\n=== 结果验证 ===\n");
    bool success = true;
    for (int i = 0; i < 5; i++) {
        int expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            success = false;
            break;
        }
    }
    
    printf("计算结果: %s\n", success ? "正确" : "错误");
    
    // 显示部分结果
    printf("前5个元素验证:\n");
    for (int i = 0; i < 5; i++) {
        printf("  h_a[%d]=%d + h_b[%d]=%d = h_c[%d]=%d\n", 
               i, h_a[i], i, h_b[i], i, h_c[i]);
    }
    
    // 8. 使用UPTKDeviceReset重置设备
    printf("\n=== 资源清理 ===\n");
    UPTKDeviceReset();
    printf("✓ UPTKDeviceReset: 设备已重置，所有资源已释放\n");
    
    // 释放主机内存
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("\n所有UPTK设备管理接口演示完成！\n");
    return 0;
}

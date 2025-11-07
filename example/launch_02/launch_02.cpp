#include <cstdio>
#include <vector>
#include <hip/hip_runtime.h>
#include "UPTK_runtime_api.h"

// 核函数 - 每个线程设置一个值
__global__ void simpleKernel(int *data, int value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = value + idx;
}

int main() {
    // 初始化
    UPTKInit(0);
    
    // 获取设备数量
    int deviceCount = 0;
    UPTKGetDeviceCount(&deviceCount);
    printf("Found %d UPTK devices\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("This test requires at least 2 devices\n");
        return -1;
    }

    // 使用前2个设备
    const int usedDevices = 2;
    const int dataSize = 256;  // 每个设备处理256个元素
    const int blockSize = 64;  // 每个块64个线程
    const int gridSize = dataSize / blockSize;  // 4个块

    // 存储设备和内存句柄
    std::vector<int*> deviceData(usedDevices);
    std::vector<int*> hostData(usedDevices);
    std::vector<UPTKstream> streams(usedDevices);
    
    // 准备启动参数
    std::vector<UPTKLaunchParams> launchParams(usedDevices);
    std::vector<void*> kernelArgs[usedDevices];

    // 为每个设备初始化
    for (int dev = 0; dev < usedDevices; dev++) {
        UPTKSetDevice(dev);
        
        // 分配内存
        UPTKMalloc(&deviceData[dev], dataSize * sizeof(int));
        hostData[dev] = new int[dataSize];
        UPTKStreamCreate(&streams[dev]);
        
        // 准备内核参数
        int value = (dev + 1) * 1000;  // 每个设备使用不同的初始值
        kernelArgs[dev].push_back((void*)&deviceData[dev]);
        kernelArgs[dev].push_back((void*)&value);
        
        // 设置启动参数
        launchParams[dev].func = (void*)simpleKernel;
        launchParams[dev].gridDim = gridSize;
        launchParams[dev].blockDim = blockSize;
        launchParams[dev].sharedMem = 0;
        launchParams[dev].stream = streams[dev];
        launchParams[dev].args = kernelArgs[dev].data();
    }

    // 在多设备上协同启动内核
    UPTKError_t result = UPTKLaunchCooperativeKernelMultiDevice(
        launchParams.data(), 
        usedDevices, 
        0
    );

    if (result != UPTKSuccess) {
        printf("Multi-device kernel launch failed: %s\n", UPTKGetErrorString(result));
        return -1;
    }

    // 同步设备并获取结果
    for (int dev = 0; dev < usedDevices; dev++) {
        UPTKSetDevice(dev);
        UPTKDeviceSynchronize();
        
        // 复制数据回主机
        UPTKMemcpy(hostData[dev], deviceData[dev], dataSize * sizeof(int), UPTKMemcpyDeviceToHost);
        
        // 打印前10个结果
        printf("Device %d results (first 10): ", dev);
        for (int i = 0; i < 10 && i < dataSize; i++) {
            printf("%d ", hostData[dev][i]);
        }
        printf("\n");
        
        // 验证结果
        int expectedBase = (dev + 1) * 1000;
        bool success = true;
        for (int i = 0; i < dataSize; i++) {
            if (hostData[dev][i] != expectedBase + i) {
                printf("Error at device %d, index %d: expected %d, got %d\n", 
                       dev, i, expectedBase + i, hostData[dev][i]);
                success = false;
                break;
            }
        }
        if (success) {
            printf("Device %d: All results correct!\n", dev);
        }
        
        // 清理资源
        delete[] hostData[dev];
        UPTKFree(deviceData[dev]);
        UPTKStreamDestroy(streams[dev]);
    }

    printf("Multi-device cooperative kernel test completed!\n");
    return 0;
}

#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

// 同步P2P拷贝函数
void demonstrateSyncP2PCopy(int* d_src, int srcDevice, int* d_dst, int dstDevice, size_t size) {
    printf("\n=== 同步P2P拷贝演示 ===\n");
    
    // 使用UPTKMemcpyPeer进行同步P2P拷贝
    UPTKMemcpyPeer(d_dst, dstDevice,    // 目标：目标设备的内存
                   d_src, srcDevice,    // 源：源设备的内存
                   size);               // 数据大小
    printf("✓ UPTKMemcpyPeer: 同步P2P拷贝完成\n");
    
    // 验证同步拷贝结果
    int *h_verify = (int*)malloc(size);
    UPTKSetDevice(dstDevice);
    UPTKMemcpy(h_verify, d_dst, size, UPTKMemcpyDeviceToHost);
    
    bool success = true;
    for (int i = 0; i < 5; i++) {
        if (h_verify[i] != i * 10) {
            success = false;
            break;
        }
    }
    printf("同步P2P拷贝验证: %s\n", success ? "成功" : "失败");
    
    // 显示部分数据以验证
    printf("数据验证（前5个元素）:\n");
    for (int i = 0; i < 5; i++) {
        printf("  元素[%d]: 期望值=%d, 实际值=%d\n", 
               i, i * 10, h_verify[i]);
    }
    
    free(h_verify);
}

// 异步P2P拷贝函数
void demonstrateAsyncP2PCopy(int* d_src, int srcDevice, int* d_dst, int dstDevice, size_t size) {
    printf("\n=== 异步P2P拷贝演示 ===\n");
    
    // 创建流用于异步操作
    UPTKStream_t stream;
    UPTKSetDevice(srcDevice);
    UPTKStreamCreate(&stream);
    
    // 修改源设备上的数据（乘以2以便区分）
    int *h_modified_data = (int*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_modified_data[i] = i * 20;
    }
    UPTKMemcpyAsync(d_src, h_modified_data, size, UPTKMemcpyHostToDevice, stream);
    printf("✓ 已修改源设备数据（乘以2）\n");
    
    // 使用UPTKMemcpyPeerAsync进行异步P2P拷贝
    UPTKMemcpyPeerAsync(d_dst, dstDevice,      // 目标：目标设备的内存
                        d_src, srcDevice,      // 源：源设备的内存
                        size,                  // 数据大小
                        stream);               // 使用流进行异步操作
    printf("✓ UPTKMemcpyPeerAsync: 异步P2P拷贝已启动\n");
    
    // 等待异步操作完成
    UPTKStreamSynchronize(stream);
    printf("✓ 异步P2P拷贝完成\n");
    
    // 验证异步拷贝结果
    int *h_verify = (int*)malloc(size);
    UPTKSetDevice(dstDevice);
    UPTKMemcpy(h_verify, d_dst, size, UPTKMemcpyDeviceToHost);
    
    bool success = true;
    for (int i = 0; i < 5; i++) {
        if (h_verify[i] != i * 20) {
            success = false;
            break;
        }
    }
    printf("异步P2P拷贝验证: %s\n", success ? "成功" : "失败");
    
    // 显示部分数据以验证
    printf("数据验证（前5个元素）:\n");
    for (int i = 0; i < 5; i++) {
        printf("  元素[%d]: 期望值=%d, 实际值=%d\n", 
               i, i * 20, h_verify[i]);
    }
    
    // 清理流和临时内存
    UPTKStreamDestroy(stream);
    free(h_modified_data);
    free(h_verify);
}

int main() {
    printf("UPTK P2P访问接口演示\n");
    
    // 检查GPU数量
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    printf("系统中GPU数量: %d\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("需要至少2个GPU才能演示P2P访问\n");
        return 0;
    }
    
    // 检查设备0和设备1之间的P2P支持
    // int canAccessPeer01, canAccessPeer10;
    // UPTKDeviceCanAccessPeer(&canAccessPeer01, 0, 1);
    // UPTKDeviceCanAccessPeer(&canAccessPeer10, 1, 0);
    
    // printf("P2P支持状态:\n");
    // printf("  设备0 → 设备1: %s\n", canAccessPeer01 ? "支持" : "不支持");
    // printf("  设备1 → 设备0: %s\n", canAccessPeer10 ? "支持" : "不支持");
    
    // if (!canAccessPeer01) {
    //     printf("设备0无法访问设备1，退出程序\n");
    //     return 0;
    // }
    
    size_t size = N * sizeof(int);
    
    // 在设备0和设备1上分配内存
    int *d_data0, *d_data1;
    
    UPTKSetDevice(0);
    UPTKMalloc(&d_data0, size);
    printf("✓ 在设备0上分配内存: %p\n", d_data0);
    
    UPTKSetDevice(1);
    UPTKMalloc(&d_data1, size);
    printf("✓ 在设备1上分配内存: %p\n", d_data1);
    
    // 初始化设备0上的数据
    UPTKSetDevice(0);
    int *h_init_data = (int*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_init_data[i] = i * 10;
    }
    UPTKMemcpy(d_data0, h_init_data, size, UPTKMemcpyHostToDevice);
    printf("✓ 初始化设备0上的数据\n");
    
    // 1. 使用UPTKDeviceEnablePeerAccess启用P2P访问
    UPTKSetDevice(0);
    UPTKError_t error = UPTKDeviceEnablePeerAccess(1, 0);
    if (error != UPTKSuccess) {
        printf("UPTKDeviceEnablePeerAccess失败: %s\n", UPTKGetErrorString(error));
        UPTKFree(d_data0);
        UPTKFree(d_data1);
        free(h_init_data);
        return 1;
    }
    printf("✓ UPTKDeviceEnablePeerAccess: 已启用设备0对设备1的P2P访问\n");
    
    // 2. 调用同步P2P拷贝函数
    demonstrateSyncP2PCopy(d_data0, 0, d_data1, 1, size);
    
    // 3. 调用异步P2P拷贝函数
    demonstrateAsyncP2PCopy(d_data0, 0, d_data1, 1, size);
    
    // 清理资源
    // UPTKSetDevice(0);
    // UPTKDeviceDisablePeerAccess(1);
    // printf("✓ 已禁用P2P访问\n");
    
    UPTKFree(d_data0);
    UPTKFree(d_data1);
    free(h_init_data);
    
    printf("\n所有P2P接口演示完成！\n");
    return 0;
}

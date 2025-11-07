#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#define N 256
#define SHM_NAME "/UPTK_ipc_demo"

int main() {
    printf("UPTK IPC 生产者进程 - 创建共享内存\n");
    
    UPTKError_t error;
    size_t size = N * sizeof(int);
    
    // 设置设备
    int device_id = 0;
    error = UPTKSetDevice(device_id);
    if (error != UPTKSuccess) {
        printf("设置设备失败: %s\n", UPTKGetErrorString(error));
        return 1;
    }
    
    // 1. 分配设备内存
    int *d_data;
    error = UPTKMalloc(&d_data, size);
    if (error != UPTKSuccess) {
        printf("UPTKMalloc失败: %s\n", UPTKGetErrorString(error));
        return 1;
    }
    printf("✓ UPTKMalloc: 分配设备内存成功\n");
    
    // 初始化设备数据
    int *h_data = (int*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 10;  // 乘以10以便区分
    }
    UPTKMemcpy(d_data, h_data, size, UPTKMemcpyHostToDevice);
    printf("初始化设备数据完成\n");
    
    // 2. 获取IPC内存句柄
    UPTKIpcMemHandle_t ipc_handle;
    error = UPTKIpcGetMemHandle(&ipc_handle, d_data);
    if (error != UPTKSuccess) {
        printf("UPTKIpcGetMemHandle失败: %s\n", UPTKGetErrorString(error));
        UPTKFree(d_data);
        free(h_data);
        return 1;
    }
    printf("✓ UPTKIpcGetMemHandle: 获取IPC内存句柄成功\n");
    
    // 创建共享内存区域来传递句柄
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open失败");
        UPTKFree(d_data);
        free(h_data);
        return 1;
    }
    
    // 设置共享内存大小
    if (ftruncate(shm_fd, sizeof(UPTKIpcMemHandle_t)) == -1) {
        perror("ftruncate失败");
        close(shm_fd);
        shm_unlink(SHM_NAME);
        UPTKFree(d_data);
        free(h_data);
        return 1;
    }
    
    // 映射共享内存
    void *shm_ptr = mmap(0, sizeof(UPTKIpcMemHandle_t), PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("mmap失败");
        close(shm_fd);
        shm_unlink(SHM_NAME);
        UPTKFree(d_data);
        free(h_data);
        return 1;
    }
    
    // 将IPC句柄复制到共享内存
    memcpy(shm_ptr, &ipc_handle, sizeof(UPTKIpcMemHandle_t));
    printf("✓ IPC句柄已写入共享内存\n");
    
    printf("\n生产者准备就绪，IPC句柄已共享。\n");
    printf("请在另一个终端运行消费者程序...\n");
    printf("按Enter键退出生产者...\n");
    getchar();
    
    // 清理资源
    munmap(shm_ptr, sizeof(UPTKIpcMemHandle_t));
    close(shm_fd);
    shm_unlink(SHM_NAME);
    UPTKFree(d_data);
    free(h_data);
    
    printf("生产者退出\n");
    return 0;
}

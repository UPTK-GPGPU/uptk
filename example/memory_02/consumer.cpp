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
    printf("UPTK IPC 消费者进程 - 使用共享内存\n");
    
    UPTKError_t error;
    size_t size = N * sizeof(int);
    
    // 设置设备（与生产者相同的设备）
    int device_id = 0;
    error = UPTKSetDevice(device_id);
    if (error != UPTKSuccess) {
        printf("设置设备失败: %s\n", UPTKGetErrorString(error));
        return 1;
    }
    
    // 打开共享内存区域
    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    if (shm_fd == -1) {
        perror("shm_open失败 - 请先运行生产者程序");
        return 1;
    }
    
    // 映射共享内存
    void *shm_ptr = mmap(0, sizeof(UPTKIpcMemHandle_t), PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("mmap失败");
        close(shm_fd);
        return 1;
    }
    
    // 从共享内存读取IPC句柄
    UPTKIpcMemHandle_t ipc_handle;
    memcpy(&ipc_handle, shm_ptr, sizeof(UPTKIpcMemHandle_t));
    printf("✓ 从共享内存读取IPC句柄成功\n");
    
    // 3. 使用UPTKIpcOpenMemHandle打开IPC内存
    int *d_shared_data;
    error = UPTKIpcOpenMemHandle((void**)&d_shared_data, ipc_handle, UPTKIpcMemLazyEnablePeerAccess);
    if (error != UPTKSuccess) {
        printf("UPTKIpcOpenMemHandle失败: %s\n", UPTKGetErrorString(error));
        munmap(shm_ptr, sizeof(UPTKIpcMemHandle_t));
        close(shm_fd);
        return 1;
    }
    printf("✓ UPTKIpcOpenMemHandle: 打开IPC内存成功\n");
    
    // 验证共享内存内容
    int *h_verify = (int*)malloc(size);
    UPTKMemcpy(h_verify, d_shared_data, size, UPTKMemcpyDeviceToHost);
    
    printf("从共享内存读取的数据:\n");
    bool success = true;
    for (int i = 0; i < 10; i++) {  // 只显示前10个元素
        printf("  h_verify[%d] = %d\n", i, h_verify[i]);
        if (h_verify[i] != i * 10) {
            success = false;
        }
    }
    printf("数据验证: %s\n", success ? "成功" : "失败");
    
    // 修改共享内存中的数据（演示双向通信）
    printf("\n修改共享内存中的数据（乘以2）...\n");
    for (int i = 0; i < N; i++) {
        h_verify[i] = h_verify[i] * 2;
    }
    UPTKMemcpy(d_shared_data, h_verify, size, UPTKMemcpyHostToDevice);
    printf("数据修改完成\n");
    
    // 4. 使用UPTKIpcCloseMemHandle关闭IPC内存
    error = UPTKIpcCloseMemHandle(d_shared_data);
    if (error != UPTKSuccess) {
        printf("UPTKIpcCloseMemHandle失败: %s\n", UPTKGetErrorString(error));
    } else {
        printf("✓ UPTKIpcCloseMemHandle: 关闭IPC内存成功\n");
    }
    
    // 清理资源
    munmap(shm_ptr, sizeof(UPTKIpcMemHandle_t));
    close(shm_fd);
    free(h_verify);
    
    printf("消费者退出\n");
    return 0;
}

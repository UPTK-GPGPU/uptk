#include <cuda.h>
#include <iostream>
#include <cstring>

// 极简错误信息打印函数
const char* cuGetErrorStr(CUresult err) {
    return (err == CUDA_SUCCESS) ? "CUDA_SUCCESS" : "CUDA Error";
}

int main() {
    // ===================== 1. 初始化参数（极简2D数组：2行3列int） =====================
    const int rows = 2;    // 行数
    const int cols = 3;    // 列数
    const int elem_size = sizeof(int); // 每个元素字节数
    const int row_bytes = cols * elem_size; // 每行字节数

    // 主机2D数组（连续内存模拟）
    int host_src[rows][cols] = {{1,2,3}, {4,5,6}};
    int host_dst[rows][cols] = {0}; // 用于验证拷贝结果

    // ===================== 2. 初始化CUDA上下文 =====================
    CUresult status = cuInit(0);
    if (status != CUDA_SUCCESS) {
        std::cerr << "cuInit failed: " << cuGetErrorStr(status) << std::endl;
        return -1;
    }

    CUdevice device;
    cuDeviceGet(&device, 0); // 获取第0块GPU
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, device); // 创建上下文

    // ===================== 3. 设备端分配2D内存 =====================
    CUdeviceptr dev_ptr;
    status = cuMemAlloc(&dev_ptr, rows * row_bytes); // 分配连续设备内存
    if (status != CUDA_SUCCESS) {
        std::cerr << "cuMemAlloc failed: " << cuGetErrorStr(status) << std::endl;
        cuCtxDestroy(ctx);
        return -1;
    }

    // ===================== 4. 配置cuMemcpy2D参数并执行拷贝（核心） =====================
    CUmemcpy2DParms copy_params = {}; // 初始化参数结构体（必须清零）
    copy_params.srcHost = host_src;   // 源：主机内存
    copy_params.dstDevice = dev_ptr;  // 目标：设备内存
    copy_params.WidthInBytes = row_bytes; // 每行拷贝字节数
    copy_params.Height = rows;        // 拷贝行数
    copy_params.srcPitch = row_bytes; // 源行间距（连续内存=每行字节数）
    copy_params.dstPitch = row_bytes; // 目标行间距（连续内存=每行字节数）
    copy_params.Kind = CU_MEMCPY_HOST_TO_DEVICE; // 拷贝类型：主机→设备

    status = cuMemcpy2D(&copy_params);
    if (status != CUDA_SUCCESS) {
        std::cerr << "cuMemcpy2D failed: " << cuGetErrorStr(status) << std::endl;
        cuMemFree(dev_ptr);
        cuCtxDestroy(ctx);
        return -1;
    }

    // ===================== 5. 验证：从设备拷贝回主机 =====================
    CUmemcpy2DParms verify_params = {};
    verify_params.srcDevice = dev_ptr; // 源：设备内存
    verify_params.dstHost = host_dst;  // 目标：主机内存
    verify_params.WidthInBytes = row_bytes;
    verify_params.Height = rows;
    verify_params.srcPitch = row_bytes;
    verify_params.dstPitch = row_bytes;
    verify_params.Kind = CU_MEMCPY_DEVICE_TO_HOST; // 设备→主机

    cuMemcpy2D(&verify_params);

    // ===================== 6. 打印结果验证 =====================
    std::cout << "Original host data:\n";
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) std::cout << host_src[i][j] << " ";
        std::cout << "\n";
    }

    std::cout << "\nCopied back from device:\n";
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) std::cout << host_dst[i][j] << " ";
        std::cout << "\n";
    }

    // ===================== 7. 清理资源 =====================
    cuMemFree(dev_ptr);
    cuCtxDestroy(ctx);

    return 0;
}

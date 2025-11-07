#include <iostream>
#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h> 

// UPTK错误检查宏
#define UPTK_CHECK(call) \
do { \
    UPTKError_t error = call; \
    if (error != UPTKSuccess) { \
        std::cerr << "UPTK error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "  Code: " << error << " (" << UPTKGetErrorName(error) << ")" << std::endl; \
        std::cerr << "  Description: " << UPTKGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

#define UPTK_LAST_ERROR() \
do { \
    UPTKError_t error = UPTKGetLastError(); \
    if (error != UPTKSuccess) { \
        std::cerr << "UPTK last error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "  Code: " << error << " (" << UPTKGetErrorName(error) << ")" << std::endl; \
        std::cerr << "  Description: " << UPTKGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 故意设计的错误核函数 - 会访问越界内存
__global__ void badKernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 故意访问越界内存
    if (idx < n * 2) {  // 故意超出范围
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 10; 
    const int size = N * sizeof(float); 

    float *h_a = (float*)malloc(size); 
    float *h_b = (float*)malloc(size); 
    float *h_c = (float*)malloc(size); 

    if (h_a == nullptr || h_b == nullptr || h_c == nullptr) {
        std::cerr << "Failed to allocate host memory!" << std::endl;
        return 1;
    }

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);      
        h_b[i] = static_cast<float>(i * 2);  
    }

    float *d_a, *d_b, *d_c;
    
    std::cout << "=== 测试1: 正常核函数执行 ===" << std::endl;
    UPTK_CHECK(UPTKMalloc(&d_a, size)); 
    UPTK_CHECK(UPTKMalloc(&d_b, size)); 
    UPTK_CHECK(UPTKMalloc(&d_c, size)); 

    UPTK_CHECK(UPTKMemcpy(d_a, h_a, size, UPTKMemcpyHostToDevice));
    UPTK_CHECK(UPTKMemcpy(d_b, h_b, size, UPTKMemcpyHostToDevice));

    int threadsPerBlock = 256; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 

    // 正常启动核函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // 检查核函数启动 - 没有错误
    std::cout << "检查正常核函数启动..." << std::endl;
    UPTK_LAST_ERROR();  // 不会报错

    UPTK_CHECK(UPTKDeviceSynchronize());

    UPTK_CHECK(UPTKMemcpy(h_c, d_c, size, UPTKMemcpyDeviceToHost));

    std::cout << "正常核函数执行成功!" << std::endl;

    // 清理设备内存
    UPTK_CHECK(UPTKFree(d_a)); 
    UPTK_CHECK(UPTKFree(d_b)); 
    UPTK_CHECK(UPTKFree(d_c)); 

    std::cout << "\n=== 测试2: 故意设计错误的核函数启动 ===" << std::endl;
    
    // 重新分配设备内存
    UPTK_CHECK(UPTKMalloc(&d_a, size)); 
    UPTK_CHECK(UPTKMalloc(&d_b, size)); 
    UPTK_CHECK(UPTKMalloc(&d_c, size)); 

    UPTK_CHECK(UPTKMemcpy(d_a, h_a, size, UPTKMemcpyHostToDevice));
    UPTK_CHECK(UPTKMemcpy(d_b, h_b, size, UPTKMemcpyHostToDevice));

    // 场景1: 启动配置错误 - 线程块过大
    std::cout << "场景1: 测试过大的线程块配置..." << std::endl;
    vectorAdd<<<1, 2000>>>(d_a, d_b, d_c, N); // 故意设置过大的线程块
    
    // 使用UPTKGetLastError检查异步错误
    UPTKError_t asyncError = UPTKGetLastError();
    if (asyncError != UPTKSuccess) {
        std::cerr << "异步错误捕获成功!" << std::endl;
        std::cerr << "错误代码: " << asyncError << std::endl;
        std::cerr << "错误名称: " << UPTKGetErrorName(asyncError) << std::endl;
        std::cerr << "错误描述: " << UPTKGetErrorString(asyncError) << std::endl;
    } else {
        std::cout << "意外: 没有检测到配置错误" << std::endl;
    }

    // 场景2: 启动错误的核函数 - 会访问越界内存
    std::cout << "\n场景2: 测试内存访问越界..." << std::endl;
    
    // 重置最后一个错误
    UPTKGetLastError();
    
    badKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // 立即检查启动错误
    asyncError = UPTKGetLastError();
    if (asyncError != UPTKSuccess) {
        std::cerr << "核函数启动错误: " << UPTKGetErrorString(asyncError) << std::endl;
    } else {
        std::cout << "核函数启动成功，等待同步时可能检测到运行时错误..." << std::endl;
    }

    // 同步时可能会检测到运行时错误
    UPTKError_t syncError = UPTKDeviceSynchronize();
    if (syncError != UPTKSuccess) {
        std::cerr << "设备同步时检测到错误!" << std::endl;
        std::cerr << "错误名称: " << UPTKGetErrorName(syncError) << std::endl;
        std::cerr << "错误描述: " << UPTKGetErrorString(syncError) << std::endl;
    }

    // 场景3: 使用无效的设备指针
    std::cout << "\n场景3: 测试无效设备指针..." << std::endl;
    
    // 重置错误状态
    UPTKGetLastError();
    
    float *invalid_ptr = nullptr;
    vectorAdd<<<1, 1>>>(invalid_ptr, d_b, d_c, 1);
    
    asyncError = UPTKGetLastError();
    if (asyncError != UPTKSuccess) {
        std::cerr << "检测到无效指针错误!" << std::endl;
        std::cerr << UPTKGetErrorName(asyncError) << ": " << UPTKGetErrorString(asyncError) << std::endl;
    }

    // 场景4: 测试各种错误情况的组合
    std::cout << "\n场景4: 测试错误信息函数的详细输出..." << std::endl;
    
    // 故意创建一个已知错误
    UPTKGetLastError(); // 重置
    
    // 尝试启动不存在的核函数（通过错误的启动配置模拟）
    vectorAdd<<<0, 0>>>(d_a, d_b, d_c, N); // 0线程
    
    UPTKError_t testError = UPTKGetLastError();
    std::cout << "详细错误分析:" << std::endl;
    std::cout << "  错误代码: " << testError << std::endl;
    std::cout << "  错误名称: " << UPTKGetErrorName(testError) << std::endl;
    std::cout << "  错误描述: " << UPTKGetErrorString(testError) << std::endl;

    // 清理
    UPTK_CHECK(UPTKFree(d_a)); 
    UPTK_CHECK(UPTKFree(d_b)); 
    UPTK_CHECK(UPTKFree(d_c)); 

    free(h_a); 
    free(h_b); 
    free(h_c); 

    std::cout << "\n=== 错误检查测试完成 ===" << std::endl;
    return 0;
}

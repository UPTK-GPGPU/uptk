#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstring>

// 读取文件内容的辅助函数
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cout << "无法打开文件: " << filename << std::endl;
        return {};
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cout << "读取文件失败: " << filename << std::endl;
        return {};
    }
    
    return buffer;
}

int main() {
    std::cout << "UPTK模块管理接口演示" << std::endl << std::endl;
    
    UPTKresult result;
    
    // 1. 使用UPTKModuleLoad从文件加载模块
    std::cout << "=== 从文件加载模块 ===" << std::endl;
    UPTKmodule module;
    
    result = UPTKModuleLoad(&module, "kernel.code");
    if (result == UPTKSuccess) {
        std::cout << "✓ UPTKModuleLoad: 模块加载成功 (错误码: " << result << ")" << std::endl;
    } else {
        std::cout << "✗ UPTKModuleLoad失败 (错误码: " << result << ")" << std::endl;
        std::cout << "请先编译内核: hipcc --genco -o kernel.code kernel.cpp" << std::endl;
        return 1;
    }
    
    // 2. 使用UPTKModuleLoadData从内存加载模块
    std::cout << std::endl << "=== 从内存数据加载模块 ===" << std::endl;
    UPTKmodule module_from_data;
    
    auto fileData = readFile("kernel.code");
    if (!fileData.empty()) {
        result = UPTKModuleLoadData(&module_from_data, fileData.data());
        if (result == UPTKSuccess) {
            std::cout << "✓ UPTKModuleLoadData: 从内存数据加载模块成功 (错误码: " << result << ")" << std::endl;
        } else {
            std::cout << "✗ UPTKModuleLoadData失败 (错误码: " << result << ")" << std::endl;
        }
    } else {
        std::cout << "无法读取代码对象文件，跳过内存加载测试" << std::endl;
    }
    
    // 3. 使用UPTKModuleGetFunction获取函数
    std::cout << std::endl << "=== 获取内核函数 ===" << std::endl;
    UPTKFunction_t vectorAdd_func, increment_func;
    
    result = UPTKModuleGetFunction(&vectorAdd_func, module, "vectorAdd");
    if (result == UPTKSuccess) {
        std::cout << "✓ UPTKModuleGetFunction: 获取vectorAdd函数成功 (错误码: " << result << ")" << std::endl;
    } else {
        std::cout << "✗ 获取vectorAdd函数失败 (错误码: " << result << ")" << std::endl;
    }
    
    result = UPTKModuleGetFunction(&increment_func, module, "incrementAndGet");
    if (result == UPTKSuccess) {
        std::cout << "✓ UPTKModuleGetFunction: 获取incrementAndGet函数成功 (错误码: " << result << ")" << std::endl;
    } else {
        std::cout << "✗ 获取incrementAndGet函数失败 (错误码: " << result << ")" << std::endl;
    }
    
    // 4. 使用UPTKModuleGetGlobal获取全局变量
    std::cout << std::endl << "=== 获取全局变量 ===" << std::endl;
    UPTKdeviceptr global_counter_ptr;
    size_t global_counter_size;
    
    result = UPTKModuleGetGlobal(&global_counter_ptr, &global_counter_size, module, "global_counter");
    if (result == UPTKSuccess) {
        std::cout << "✓ UPTKModuleGetGlobal: 获取全局变量global_counter成功 (错误码: " << result << ")" << std::endl;
        std::cout << "  全局变量地址: " << global_counter_ptr << std::endl;
        std::cout << "  全局变量大小: " << global_counter_size << " 字节" << std::endl;
        std::cout << "  已成功获取全局变量信息" << std::endl;
    } else {
        std::cout << "✗ 获取全局变量失败 (错误码: " << result << ")" << std::endl;
    }
    
    std::cout << std::endl << "=== 内存操作演示 ===" << std::endl;
    float* d_data;
    UPTKError_t malloc_result = UPTKMalloc(&d_data, 1024);
    if (malloc_result == UPTKSuccess) {
        std::cout << "✓ UPTKMalloc: 设备内存分配成功 (错误码: " << malloc_result << ")" << std::endl;
        
        // 释放内存
        UPTKError_t free_result = UPTKFree(d_data);
        if (free_result == UPTKSuccess) {
            std::cout << "✓ UPTKFree: 设备内存释放成功 (错误码: " << free_result << ")" << std::endl;
        } else {
            std::cout << "✗ UPTKFree失败 (错误码: " << free_result << ")" << std::endl;
        }
    } else {
        std::cout << "✗ UPTKMalloc失败 (错误码: " << malloc_result << ")" << std::endl;
    }
    
    // 5. 使用UPTKModuleUnload卸载模块
    std::cout << std::endl << "=== 模块卸载 ===" << std::endl;
    result = UPTKModuleUnload(module);
    if (result == UPTKSuccess) {
        std::cout << "✓ UPTKModuleUnload: 文件模块卸载成功 (错误码: " << result << ")" << std::endl;
    } else {
        std::cout << "✗ UPTKModuleUnload失败 (错误码: " << result << ")" << std::endl;
    }
    
    if (!fileData.empty()) {
        result = UPTKModuleUnload(module_from_data);
        if (result == UPTKSuccess) {
            std::cout << "✓ UPTKModuleUnload: 内存数据模块卸载成功 (错误码: " << result << ")" << std::endl;
        } else {
            std::cout << "✗ UPTKModuleUnload失败 (错误码: " << result << ")" << std::endl;
        }
    }
    
    std::cout << std::endl << "所有UPTK模块管理接口演示完成！" << std::endl;
    
    // 总结错误码
    std::cout << std::endl << "=== 执行结果总结 ===" << std::endl;
    std::cout << "错误码为0表示成功，非0表示失败" << std::endl;
    
    return 0;
}

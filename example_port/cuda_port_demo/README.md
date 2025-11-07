# CUDA程序移植到UPTK指南

## 简介

本示例演示如何将CUDA程序通过最小修改移植到UPTK框架。`cuda_port_demo`目录包含两个项目：

- `MyDemo_CUDA`：原始CUDA程序
- `MyDemo_UPTK`：移植后的UPTK程序

## 移植步骤

### 代码修改

#### API替换

- 将所有CUDA运行时API替换为对应的UPTK API
  - 例如：`cudaDeviceSynchronize()` → `UPTKDeviceSynchronize()`

#### 头文件添加

- 在程序中引入UPTK头文件`#include <UPTK_runtime_api.h>`

  - 在源文件中引入UPTK头文件时，需遵循特定的包含顺序：

  ```
  #include <cuda_runtime_api.h>  // 首先包含CUDA头文件
  #include <UPTK_runtime_api.h>  // 然后包含UPTK头文件
  ```

#### CMakeLists.txt修改

- 链接UPTK运行时库

**修改前：**

```
target_link_libraries(MyDemo cudart)
```

**修改后：**

```
target_link_libraries(MyDemo cudart UPTKrt)
```

## 编译运行指南

### MyDemo_CUDA (原始CUDA版本)

```
# 前提：加载dtk环境变量
source dtk/env.sh
source dtk/cuda/cuda-12/env.sh

cd uptk/example_port/cuda_port_demo/MyDemo_CUDA
mkdir build && cd build
cmake ..
make
./MyDemo
```

**预期输出：**

![image-20251106143057213](CUDA程序移植到UPTK指南.assets/image-20251106143057213.png)

### MyDemo_UPTK (移植后的UPTK版本)

```
# 前提：加载dtk环境变量
source dtk/env.sh
source dtk/cuda/cuda-12/env.sh
# 前提：加载UPTK环境变量
source uptk/output/env.sh

cd uptk/example_port/cuda_port_demo/MyDemo_UPTK
mkdir build && cd build
cmake ..
make
./MyDemo
```

**预期输出：**

![image-20251106145318764](CUDA程序移植到UPTK指南.assets/image-20251106145318764.png)

## 环境要求

- dtk开发环境

- UPTK开发环境

- CMake 3.8+
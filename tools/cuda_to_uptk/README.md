# CUDA → UPTK 转换工具

将 CUDA 主机端 API 调用自动替换为 UPTK 等价接口，并补充所需头文件与 CMake 链接项。

## 功能

- **Runtime API**：`cudaMalloc` → `UPTKMalloc`，`cudaMemcpyHostToDevice` → `UPTKMemcpyHostToDevice` 等
- **Driver API**：`cuInit` → `UPInit`，`cuModuleLoad` → `UPModuleLoad` 等
- **类型/枚举**：`cudaError_t` → `UPTKError_t`，`CUdevice` → `UPTKdevice` 等
- **头文件**：在 CUDA 头文件之后插入 `#include <UPTK_runtime_api.h>` / `#include <UPTK.h>`
- **CMake**：在 `target_link_libraries` 中追加 `UPTKrt` / `UPTKdrt`

**不会修改** kernel 代码（`__global__`、`<<<>>>`、`blockIdx` 等保持不变）。

## 构建与测试

这个工具是 **Python/Node 脚本**，不需要 `cmake` / `make` 编译。所谓“构建”就是生成 API 映射表 `api_map.json`。

### 环境要求

任选其一即可：

| 组件 | 用途 |
|------|------|
| **Node.js**（推荐，Windows 上通常已安装） | 生成映射表、运行测试 |
| **Python 3.8+** | 使用 CLI 转换文件/目录 |

### 构建（首次使用）

在项目根目录 `uptk-1.0/` 下执行：

```bash
# 生成 api_map.json（仓库里已有一份，uptk 源码更新后需重新生成）
node tools/cuda_to_uptk/generate_map.js
```

成功时会看到类似输出：

```
Wrote .../tools/cuda_to_uptk/api_map.json
runtime_functions=257, driver_functions=445, ...
```

### 测试工具本身

仍在项目根目录执行：

```bash
# 单元测试：对照官方 MyDemo_CUDA / MyDemo_UPTK 示例
node tools/cuda_to_uptk/test_converter.js

# 端到端测试：转换 demo 并写入 _test_output/MyDemo.cu
node tools/cuda_to_uptk/run_demo_test.js
```

全部通过时输出：

```
PASS: demo matches reference
PASS: kernel code unchanged
PASS: runtime sample conversion
PASS: end-to-end demo conversion
```

若已安装 Python，也可运行：

```bash
python tools/cuda_to_uptk/test_converter.py
```

### 使用工具转换代码

```bash
# 转换单个文件
python tools/cuda_to_uptk/cuda_to_uptk.py example_port/cuda_port_demo/MyDemo_CUDA/MyDemo.cu -o MyDemo_uptk.cu

# 转换整个目录，并修改 CMakeLists.txt
python tools/cuda_to_uptk/cuda_to_uptk.py example_port/cuda_port_demo/MyDemo_CUDA -o ./MyDemo_UPTK_out --cmake
```

### 测试转换后的程序能否编译运行（可选）

工具测试只验证 **源码转换是否正确**。若要验证转换结果能在 GPU 上跑通，需要先构建 UPTK，再编译转换后的 demo：

```bash
# 1. 构建 UPTK（需 DTK + CUDA 环境）
source dtk/env.sh
source dtk/cuda/cuda-12/env.sh
mkdir -p output build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../output
make && make install
source ../output/env.sh

# 2. 用工具生成 UPTK 版源码
python tools/cuda_to_uptk/cuda_to_uptk.py ../example_port/cuda_port_demo/MyDemo_CUDA -o ../MyDemo_converted --cmake

# 3. 编译运行
cd ../MyDemo_converted
mkdir build && cd build
cmake .. && make && ./MyDemo
```

预期输出为向量加法结果（与 `MyDemo_UPTK` 相同）。

## 目录结构

```
tools/cuda_to_uptk/
├── cuda_to_uptk.py    # Python CLI 入口
├── converter.py       # Python 转换逻辑
├── generate_map.py    # 从 uptk 源码生成 api_map.json
├── generate_map.js    # Node 版映射生成脚本
├── convert_lib.js     # Node 版转换逻辑（用于测试）
├── api_map.json       # API 映射表
├── test_converter.py  # Python 单元测试
├── test_converter.js  # Node 单元测试
├── run_demo_test.js   # 端到端示例测试
└── README.md
```

## 使用方法

```bash
# 1. 生成/更新 API 映射表（首次使用或 uptk 源码更新后）
node tools/cuda_to_uptk/generate_map.js
# 或: python tools/cuda_to_uptk/generate_map.py

# 2. 转换单个文件（Python CLI）
python tools/cuda_to_uptk/cuda_to_uptk.py input.cu -o output.cu

# 3. 转换目录并处理 CMakeLists.txt
python tools/cuda_to_uptk/cuda_to_uptk.py ./MyDemo_CUDA -o ./MyDemo_UPTK_out --cmake

# 4. 运行测试
node tools/cuda_to_uptk/test_converter.js
node tools/cuda_to_uptk/run_demo_test.js
# 或: python tools/cuda_to_uptk/test_converter.py
```

## 参考

映射规则来源于：

- `src/runtime/runtime_fun_convert.cpp`
- `src/driver/driver_fun_convert.cpp`
- `include/UPTK_runtime_api.h`
- `include/UPTK.h`

示例对照见 `example_port/cuda_port_demo/MyDemo_CUDA` 与 `MyDemo_UPTK`。

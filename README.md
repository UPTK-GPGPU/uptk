# UPTK

## 简介

UPTK是一个通用异构程序编程框架，能够支持cuda程序快速迁移运行至国产GPU平台。

## 构建安装步骤

UPTK安装前需加载dtk环境，dtk环境加载步骤如下：

```
source dtk/env.sh
```

UPTK构建安装步骤如下：

```
cd UPTK
mkdir output    //创建output目录，后续编译生成的库安装到该目录下
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../output     //CMAKE_INSTALL_PREFIX 指定为项目根目录下output目录
make
make install
```

## 使用示例

进入项目example目录，进入各对应模块，编译运行使用示例。
demo目录的示例为最基础的异构计算程序，安装好UPTK库之后，可以用该示例来测试是否安装成功。

```
cd uptk/example
ls
cd demo
make run
```

## 程序移植指南

项目提供了从CUDA/HIP程序迁移到UPTK的完整示例，位于 `uptk/example_port` 目录：

- **CUDA程序迁移**：参考 `uptk/example_port/cuda_port_demo/README.md`
- **HIP程序迁移**：参考 `uptk/example_port/hip_port_demo/README.md`

这些示例演示了如何通过最小代码修改，将现有的CUDA或HIP程序快速移植到UPTK框架。

### 构建方式支持

- **CMake项目**：参考 `uptk/example_port` 目录中的示例
- **Makefile项目**：参考 `uptk/example` 目录中的示例
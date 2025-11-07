# UPTK IPC 内存共享示例

## 概述

本例子演示了如何使用UPTK IPC接口在不同进程间共享GPU内存。通过两个独立的程序（生产者和消费者），展示了完整的UPTK IPC工作流程。

## 功能特性

- ✅ 使用 `UPTKIpcGetMemHandle` 获取IPC内存句柄
- ✅ 使用 `UPTKIpcOpenMemHandle` 打开共享内存
- ✅ 使用 `UPTKIpcCloseMemHandle` 关闭共享内存
- ✅ 使用POSIX共享内存传递IPC句柄
- ✅ 跨进程GPU内存数据共享和修改

## 运行演示

**终端1 - 启动生产者:**
```bash
make run
```

**终端2 - 启动消费者:**
```bash
make run-consumer
```

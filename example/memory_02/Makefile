# UPTK编译器
HIPCC = hipcc

# 编译选项
UPTK_ROOT = ../..
INCLUDE_DIR = $(UPTK_ROOT)/output/include
LIB_DIR = $(UPTK_ROOT)/output/lib

CFLAGS = -I$(INCLUDE_DIR)
LDFLAGS = -L$(LIB_DIR) -lUPTKrt

# 目标文件
TARGETS = producer consumer

# 默认目标
all: $(TARGETS)

# 编译生产者
producer: producer.cpp
	$(HIPCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# 编译消费者
consumer: consumer.cpp
	$(HIPCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# 运行目标 - 先启动生产者，然后在新终端中启动消费者
run: producer consumer
	@echo "=== 启动UPTK IPC演示 ==="
	@echo "1. 首先启动生产者进程..."
	@echo "2. 然后在另一个终端中运行: make run-consumer"
	@echo "3. 观察输出后，回到生产者终端按Enter退出"
	@echo ""
	export LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH && ./producer

# 运行消费者
run-consumer: consumer
	@echo "=== 启动消费者进程 ==="
	export LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH && ./consumer

# 清理
clean:
	rm -f $(TARGETS)
	rm -f /dev/shm/UPTK_ipc_demo 2>/dev/null || true

.PHONY: all run run-consumer clean
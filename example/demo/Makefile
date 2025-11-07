# Makefile for UPTK Demo

HIPCC = hipcc

UPTK_ROOT = ../..
INCLUDE_DIR = $(UPTK_ROOT)/output/include
LIB_DIR = $(UPTK_ROOT)/output/lib

CFLAGS = -I$(INCLUDE_DIR)
LDFLAGS = -L$(LIB_DIR) -lUPTKrt

TARGET = demo
SOURCE = demo.cpp

.PHONY: all
all: $(TARGET)
	@echo "compilation completed. Use 'make run' to run the program."


$(TARGET): $(SOURCE)
	$(HIPCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
	@echo "compilation successful: $@"

.PHONY: run
run: $(TARGET)
	export LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH && ./$(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET)
	@echo "cleaned build file"

# 显示帮助信息
.PHONY: help
help:
	@echo "UPTK Demo Makefile 使用说明:"
	@echo "  make          编译demo程序"
	@echo "  make run      运行demo程序"
	@echo "  make clean    清理生成的文件"
	@echo "  make help     显示此帮助信息"

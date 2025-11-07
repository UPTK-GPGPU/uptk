HIPCC = hipcc

UPTK_ROOT = ../..
INCLUDE_DIR = $(UPTK_ROOT)/output/include
LIB_DIR = $(UPTK_ROOT)/output/lib

CFLAGS = -I$(INCLUDE_DIR)
LDFLAGS = -L$(LIB_DIR) -lUPTKrt

TARGET = graph_01
SOURCE = graph_01.cpp

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


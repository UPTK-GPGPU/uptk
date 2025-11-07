HIPCC = hipcc

UPTK_ROOT = ../..
INCLUDE_DIR = $(UPTK_ROOT)/output/include
LIB_DIR = $(UPTK_ROOT)/output/lib

CFLAGS = -I$(INCLUDE_DIR)
LDFLAGS = -L$(LIB_DIR) -lUPTKrt

TARGET = module_01
CO_TARGET = kernel.code

all: $(CO_TARGET) $(TARGET)
	@echo "compilation completed. Use 'make run' to run the program."

$(CO_TARGET): kernel.cpp
	$(HIPCC) $(CFLAGS) --genco -o $@ $< $(LDFLAGS)

$(TARGET): module_01.cpp
	$(HIPCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: run
run: $(CO_TARGET) $(TARGET)
	export LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH && ./$(TARGET)

clean:
	rm -f $(TARGET) $(CO_TARGET)
	@echo "cleaned build file"

.PHONY: all run clean

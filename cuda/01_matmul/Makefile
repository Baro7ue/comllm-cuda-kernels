# 在 Jetson 上运行：make run
NVCC  ?= /usr/local/cuda/bin/nvcc
ARCH  ?= sm_87
FLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo
BUILD := ../../build

$(BUILD)/matmul: matmul.cu ref_kernels.cuh my_kernels.cuh my_v3.cuh my_v4.cuh
	mkdir -p $(BUILD)
	$(NVCC) $(FLAGS) -o $@ $<

run: $(BUILD)/matmul
	$(BUILD)/matmul

clean:
	rm -f $(BUILD)/matmul

.PHONY: run clean

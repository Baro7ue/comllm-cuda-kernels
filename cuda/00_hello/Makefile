# 在 Jetson 上运行：make && ../../build/vector_add
# sm_87 是 Orin 系列 GPU 的计算能力；-lineinfo 让 Nsight 能把耗时对到源码行。
NVCC  ?= /usr/local/cuda/bin/nvcc
ARCH  ?= sm_87
FLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo
BUILD := ../../build

$(BUILD)/vector_add: vector_add.cu
	mkdir -p $(BUILD)
	$(NVCC) $(FLAGS) -o $@ $<

run: $(BUILD)/vector_add
	$(BUILD)/vector_add

clean:
	rm -f $(BUILD)/vector_add

.PHONY: run clean

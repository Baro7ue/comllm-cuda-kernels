#!/bin/bash
# 在 Jetson 上运行：查 llama.cpp 的 mul_mat_q 用的是 int8 张量核（IMMA 指令）还是 dp4a（IDP.4A 指令）。
# 只读：看源码、编译选项和编译好的机器码，不跑任何 GPU 任务。
# 用法（在 Mac 上）：ssh Jetson 'bash -s' < scripts/check_mmq_mma.sh
set -u
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
CUOBJDUMP=/usr/local/cuda/bin/cuobjdump

section() { echo; echo "=== $1"; }

section "正在运行的 llama 进程（有的话先不查，免得抢 CPU）"
if pgrep -af 'llama-(server|bench|cli)'; then echo "有任务在跑，退出"; exit 1; fi
echo "无"

section "llama.cpp 版本"
git -C "$LLAMA_DIR" log -1 --format='%H  %cd' 2>/dev/null

section "编译选项（build/CMakeCache.txt）"
grep -E '^(CMAKE_CUDA_ARCHITECTURES|CMAKE_BUILD_TYPE|GGML_CUDA[A-Z0-9_]*)[:=]' "$LLAMA_DIR/build/CMakeCache.txt"

section "CUDA 库文件"
LIB=$(find "$LLAMA_DIR/build" -name 'libggml-cuda.so*' -type f | head -1)
if [ -z "$LIB" ]; then
  echo "没找到 libggml-cuda.so，可能是静态编译；改查 llama-bench"
  LIB="$LLAMA_DIR/build/bin/llama-bench"
fi
ls -l "$LIB"

section "库里带了哪些架构的机器码（ELF）和中间码（PTX）"
"$CUOBJDUMP" -lelf "$LIB" | awk -F. '{print $(NF-1)}' | sort | uniq -c
"$CUOBJDUMP" -lptx "$LIB" | awk -F. '{print $(NF-1)}' | sort | uniq -c

section "源码：什么条件下算 TURING/AMPERE_MMA_AVAILABLE（common.cuh）"
grep -nE '(TURING|AMPERE|NEW|VOLTA|BLACKWELL)_MMA_AVAILABLE' "$LLAMA_DIR/ggml/src/ggml-cuda/common.cuh" | head -20

section "源码：mmq.cuh 里按什么条件选张量核"
grep -nE '_MMA_AVAILABLE|mma_available' "$LLAMA_DIR/ggml/src/ggml-cuda/mmq.cuh" | head -30

section "机器码：每个 Q8_0 的 mul_mat_q 函数里，各种乘加指令的条数"
# ggml_type 8 = GGML_TYPE_Q8_0。函数名是编译器改写过的名字，形如 _Z9mul_mat_qIL9ggml_type8ELi128ELb0EEv...
# IMMA = int8 张量核；HMMA = fp16 张量核；IDP.4A = dp4a（一条做 4 个 8 位乘加）
"$CUOBJDUMP" -sass "$LIB" | awk '
  /arch = sm_/      { arch = $3 }
  /Function :/      { fn = $3; next }
  fn ~ /mul_mat_qIL9ggml_type8E/ {
    key = arch "  " fn
    seen[key] = 1
    if ($0 ~ /IMMA/)   imma[key]++
    if ($0 ~ /HMMA/)   hmma[key]++
    if ($0 ~ /IDP\.4A/) idp[key]++
  }
  END {
    printf "%-8s %-8s %-8s  %s\n", "IMMA", "HMMA", "IDP.4A", "架构  函数"
    for (k in seen) printf "%-8d %-8d %-8d  %s\n", imma[k], hmma[k], idp[k], k
  }' | sort -k4

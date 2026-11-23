#!/usr/bin/env bash
# 在 Jetson 上运行：核对系统、CUDA、llama.cpp、模型、电源模式和频率，并检查有没有别的任务在用 GPU。
# 输出同时打到屏幕和 results/<日期>/env.txt。
#   bash scripts/env_check.sh
set -u
cd "$(dirname "$0")/.."
OUT_DIR=${OUT_DIR:-results/$(date +%F)}
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
MODEL=${MODEL:-$HOME/models/comllm-0.5b-soft-T0.3-q8_0.gguf}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
mkdir -p "$OUT_DIR"

section() { printf '\n=== %s\n' "$1"; }

{
section "时间与主机"
date -Is; hostname; uptime

section "L4T / JetPack"
head -1 /etc/nv_tegra_release
dpkg-query -W -f='${Package} ${Version}\n' nvidia-jetpack nvidia-l4t-core 2>/dev/null
uname -r

section "CUDA"
"$NVCC" --version | tail -2
ls -d /usr/local/cuda*/

section "电源模式"
nvpmodel -q 2>&1 | head -2

section "频率（devfreq：GPU 等；cur=当前，单位 Hz）"
for d in /sys/class/devfreq/*; do
  printf '%-28s name=%s cur=%s min=%s max=%s gov=%s\n' "$(basename "$d")" \
    "$(cat "$d/name" 2>/dev/null)" "$(cat "$d/cur_freq" 2>/dev/null)" \
    "$(cat "$d/min_freq" 2>/dev/null)" "$(cat "$d/max_freq" 2>/dev/null)" "$(cat "$d/governor" 2>/dev/null)"
done
printf 'CPU 当前频率（kHz）: '; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | tr '\n' ' '; echo

section "jetson_clocks --show（要免密 sudo；没有就跳过）"
sudo -n jetson_clocks --show 2>&1 | head -40 || true

section "内存与磁盘"
free -m
df -h "$HOME" | tail -1

section "llama.cpp"
git -C "$LLAMA_DIR" log -1 --format='%H %cd' --date=iso
git -C "$LLAMA_DIR" status --short | head
ls "$LLAMA_DIR/build/bin" | grep -E '^llama-(bench|server|cli|batched-bench|eval-callback)$'
"$LLAMA_DIR/build/bin/llama-bench" --help 2>&1 | grep -E '^\s+-(d|fa|ub|p),' || true
grep -E '^(GGML_CUDA[A-Z_]*|CMAKE_CUDA_ARCHITECTURES|CMAKE_BUILD_TYPE):' "$LLAMA_DIR/build/CMakeCache.txt"

section "部署时 llama-server 日志里的 flash_attn 和 batch 设置（找最近一份）"
LOG=$(find "$HOME" -maxdepth 4 -name 'llama_server_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [ -n "$LOG" ]; then echo "$LOG"; grep -m6 -E 'flash_attn|n_batch|n_ubatch|n_ctx ' "$LOG"; else echo "没找到"; fi

section "模型"
ls -la "$HOME/models/"
sha256sum "$MODEL"

section "性能分析工具"
command -v nsys ncu || true
ls -d /opt/nvidia/nsight-systems*/*/ /opt/nvidia/nsight-compute*/*/ 2>/dev/null || true

section "正在运行的 llama / python 进程（有的话先别跑基准）"
pgrep -af 'llama-|python3' || echo "无"

section "tegrastats 快照（GR3D_FREQ 是 GPU 占用）"
timeout 3 tegrastats --interval 1000 2>&1 | head -2
} 2>&1 | tee "$OUT_DIR/env.txt"

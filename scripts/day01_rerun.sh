#!/usr/bin/env bash
# 在 Jetson 上运行：第 1 天的补测。
#   1. 部署时 llama-server 的 flash attention 到底开没开（查日志）
#   2. nsys 逐内核计时（进到 CUDA graph 里面看），flash attention 开/关各一次，再加部署场景
#   3. 跑基准的同时每 5 ms 采一次 GPU 频率，看它是不是一直在最高频率
set -uo pipefail
cd "$(dirname "$0")/.."
OUT_DIR=${OUT_DIR:-results/$(date +%F)}
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
MODEL=${MODEL:-$HOME/models/comllm-0.5b-soft-T0.3-q8_0.gguf}
mkdir -p "$OUT_DIR"

echo "=== 1. 部署日志里的 flash attention"
LOG=$(find "$HOME" -maxdepth 4 -name 'llama_server_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
echo "$LOG"
grep -n -i -m8 -E 'flash|n_batch|n_ubatch|graph' "$LOG" | cut -c1-220 | tee "$OUT_DIR/server_log_flash.txt"

echo "=== 2. nsys 逐内核"
FA=1 N=278 D=0   bash scripts/profile_nsys.sh > "$OUT_DIR/rerun_fa1_p278.log" 2>&1; echo "fa1 p278 exit=$?"
FA=0 N=278 D=0   bash scripts/profile_nsys.sh > "$OUT_DIR/rerun_fa0_p278.log" 2>&1; echo "fa0 p278 exit=$?"
FA=1 N=145 D=133 bash scripts/profile_nsys.sh > "$OUT_DIR/rerun_fa1_p145_d133.log" 2>&1; echo "fa1 p145 d133 exit=$?"

echo "=== 3. 基准期间的 GPU 频率"
F=/sys/class/devfreq/17000000.gpu/cur_freq
( while :; do echo "$(date +%s.%N) $(cat $F)"; sleep 0.005; done > "$OUT_DIR/gpu_freq_during_bench.txt" ) &
SAMPLER=$!
sleep 1
"$LLAMA_DIR/build/bin/llama-bench" -m "$MODEL" -ngl 99 -n 0 -p 278 -fa 1 -r 20 -o jsonl 2>/dev/null \
  | grep '^{' > "$OUT_DIR/bench_freqrun.jsonl"
sleep 1
kill $SAMPLER 2>/dev/null
awk '{c[$2]++} END {for (f in c) printf "%d MHz: %d 次采样\n", f/1e6, c[f]}' "$OUT_DIR/gpu_freq_during_bench.txt" | sort -n | tee "$OUT_DIR/gpu_freq_hist.txt"
ls "$OUT_DIR" | grep -E 'node|freq|rerun'
echo RERUN-DONE

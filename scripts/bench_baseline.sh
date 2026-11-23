#!/usr/bin/env bash
# 在 Jetson 上运行：用 llama-bench 量 0.5B Q8_0 "读提示词"要多久（单位 ms）。
#   bash scripts/bench_baseline.sh
# 三组：
#   deploy  部署场景：缓存里已有 133 个 token，再读 145 个（-d 133 -p 145）
#   nocache 不开缓存：读 145 个、读 278 个
#   sweep   扫提示词长度，画 t(N) 曲线，看截距（固定开销）和台阶
# 每组都同时跑 flash attention 关（0）和开（1），对照部署日志里实际用的那个。
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
MODEL=${MODEL:-$HOME/models/comllm-0.5b-soft-T0.3-q8_0.gguf}
REPS=${REPS:-20}
OUT_DIR=${OUT_DIR:-results/$(date +%F)}
BENCH="$LLAMA_DIR/build/bin/llama-bench"
mkdir -p "$OUT_DIR"

# 别的 llama 进程会和我们抢 GPU 和内存带宽，量出来的数就不准。
if pgrep -f 'llama-(server|bench|cli)' >/dev/null && [ "${FORCE:-0}" != 1 ]; then
  echo "有 llama 进程在运行，先不跑（确认不影响可加 FORCE=1）："; pgrep -af 'llama-'; exit 1
fi

clocks() { for d in /sys/class/devfreq/*; do echo "$(cat "$d/name" 2>/dev/null || basename "$d") cur=$(cat "$d/cur_freq" 2>/dev/null)"; done; }

run() {  # run <组名> <llama-bench 参数...>
  local name=$1; shift
  echo ">> $name: llama-bench $*"
  "$BENCH" -m "$MODEL" -ngl 99 -n 0 -r "$REPS" -o jsonl "$@" > "$OUT_DIR/bench_$name.jsonl"
}

clocks > "$OUT_DIR/clocks_before_bench.txt"
run deploy  -p 145 -d 133 -fa 0,1
run nocache -p 145,278 -fa 0,1
run sweep   -p 16,32,64,96,128,144,145,160,192,256,278,384,512 -fa 0,1
clocks > "$OUT_DIR/clocks_after_bench.txt"

python3 tools/bench_summary.py "$OUT_DIR"/bench_deploy.jsonl "$OUT_DIR"/bench_nocache.jsonl "$OUT_DIR"/bench_sweep.jsonl \
  | tee "$OUT_DIR/bench_summary.md"

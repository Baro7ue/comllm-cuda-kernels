#!/usr/bin/env bash
# 在 Jetson 上运行：用 Nsight Systems（nsys）记下 llama-bench 读提示词时每个 GPU 内核的起止时间。
#   bash scripts/profile_nsys.sh            # 默认读 278 个 token，不带缓存
#   N=145 D=133 bash scripts/profile_nsys.sh  # 部署场景
# 产物：results/<日期>/nsys_p<N>_d<D>_fa<FA>.sqlite，再由 tools/nsys_summary.py 汇总成表。
#
# 注意：记录本身有开销（每次发起内核多几微秒），所以这里的总时间会比 bench_baseline.sh 的长。
# 这张表用来看"比例"，绝对时间以 bench_baseline.sh 为准。
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
MODEL=${MODEL:-$HOME/models/comllm-0.5b-soft-T0.3-q8_0.gguf}
N=${N:-278}; D=${D:-0}; FA=${FA:-1}; REPS=${REPS:-5}
# llama.cpp 读提示词时用 CUDA graph（把一趟前向的全部内核打包成一次发起）。
# nsys 默认把整张 graph 记成一条，看不到里面的内核；GRAPH_TRACE=node 让它逐个内核记。
GRAPH_TRACE=${GRAPH_TRACE:-node}
OUT_DIR=${OUT_DIR:-results/$(date +%F)}
NSYS=${NSYS:-$(command -v nsys || ls /opt/nvidia/nsight-systems*/*/bin/nsys 2>/dev/null | tail -1)}
[ -x "$NSYS" ] || { echo "找不到 nsys。装法：sudo apt install nsight-systems（或设 NSYS=路径）"; exit 1; }
mkdir -p "$OUT_DIR"
BASE="$OUT_DIR/nsys_p${N}_d${D}_fa${FA}_${GRAPH_TRACE}${TAG:-}"

if pgrep -f 'llama-(server|bench|cli)' >/dev/null && [ "${FORCE:-0}" != 1 ]; then
  echo "有 llama 进程在运行，先不跑（确认不影响可加 FORCE=1）："; pgrep -af 'llama-'; exit 1
fi

# --trace=cuda：只记 CUDA 调用和 GPU 上的内核/拷贝；--sample/--cpuctxsw=none：不采 CPU，减小开销。
"$NSYS" profile --trace=cuda --cuda-graph-trace="$GRAPH_TRACE" --sample=none --cpuctxsw=none --force-overwrite=true -o "$BASE" \
  "$LLAMA_DIR/build/bin/llama-bench" -m "$MODEL" -ngl 99 -n 0 -p "$N" -d "$D" -fa "$FA" -r "$REPS" -o jsonl \
  | tee "$BASE.bench.jsonl"

"$NSYS" export --type=sqlite --force-overwrite=true -o "$BASE.sqlite" "$BASE.nsys-rep"
python3 tools/nsys_summary.py "$BASE.sqlite" | tee "$BASE.summary.md"

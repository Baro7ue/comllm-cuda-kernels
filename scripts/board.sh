#!/usr/bin/env bash
# 在 Mac 上运行：把本仓库传到 Jetson，编译矩阵乘练习，运行，打印结果。
#   bash scripts/board.sh              # 默认大小：145 个 token，896 → 4864
#   bash scripts/board.sh 145 896 5000 # 换个大小试试（N K M）
#
# 传的是"这个脚本所在的仓库"，和你当前在哪个目录无关，不会传错。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$REPO/cuda/01_matmul/matmul.cu" ] || { echo "找不到 $REPO/cuda/01_matmul/matmul.cu，仓库不对"; exit 1; }

rsync -a --exclude .git --exclude build --exclude results --exclude 'HANDOFF_*' "$REPO/" Jetson:jetson-llm-kernels/
ssh Jetson "cd ~/jetson-llm-kernels && make -s -C cuda/01_matmul && ./build/matmul $*"

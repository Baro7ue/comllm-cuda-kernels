"""把 llama-bench 的 jsonl 输出整理成"每次要多少毫秒"的表。

llama-bench 默认报 token/s；我们关心的是一次决策要等多久，所以换算成每次的 ms，
并给出中位数（p50）。每行一种设置：读 N 个新 token、缓存里已有 D 个、flash attention 开或关。

用法：python3 tools/bench_summary.py results/<日期>/bench_*.jsonl
"""
import json
import statistics
import sys


def rows(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    yield p, json.loads(line)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("| 文件 | 新 token N | 已缓存 D | FA | 次数 | p50 ms | 平均 ms | 标准差 ms | 最快 ms | 最慢 ms | token/s（平均） |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for path, r in rows(sys.argv[1:]):
        if r.get("n_gen", 0) != 0:
            continue
        samples = [s / 1e6 for s in r.get("samples_ns") or []] or [r["avg_ns"] / 1e6]
        n = r["n_prompt"]
        print("| %s | %d | %d | %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.0f |" % (
            path.rsplit("/", 1)[-1], n, r.get("n_depth", 0), r.get("flash_attn"), len(samples),
            statistics.median(samples), statistics.fmean(samples),
            statistics.pstdev(samples), min(samples), max(samples), r.get("avg_ts", 0)))
    first = next(rows(sys.argv[1:]))[1]
    print("\nllama.cpp %s（build %s），%s，GPU：%s" % (
        first.get("build_commit"), first.get("build_number"), first.get("model_type"), first.get("gpu_info")))


if __name__ == "__main__":
    main()

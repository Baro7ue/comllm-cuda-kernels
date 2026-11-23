"""把 nsys 导出的 sqlite 汇总成"每类算子花了多少时间"的表。

nsys 记下了每个 GPU 内核（kernel）的名字、开始时刻、结束时刻。这个脚本做三件事：
  1. 切段：一次"读提示词"是一串紧挨着的内核；两次之间 CPU 要做别的事，会留下一段空隙。
     按空隙把全部内核切成一趟一趟（pass），每趟对应一次前向计算。
  2. 自检：同一张计算图每趟发起的内核个数应该相同。个数不一致就说明切错了，会报警。
  3. 汇总：按内核名字归类，报每类每趟的时间（各趟的中位数）和占比。

"GPU 忙碌时间" = 各内核的 (结束 − 开始) 之和；"墙上时间" = 这一趟第一个内核开始到最后一个结束。
两者的差是 GPU 在等 CPU 发下一个内核的空闲时间，这部分和算子本身快慢无关。

用法：
  python3 tools/nsys_summary.py results/<日期>/nsys_p278_d0_fa0.sqlite
  python3 tools/nsys_summary.py x.sqlite --gap-ms 0.5      # 手动指定切段的空隙
  python3 tools/nsys_summary.py x.sqlite --every 2 --offset 1   # 带缓存时两趟一组，只看第 2 趟
"""
import argparse
import collections
import re
import sqlite3
import statistics
import sys

# 内核名字 → 类别。名字来自 llama.cpp 的 ggml-cuda 源码，按出现顺序匹配；
# 没匹配上的归到"其他"并在表后列出原名，核对后把规则补进来。
RULES = [
    (r"quantize_mmq|quantize_q8_1|quantize_row", "激活量化"),
    (r"mul_mat_q_stream_k_fixup", "矩阵乘（量化）收尾"),
    (r"mul_mat_q", "矩阵乘（量化，多行）"),
    (r"mul_mat_vec_q", "矩阵乘（量化，单行）"),
    (r"flash_attn", "注意力（flash）"),
    (r"soft_max", "注意力（非 flash）：softmax"),
    (r"cutlass|gemm|gemv|cublas|k_compute_batched_ptrs|mul_mat_vec_f", "注意力（非 flash）：cuBLAS 矩阵乘"),
    (r"rms_norm|norm_f32", "RMSNorm"),
    (r"rope", "RoPE"),
    (r"silu|gelu|unary_gated|swiglu", "SwiGLU"),
    (r"bin_bcast|op_add|op_mul|scale|k_add|k_mul", "逐元素（加偏置、残差）"),
    (r"get_rows", "取行"),
    (r"set_rows|cpy|dup|concat|convert_unary", "拷贝/类型转换/写KV缓存"),
]


def classify(name):
    for pat, cat in RULES:
        if re.search(pat, name):
            return cat
    return "其他"


def short(name):
    """去掉模板参数和形参表，只留函数名：void mul_mat_q<(ggml_type)8,...>(...) → mul_mat_q"""
    name = re.sub(r"^(void|static|inline)\s+", "", name)
    base = re.split(r"[<(]", name, maxsplit=1)[0].strip() or name
    m = re.search(r"<&(op_\w+|cpy_\w+)|<(cutlass_\w+)|<(float, __half|__half, float)>", name)
    return base + ("[" + next(g for g in m.groups() if g) + "]" if m else "")


def load(path):
    db = sqlite3.connect(path)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        sys.exit("这个文件里没有 CUPTI_ACTIVITY_KIND_KERNEL 表（没记到 GPU 内核）。现有的表：\n  "
                 + "\n  ".join(sorted(tables)))
    cols = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    name_col = "demangledName" if "demangledName" in cols else "shortName"
    kernels = db.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.%s ORDER BY k.start" % name_col).fetchall()
    memcpy = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        memcpy = db.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall()
    return kernels, memcpy


def split_passes(kernels, gap_ns):
    passes, cur = [], [kernels[0]]
    for prev, k in zip(kernels, kernels[1:]):
        if k[0] - prev[1] > gap_ns:
            passes.append(cur); cur = []
        cur.append(k)
    passes.append(cur)
    return passes


def auto_gap(kernels):
    """空隙从大到小排，找相邻两个差得最多的地方当分界：分界以上是"两趟之间"，以下是"一趟之内"。"""
    gaps = sorted((k[0] - p[1] for p, k in zip(kernels, kernels[1:])), reverse=True)
    gaps = [g for g in gaps[:200] if g > 0]
    if len(gaps) < 2:
        return 10**12
    best = max(range(len(gaps) - 1), key=lambda i: gaps[i] / gaps[i + 1])
    return (gaps[best] + gaps[best + 1]) // 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--gap-ms", type=float, default=2.0, help="切段用的空隙（ms），默认 2")
    ap.add_argument("--every", type=int, default=1, help="每几趟为一组")
    ap.add_argument("--offset", type=int, default=0, help="取每组里的第几趟（从 0 数）")
    ap.add_argument("--top", type=int, default=15, help="按内核名列出前几名")
    a = ap.parse_args()

    kernels, memcpy = load(a.sqlite)
    if not kernels:
        sys.exit("没有记到任何 GPU 内核")
    gap_ns = int(a.gap_ms * 1e6)
    passes = split_passes(kernels, gap_ns)

    print("## 切段（空隙阈值 %.3f ms，共 %d 个内核，%d 趟）\n" % (gap_ns / 1e6, len(kernels), len(passes)))
    print("| 趟 | 内核个数 | 墙上 ms | GPU 忙碌 ms | 空闲占比 |")
    print("|---:|---:|---:|---:|---:|")
    stats = []
    for i, p in enumerate(passes):
        wall = (p[-1][1] - p[0][0]) / 1e6
        busy = sum(e - s for s, e, _ in p) / 1e6
        stats.append((len(p), wall, busy))
    big_n = max(s[0] for s in stats) // 2
    small = [s for s in stats if s[0] < big_n]
    for i, (n, wall, busy) in enumerate(stats):
        if n >= big_n:
            print("| %d | %d | %.2f | %.2f | %.0f%% |" % (i, n, wall, busy, 100 * (1 - busy / wall) if wall else 0))
    print("\n另有 %d 段零散内核（加载模型、清缓存等，共 %d 个内核，%.2f ms），不算前向。"
          % (len(small), sum(s[0] for s in small), sum(s[2] for s in small)))

    # 选哪几趟来汇总：完整的前向里内核个数最常见的那些，去掉其中第一趟（热身），再按 every/offset 抽
    bigs = [i for i, s in enumerate(stats) if s[0] >= big_n]
    modal = collections.Counter(stats[i][0] for i in bigs).most_common(1)[0][0]
    same = [i for i in bigs if stats[i][0] == modal]
    if len(same) > 2:
        same = same[1:]
    chosen = [i for j, i in enumerate(same) if j % a.every == a.offset]
    odd = [i for i in bigs if stats[i][0] != modal]
    if odd:
        print("\n注意：第 %s 趟的内核个数不是 %d，没算进汇总（多半是第一趟：还没用 CUDA graph，或图不一样）。" % (odd, modal))
    if not chosen:
        sys.exit("没有可汇总的趟")

    print("\n## 按类别（第 %s 趟，每趟 %d 个内核；时间取各趟中位数）\n" % (chosen, modal))
    per_cat = collections.defaultdict(lambda: [0, []])     # 类别 → [每趟个数, [每趟 ms]]
    per_name = collections.defaultdict(lambda: [0, []])
    for i in chosen:
        cat_ms, name_ms = collections.Counter(), collections.Counter()
        cat_n, name_n = collections.Counter(), collections.Counter()
        for s, e, name in passes[i]:
            sn = short(name); c = classify(sn)
            cat_ms[c] += (e - s) / 1e6; cat_n[c] += 1
            name_ms[sn] += (e - s) / 1e6; name_n[sn] += 1
        for c in cat_ms:
            per_cat[c][0] = cat_n[c]; per_cat[c][1].append(cat_ms[c])
        for n in name_ms:
            per_name[n][0] = name_n[n]; per_name[n][1].append(name_ms[n])
    busy_med = statistics.median(stats[i][2] for i in chosen)
    wall_med = statistics.median(stats[i][1] for i in chosen)

    def table(d, head, limit=None):
        print("| %s | 每趟个数 | 每趟 ms | 平均每个 µs | 占 GPU 忙碌 |" % head)
        print("|---|---:|---:|---:|---:|")
        items = sorted(d.items(), key=lambda kv: -statistics.median(kv[1][1]))
        for name, (cnt, ms) in items[:limit]:
            m = statistics.median(ms)
            print("| %s | %d | %.3f | %.1f | %.1f%% |" % (name, cnt, m, 1e3 * m / cnt, 100 * m / busy_med))

    table(per_cat, "类别")
    print("| **GPU 忙碌合计** | %d | **%.3f** | | 100%% |" % (modal, busy_med))
    print("| 墙上时间 | | %.3f | | 空闲 %.0f%% |" % (wall_med, 100 * (1 - busy_med / wall_med)))

    print("\n## 按内核名（前 %d 名）\n" % a.top)
    table(per_name, "内核", a.top)
    others = sorted(n for n in per_name if classify(n) == "其他")
    if others:
        print("\n没归类的内核名（补进 RULES）：" + "、".join(others))

    if memcpy:
        lo, hi = passes[chosen[0]][0][0], passes[chosen[0]][-1][1]
        inside = [(e - s, b) for s, e, b in memcpy if lo <= s <= hi]
        print("\n第 %d 趟期间的显存拷贝：%d 次，共 %.1f KB，%.3f ms"
              % (chosen[0], len(inside), sum(b for _, b in inside) / 1024, sum(t for t, _ in inside) / 1e6))


if __name__ == "__main__":
    main()

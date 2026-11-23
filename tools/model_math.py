"""估算 Qwen2.5-0.5B（Q8_0）读一段提示词时，每类算子至少要花多少时间。

这是纸面估算，不是实测：
  - 每个算子的计算量（FLOP）和要读写的字节数，按模型结构和数据类型算出来；
  - 硬件峰值按规格推算（默认值见下面 HW），板子上核对后用命令行参数改；
  - 一个算子的"下限时间" = max(计算量 / 算力峰值, 字节数 / 带宽峰值)，即 roofline 模型。
实际时间一定比下限长。拿它和 nsys 实测表对照，差得多的地方就是优化空间。

用法：
  python3 tools/model_math.py --new 145 --cached 133
  python3 tools/model_math.py --new 278 --cached 0 --bw-gbs 80   # 用实测带宽代替规格值
"""
import argparse

# Qwen2.5-0.5B-Instruct 的 config.json
CFG = dict(d=896, ffn=4864, layers=24, heads=14, kv_heads=2, head_dim=64, vocab=151936)

# 数据类型的字节数（每个元素）
Q8_0 = 34 / 32     # 权重：每 32 个数存 32 个 int8 + 1 个 fp16 缩放
Q8_1 = 36 / 32     # llama.cpp 做矩阵乘前把激活量化成 q8_1：32 个 int8 + fp16 缩放 + fp16 和
F32, F16 = 4, 2

# Orin Nano 8GB、MAXN_SUPER 的规格推算值（估算，待在板子上核对）
HW = dict(
    sms=8,              # 1024 个 CUDA 核 / 每个 SM 128 个
    clock_mhz=1020,     # MAXN_SUPER 下 GPU 最高频率
    dp4a_mac_per_sm=512,    # CUDA 核的 dp4a 指令：一条指令做 4 次 int8 乘加，每 SM 128 个核
    int8_mac_per_sm=2048,   # Ampere（GA10B）张量核：每 SM 每周期 2048 次 int8 乘加（稠密）
    f16_mac_per_sm=1024,    # 同上，fp16 乘加
    f32_fma_per_sm=128,     # CUDA 核：每 SM 每周期 128 次 fp32 乘加
    bw_gbs=102.4,       # LPDDR5 128 bit × 6400 MT/s
)


def ops_for_forward(n, cached, c=CFG, mm_unit="dp4a"):
    """返回 [(段, 类别, 名字, flop, 字节, 用哪种算力)]，一次前向（读 n 个新 token）。

    mm_unit：矩阵乘用哪种硬件单元算。"dp4a" = CUDA 核上的 int8 点积指令（llama.cpp 的量化矩阵乘
    走的是这条路，待在板子上读源码核对）；"int8" = 张量核（自己写算子时可以争取的上限）。
    """
    d, f, H, hd = c["d"], c["ffn"], c["heads"], c["head_dim"]
    kv = c["kv_heads"] * hd
    out = []

    def add(seg, cat, name, flop, nbytes, unit):
        out.append((seg, cat, name, flop, nbytes, unit))

    def matmul(seg, name, rows, k, m, bias=False):
        """rows 个 token 乘一个 k×m 的 Q8_0 权重。先把激活量化成 q8_1，再算 int8 矩阵乘。"""
        add(seg, "激活量化", name + ".quant", 0, rows * k * (F32 + Q8_1), "f32")
        add(seg, "矩阵乘", name, 2 * rows * k * m,
            k * m * Q8_0 + rows * k * Q8_1 + rows * m * F32, mm_unit)
        if bias:
            add(seg, "逐元素", name + ".bias", rows * m, rows * m * F32 * 3, "f32")

    def norm(seg, name, rows):
        add(seg, "RMSNorm", name, 4 * rows * d, rows * d * F32 * 2 + d * F32, "f32")

    def resid(seg, name, rows):
        add(seg, "逐元素", name, rows * d, rows * d * F32 * 3, "f32")

    # 输入：按 token 编号从词表里取行（Q8_0 反量化成 f32）
    add("输入", "取行", "embedding", 0, n * d * (Q8_0 + F32), "f32")

    for il in range(c["layers"]):
        seg = "层%02d" % il
        last = il == c["layers"] - 1
        norm(seg, "attn_norm", n)
        matmul(seg, "q_proj", n, d, d, bias=True)
        matmul(seg, "k_proj", n, d, kv, bias=True)
        matmul(seg, "v_proj", n, d, kv, bias=True)
        add(seg, "RoPE", "rope_q", 0, n * d * F32 * 2, "f32")
        add(seg, "RoPE", "rope_k", 0, n * kv * F32 * 2, "f32")
        add(seg, "写KV缓存", "store_kv", 0, 2 * n * kv * (F32 + F16), "f32")
        # 注意力：第 i 个新 token 看 cached + i + 1 个位置（因果掩码）。只算有用的部分。
        pairs = n * cached + n * (n + 1) // 2
        add(seg, "注意力", "attention", 2 * 2 * H * hd * pairs,
            n * d * F32 * 2 + 2 * (cached + n) * kv * F16, "f16")
        matmul(seg, "o_proj", n, d, d)
        # 最后一层：llama.cpp 只给最后一个 token 算输出，o_proj 之后就只留 1 行
        r = 1 if last else n
        resid(seg, "attn_resid", r)
        norm(seg, "ffn_norm", r)
        matmul(seg, "gate", r, d, f)
        matmul(seg, "up", r, d, f)
        add(seg, "逐元素", "swiglu", 3 * r * f, r * f * F32 * 3, "f32")
        matmul(seg, "down", r, f, d)
        resid(seg, "ffn_resid", r)

    # 输出：最后一个 token 过 final norm，再乘词表矩阵（Qwen2.5 和 embedding 共用这张表）
    norm("输出", "final_norm", 1)
    matmul("输出", "lm_head", 1, d, c["vocab"])
    return out


def peaks(hw):
    clk = hw["clock_mhz"] * 1e6
    return {
        "dp4a": hw["sms"] * hw["dp4a_mac_per_sm"] * 2 * clk,
        "int8": hw["sms"] * hw["int8_mac_per_sm"] * 2 * clk,
        "f16": hw["sms"] * hw["f16_mac_per_sm"] * 2 * clk,
        "f32": hw["sms"] * hw["f32_fma_per_sm"] * 2 * clk,
    }, hw["bw_gbs"] * 1e9


def lower_bound_ms(flop, nbytes, unit, pk, bw):
    t_c = flop / pk[unit]
    t_m = nbytes / bw
    return max(t_c, t_m) * 1e3, ("算力" if t_c > t_m else "带宽")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=int, default=145, help="要算的新 token 数")
    ap.add_argument("--cached", type=int, default=133, help="前缀缓存里已有的 token 数")
    ap.add_argument("--matmul", choices=["dp4a", "int8"], default="dp4a",
                    help="矩阵乘走哪条路：dp4a = CUDA 核 int8 点积（默认）；int8 = 张量核")
    for k, v in HW.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    hw = {k: getattr(a, k) for k in HW}
    pk, bw = peaks(hw)
    ops = ops_for_forward(a.new, a.cached, mm_unit=a.matmul)

    print("## 估算：读 %d 个新 token（缓存里已有 %d 个），矩阵乘走 %s\n"
          % (a.new, a.cached, {"dp4a": "CUDA 核 dp4a", "int8": "张量核 int8"}[a.matmul]))
    print("硬件峰值（按规格推算）：dp4a %.1f TOPS，int8 张量核 %.1f TOPS，fp16 张量核 %.1f TFLOPS，"
          "fp32 %.2f TFLOPS，带宽 %.1f GB/s\n"
          % (pk["dp4a"] / 1e12, pk["int8"] / 1e12, pk["f16"] / 1e12, pk["f32"] / 1e12, bw / 1e9))

    cats = {}
    for seg, cat, name, flop, nb, unit in ops:
        t, _ = lower_bound_ms(flop, nb, unit, pk, bw)
        s = cats.setdefault(cat, [0, 0.0, 0.0, 0.0, 0.0])
        s[0] += 1; s[1] += flop; s[2] += nb; s[3] += t
        s[4] += flop / pk[unit] * 1e3
    total_t = sum(s[3] for s in cats.values())
    print("| 类别 | 个数 | GFLOP | 读写 MB | 下限 ms | 其中纯计算 ms | 占下限 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for cat, s in sorted(cats.items(), key=lambda kv: -kv[1][3]):
        print("| %s | %d | %.2f | %.1f | %.2f | %.2f | %.0f%% |"
              % (cat, s[0], s[1] / 1e9, s[2] / 1e6, s[3], s[4], 100 * s[3] / total_t))
    tf = sum(s[1] for s in cats.values()); tb = sum(s[2] for s in cats.values())
    print("| **合计** | %d | %.2f | %.1f | **%.2f** | | |" % (len(ops), tf / 1e9, tb / 1e6, total_t))

    # 单独列出几个大头，方便和实测对照
    print("\n大头（单个算子）：")
    for want in [("层00", "gate"), ("层00", "q_proj"), ("层00", "attention"), ("层23", "gate"), ("输出", "lm_head")]:
        for seg, cat, name, flop, nb, unit in ops:
            if (seg, name) == want:
                t, by = lower_bound_ms(flop, nb, unit, pk, bw)
                print("  %s %-9s %.3f GFLOP %6.1f MB  下限 %.3f ms（受%s限制）" % (seg, name, flop / 1e9, nb / 1e6, t, by))
    layer_t = sum(lower_bound_ms(o[3], o[4], o[5], pk, bw)[0] for o in ops if o[0] == "层00")
    print("  一层（层00）合计下限 %.3f ms；共 %d 个算子" % (layer_t, sum(1 for o in ops if o[0] == "层00")))


if __name__ == "__main__":
    main()

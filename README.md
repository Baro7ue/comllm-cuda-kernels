# comllm-cuda-kernels

[comllm-edge-scheduler](https://github.com/Baro7ue/comllm-edge-scheduler) 的拓展内容，一份学习笔记：
从零开始，在 Jetson Orin Nano 8GB 上手写 CUDA 算子，弄清一个 LoRA 微调过的 Qwen2.5-0.5B
（GGUF Q8_0，llama.cpp CUDA 后端）单次决策推理的时间花在哪里，以及能不能写得更快。

这是学习过程的记录，不是成品。**目前没有任何一版比 llama.cpp 快，也没有接进真实推理。**

## 现在到哪了

学习顺序：先在板上量出各算子的耗时 → 先写最耗时的算子，先求正确 → 和 llama.cpp 在同一输入上比 → 再优化。

| 步骤 | 状态 |
|---|---|
| 1. 量出各算子的耗时分布 | 完成。矩阵乘一类约占 GPU 时间的 78%，前馈层的 gate、up 两个内核占 36% |
| 2. 手写矩阵乘（float，先求正确） | 进行中，见下表 |
| 3. 换成 8 位整数，和 llama.cpp 同输入对比 | 未开始 |
| 4. 优化（融合、减少拷贝） | 未开始 |

矩阵乘练习，尺寸和模型的 gate 一样（145 个 token，896 → 4864，全 float）。
同一次板上运行的结果【实测】，出处 `results/2026-11-21/matmul_v4a_proto.txt`：

| 版本 | 做法 | 中位数 |
|---|---|---:|
| 练习 1 | 每个线程算 Y 的一格 | 80.3 ms |
| 练习 2 | 每个线程算 Y 的同一列、16 行 | 11.1 ms |
| 练习 3 | 练习 1 的分工 + 共享内存（朴素写法） | 13.7 ms |
| 练习 4a | 练习 2 + 把 X 搬进共享内存 | 参考版 8.2 ms，自己的还没写 |
| 参考 v4 | 练习 2 + X、W 都进共享内存 | 3.5 ms |
| llama.cpp 的 gate | 8 位整数 + 张量核，不是同一种运算，只能当量级参考 | 0.33 ms |

所有版本都和 CPU 上用 double 算的结果逐元素对比，最大误差 ÷ S 为 2.75×10⁻⁷（容许上界 5.3×10⁻⁵）。

## 怎么读

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | [docs/day01.md](docs/day01.md) | 第 1 天：模型里有哪些算子，怎么估算、怎么量 |
| 2 | [docs/day01_results.md](docs/day01_results.md) | 第 1 天的实测结果，估算哪里对、哪里错 |
| 3 | [docs/day02.md](docs/day02.md) | 矩阵乘练习 1–4：原理、写错过的地方、每一版的结果 |
| 4 | [docs/day02_review.html](docs/day02_review.html) | 练习 1–3 的图解复习页（下载后用浏览器打开） |

`docs/day02.md` 第 8.4 节记了写的时候出过的错，例如"加法得到地址，方括号才取出数"
"小表只认块内序号"。哪几处是看了答案才改对的，也如实写在里面。

## 目录

| 路径 | 内容 | 在哪跑 |
|---|---|---|
| `cuda/00_hello/` | 最小 CUDA 程序：向量加法，顺带量内存带宽 | 板子 |
| `cuda/01_matmul/matmul.cu` | 矩阵乘的测试框架：造数据、CPU 参考、逐元素对比、计时 | 板子 |
| `cuda/01_matmul/my_kernels.cuh`、`my_v3.cuh`、`my_v4.cuh` | 自己写的练习 1–2、3、4 | 板子 |
| `cuda/01_matmul/ref_kernels.cuh` | 参考答案 | 板子 |
| `tools/model_math.py` | 按模型结构估算每类算子的时间下限 | Mac 或板子 |
| `scripts/env_check.sh` | 核对系统、CUDA、llama.cpp、模型、频率 | 板子 |
| `scripts/bench_baseline.sh` + `tools/bench_summary.py` | 用 llama-bench 复现基线 | 板子 |
| `scripts/profile_nsys.sh` + `tools/nsys_summary.py` | 用 nsys 按算子类别计时 | 板子 |
| `scripts/check_mmq_mma.sh` | 查 llama.cpp 的矩阵乘用的是张量核还是 dp4a | 板子 |
| `scripts/board.sh` | 把仓库传到板子、编译、运行矩阵乘练习 | Mac |
| `results/<日期>/` | 板子上的原始输出和汇总表 | — |

## 怎么跑

需要一块装了 CUDA 的 Jetson（这里是 Orin Nano 8GB，JetPack 7.2.1，CUDA 13.2），
并且在 `~/.ssh/config` 里给它起名 `Jetson`。

```bash
bash scripts/board.sh              # 默认大小：145 个 token，896 → 4864
```

```bash
bash scripts/board.sh 145 896 5000 # 换个大小（N K M）
```

脚本里板子上的目录名是 `jetson-llm-kernels`，那是这个项目早先的名字。

## 约定

- 每个数字标明【实测】【估算】【推测】，实测写出处。
- 每个算子都和参考实现逐元素对比；没量过就不说"无损""等价"。
- 分清计划、已执行、已核对。估错了的地方保留原文，在旁边加更正。
- 提交说明用英文，`type(scope): summary`；分支 `comllm-cuda-kernels-stageN-主题-vK`。

## 关于这些文件

- 练习（`my_*.cuh`）是我自己写的。测试框架、参考答案、脚本和讲解文档由 Claude（AI）协助编写。
- 基线数字来自 comllm-edge-scheduler 的 `evidence/G2/latency_table.md`。
- `results/` 里的原始输出做过一处替换：板子上的用户名和家目录路径换成了 `user` 和 `~`，其余原样。
- nsys 的原始记录（`.nsys-rep`、`.sqlite`，共约 11 MB）没有上传，只保留了汇总表。

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。

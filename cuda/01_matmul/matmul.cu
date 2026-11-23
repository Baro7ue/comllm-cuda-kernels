// 第一个真正的算子：矩阵乘  Y = X · Wᵀ
//
//   X：N 行 × K 列，每行是一个 token（N = 145，K = 896）
//   W：M 行 × K 列，每行是"算出一个输出要用的 K 个权重"（M = 4864）
//   Y：N 行 × M 列，Y[n][m] = X[n][0]*W[m][0] + X[n][1]*W[m][1] + ... + X[n][K-1]*W[m][K-1]
//
// 大小和模型里的 gate 一样（896 → 4864）。这一版全用 float（32 位浮点），还没有量化。
//
// 这个文件是测试框架：造数据、在 CPU 上算准确值、逐元素对比、计时。内核不在这里：
//   my_kernels.cuh   你写的内核
//   ref_kernels.cuh  参考答案
// 每写一版新内核，在下面的 VERSIONS 表里加一行，就能和前面的版本并排比较。
//
// 运行（在 Mac 上）：bash scripts/board.sh
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(err_));                               \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

#include "ref_kernels.cuh"   // 参考答案
#include "my_kernels.cuh"    // 你写的内核
#include "my_v3.cuh"         // 你写的练习 3
#include "my_v4.cuh"         // 你写的练习 4

// ============================================================================
// 版本表：以后的新版本加在这里
// ============================================================================
struct Version {
    const char* name;
    void (*run)(const float* X, const float* W, float* Y, int N, int K, int M);
};
static const Version VERSIONS[] = {
    {"参考 v1：每线程一个输出", matmul_v1},
    {"练习 1（你写的）", my_v1},
    {"参考 v2：每线程 16 个 token", matmul_v2<16>},
    {"练习 2（你写的）", my_v2},
    {"参考 v2b：v2 + 固定循环次数", matmul_v2b<16>},
    {"参考 v3：共享内存，朴素", matmul_v3<16, false>},
    {"练习 3（你写的）", my_v3},
    {"参考 v3：共享内存，W 转置", matmul_v3<16, true>},
    {"参考 4a：v2 + X 进共享内存", matmul_v4a<16, 16, false>},
    {"练习 4a（你写的）", my_v4a},
    {"参考 v4：v2 + 共享内存，257", matmul_v4<16, 257>},
    {"参考 v4：v2 + 共享内存，256", matmul_v4<16, 256>},
};

// ============================================================================
// 测试框架
// ============================================================================

// 固定种子的伪随机数：每次运行、每台机器造出来的数据都一样，结果才能互相比。
static float next_uniform(uint32_t& state) {   // 返回 [-1, 1) 之间的数
    state = state * 1664525u + 1013904223u;
    return (float)(state >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 145;
    const int K = argc > 2 ? std::atoi(argv[2]) : 896;
    const int M = argc > 3 ? std::atoi(argv[3]) : 4864;
    const int reps = 20;
    std::printf("矩阵乘 Y = X·Wᵀ：N=%d 个 token，K=%d → M=%d\n", N, K, M);
    std::printf("计算量 %.3f GFLOP（乘和加各算一次）；权重 %.1f MB，输入 %.2f MB，输出 %.2f MB\n",
                2.0 * N * K * M / 1e9, (double)M * K * 4 / 1e6, (double)N * K * 4 / 1e6,
                (double)N * M * 4 / 1e6);

    // 1. 造数据。数值范围大致模仿真实情况：激活在 ±1 左右，权重在 ±0.05 左右。
    std::vector<float> hX((size_t)N * K), hW((size_t)M * K), hY((size_t)N * M);
    uint32_t seed = 20261115u;
    for (auto& v : hX) v = next_uniform(seed);
    for (auto& v : hW) v = 0.05f * next_uniform(seed);

    // 2. CPU 上算参考答案，用 double（64 位）累加，把它当"准确值"。
    //    同时算 S = 各项绝对值之和，后面用它定"允许的误差"。
    std::vector<double> ref((size_t)N * M), absum((size_t)N * M);
    for (int n = 0; n < N; ++n) {
        const float* x = &hX[(size_t)n * K];
        for (int m = 0; m < M; ++m) {
            const float* w = &hW[(size_t)m * K];
            double acc = 0.0, s = 0.0;
            for (int k = 0; k < K; ++k) {
                const double p = (double)x[k] * (double)w[k];
                acc += p;
                s += std::fabs(p);
            }
            ref[(size_t)n * M + m] = acc;
            absum[(size_t)n * M + m] = s;
        }
    }

    // 3. 数据拷到 GPU
    float *dX = nullptr, *dW = nullptr, *dY = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, hX.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW, hW.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, hY.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // 允许的误差：float 累加 K 项，每一步都可能舍入一次。数值分析给出的最坏情况上界是
    //   |算出来的 − 准确值| ≤ K × u × S，   u = 2⁻²⁴（float 的单位舍入误差），S = 各项绝对值之和
    // 超过这个界，就不是舍入能解释的，一定是程序写错了。
    const double u = std::ldexp(1.0, -24);
    const double bound_factor = K * u;

    std::printf("\n| 版本 | 正确性 | 最大误差 ÷ S（上界 %.1e） | 中位数 ms | 最快 ms | 最慢 ms | GFLOP/s |\n", bound_factor);
    std::printf("|---|---|---:|---:|---:|---:|---:|\n");
    bool all_ok = true;
    char hints[2048] = "";
    for (const Version& v : VERSIONS) {
        // 每一版开始前把输出清零：万一某个元素没被写到，会在对比时暴露出来
        CUDA_CHECK(cudaMemset(dY, 0, hY.size() * sizeof(float)));

        for (int i = 0; i < 3; ++i) v.run(dX, dW, dY, N, K, M);   // 热身
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> ms(reps);
        for (int r = 0; r < reps; ++r) {
            CUDA_CHECK(cudaEventRecord(t0));
            v.run(dX, dW, dY, N, K, M);
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            CUDA_CHECK(cudaEventElapsedTime(&ms[r], t0, t1));
        }
        CUDA_CHECK(cudaGetLastError());

        // 拷回来，逐元素和参考答案比
        CUDA_CHECK(cudaMemcpy(hY.data(), dY, hY.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double worst = 0.0;      // 最大的 误差 ÷ S
        size_t bad = 0;          // 超出上界的元素个数
        size_t first_bad = 0;    // 第一个超界的位置
        for (size_t i = 0; i < hY.size(); ++i) {
            const double rel = std::fabs((double)hY[i] - ref[i]) / absum[i];
            worst = std::max(worst, rel);
            if (!(rel <= bound_factor)) {        // 写成 !(<=) 是为了让 NaN 也算不通过
                if (bad == 0) first_bad = i;
                ++bad;
            }
        }
        all_ok = all_ok && bad == 0;

        std::sort(ms.begin(), ms.end());
        const double med = ms[reps / 2];
        char verdict[64];
        if (bad == 0) std::snprintf(verdict, sizeof verdict, "通过");
        else std::snprintf(verdict, sizeof verdict, "失败（%zu 个超界）", bad);
        char speed[32] = "—";   // 结果不对的版本，速度没有意义，不显示
        if (bad == 0) std::snprintf(speed, sizeof speed, "%.1f", 2.0 * N * K * M / 1e9 / (med * 1e-3));
        std::printf("| %s | %s | %.2e | %.3f | %.3f | %.3f | %s |\n", v.name, verdict, worst, med,
                    ms[0], ms[reps - 1], speed);
        if (bad != 0) {
            // 告诉你错在哪：第一个出错的元素，以及它前后各是什么情况，方便找规律
            std::snprintf(hints + std::strlen(hints), sizeof hints - std::strlen(hints),
                          "  %s：第一个出错的是 Y[n=%zu][m=%zu]，你的值 %.6g，准确值 %.6g；共 %zu / %zu 个出错\n",
                          v.name, first_bad / M, first_bad % M, (double)hY[first_bad], ref[first_bad], bad,
                          hY.size());
        }
    }

    if (hints[0]) std::printf("\n出错的版本：\n%s", hints);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dY));
    return all_ok ? 0 : 1;
}

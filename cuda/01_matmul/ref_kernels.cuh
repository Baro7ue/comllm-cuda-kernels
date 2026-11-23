// 参考答案。先自己在 my_kernels.cuh 里写，卡住了再来看。
//
// 每一版由两部分组成：
//   xxx_kernel  在 GPU 上跑的内核
//   xxx         在 CPU 上调用的"发起函数"：决定开多少块、每块多少线程，然后发起内核
#pragma once
#include <cuda_runtime.h>

// ============================================================================
// 第 1 版：最直接的写法——每个线程负责算 Y 里的一个数
// ============================================================================
//
// Y 一共 N×M = 145×4864 ≈ 70 万个数，那就开 70 万个线程，一人算一个。
// 线程怎么知道自己算哪个？
//   blockIdx.y                          → 我在第几行 = 第 n 个 token
//   blockIdx.x * blockDim.x + threadIdx.x → 我在这一行的第几个 = 第 m 个输出
// 然后老老实实循环 K 次，把 X 的第 n 行和 W 的第 m 行对应相乘、累加。
__global__ void matmul_v1_kernel(const float* __restrict__ X, const float* __restrict__ W,
                                 float* __restrict__ Y, int N, int K, int M) {
    const int n = blockIdx.y;
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;   // 最后一个块可能多出几个线程，没有活干，直接返回

    const float* x = X + n * K;   // X 的第 n 行从这里开始
    const float* w = W + m * K;   // W 的第 m 行从这里开始
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += x[k] * w[k];
    }
    Y[n * M + m] = acc;
}

static void matmul_v1(const float* X, const float* W, float* Y, int N, int K, int M) {
    const int threads = 256;                       // 每块 256 个线程
    const dim3 grid((M + threads - 1) / threads, N);   // 横向：够盖住 M 个输出；纵向：N 行
    matmul_v1_kernel<<<grid, threads>>>(X, W, Y, N, K, M);
}

// ============================================================================
// 第 2 版：一个线程算同一个 m 的 T 个 token——每读一个权重，连用 T 次
// ============================================================================
//
// 第 1 版里，W 的第 m 行被 N 个线程各读一遍。这一版让一个线程负责"第 m 个输出、连续 T 个 token"：
// 每读进一个权重 w，就和 T 个 token 的输入各乘一次，分别加进 T 个累加器。
// W 的每一行被读的遍数从 N 降到 N/T（向上取整）。
//
// T 是编译时就定下来的常数（模板参数），这样编译器能把 T 个累加器都放进寄存器。
template <int T>
__global__ void matmul_v2_kernel(const float* __restrict__ X, const float* __restrict__ W,
                                 float* __restrict__ Y, int N, int K, int M) {
    const int n0 = blockIdx.y * T;                        // 我负责的第一个 token
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    const int cnt = (N - n0 < T) ? (N - n0) : T;          // 最后一组可能不满 T 个

    const float* w = W + m * K;
    float acc[T];
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float wk = w[k];                            // 权重只读一次
        for (int t = 0; t < cnt; ++t) {
            acc[t] += X[(n0 + t) * K + k] * wk;           // 给 cnt 个 token 各用一次
        }
    }
    for (int t = 0; t < cnt; ++t) Y[(n0 + t) * M + m] = acc[t];
}

template <int T>
static void matmul_v2(const float* X, const float* W, float* Y, int N, int K, int M) {
    const int threads = 256;
    const dim3 grid((M + threads - 1) / threads, (N + T - 1) / T);   // 纵向：每 T 个 token 一排
    matmul_v2_kernel<T><<<grid, threads>>>(X, W, Y, N, K, M);
}

// ============================================================================
// 第 2b 版：和第 2 版一样，只是内层循环次数固定为 T
// ============================================================================
//
// 第 2 版的内层循环跑 cnt 次，cnt 要到运行时才知道。这一版固定跑 T 次：
// 不存在的 token 改读最后一行（算出来的值不写回），编译器就能把循环完全展开。
template <int T>
__global__ void matmul_v2b_kernel(const float* __restrict__ X, const float* __restrict__ W,
                                  float* __restrict__ Y, int N, int K, int M) {
    const int n0 = blockIdx.y * T;
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    const float* w = W + m * K;
    const float* xr[T];
    for (int t = 0; t < T; ++t) xr[t] = X + (n0 + t < N ? n0 + t : N - 1) * K;
    float acc[T];
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float wk = w[k];
        for (int t = 0; t < T; ++t) acc[t] += xr[t][k] * wk;
    }
    for (int t = 0; t < T; ++t)
        if (n0 + t < N) Y[(n0 + t) * M + m] = acc[t];
}
template <int T>
static void matmul_v2b(const float* X, const float* W, float* Y, int N, int K, int M) {
    const dim3 grid((M + 255) / 256, (N + T - 1) / T);
    matmul_v2b_kernel<T><<<grid, 256>>>(X, W, Y, N, K, M);
}

// ============================================================================
// 第 3 版：共享内存分块（最朴素的写法）
// ============================================================================
//
// 一个块 = 16×16 个线程，合起来算 Y 的一个 16×16 小块（16 个 token × 16 个输出），每个线程一个输出。
// K 个特征分成每 16 个一段。对每一段：
//   1. 256 个线程每人从内存搬一个 X 的数、一个 W 的数，放进共享内存（X 和 W 各 16×16 个数）
//   2. 等所有人搬完（__syncthreads）
//   3. 每个线程从共享内存里取 16 对数，乘起来加进自己的累加器
//   4. 等所有人算完（__syncthreads），才能搬下一段
// 超出范围的线程不能提前 return：别的线程还要用它搬的数。
template <int TS, bool TRANSPOSE_W>
__global__ void matmul_v3_kernel(const float* __restrict__ X, const float* __restrict__ W,
                                 float* __restrict__ Y, int N, int K, int M) {
    __shared__ float sX[TS][TS];   // sX[块内第几个 token][段内第几个特征]
    __shared__ float sW[TS][TS];   // 朴素：sW[块内第几个输出][段内第几个特征]；转置：两个下标对调
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int m0 = blockIdx.x * TS, n0 = blockIdx.y * TS;
    const int m = m0 + tx, n = n0 + ty;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        const int kx = k0 + tx;
        sX[ty][tx] = (n < N && kx < K) ? X[n * K + kx] : 0.0f;
        const float wv = (m0 + ty < M && kx < K) ? W[(m0 + ty) * K + kx] : 0.0f;
        if (TRANSPOSE_W) sW[tx][ty] = wv; else sW[ty][tx] = wv;
        __syncthreads();
        for (int kk = 0; kk < TS; ++kk)
            acc += sX[ty][kk] * (TRANSPOSE_W ? sW[kk][tx] : sW[tx][kk]);
        __syncthreads();
    }
    if (n < N && m < M) Y[n * M + m] = acc;
}
template <int TS, bool TRANSPOSE_W>
static void matmul_v3(const float* X, const float* W, float* Y, int N, int K, int M) {
    const dim3 grid((M + TS - 1) / TS, (N + TS - 1) / TS);
    matmul_v3_kernel<TS, TRANSPOSE_W><<<grid, dim3(TS, TS)>>>(X, W, Y, N, K, M);
}

// ============================================================================
// 第 4a 版：第 2 版 + 只把 X 搬进共享内存（W 仍直接从内存读）
// ============================================================================
//
// 线程排法和第 2 版完全一样：一块 256 个线程，threadIdx.x 管 Y 的列（输出 m）；
// blockIdx.y 管 Y 的哪 T 行（token n0 到 n0+T-1）。一个块盖住 Y 的 T 行 × 256 列。
// 同一个块的 256 个线程用的是同样的 T 行 X，所以每段 TK 个特征：
//   全块合作把 X 的 T 行 × TK 列搬进 sX（T×TK 个数，256 个线程分着搬）
//   → 等 → 每个线程从 sX 里取数，W 照旧从内存读 → 等
// 和第 2 版的两处不同：越界线程不能提前 return（要帮着搬、要到集合点）；X 改从 sX 里取。
// FIXED：内层循环固定跑 T 次（不存在的 token 在 sX 里是 0），还是跑 cnt 次。
template <int T, int TK, bool FIXED>
__global__ void matmul_v4a_kernel(const float* __restrict__ X, const float* __restrict__ W,
                                  float* __restrict__ Y, int N, int K, int M) {
    __shared__ float sX[T][TK];                     // sX[块内第几个 token][段内第几个特征]
    const int tid = threadIdx.x;
    const int m = blockIdx.x * 256 + tid;
    const int n0 = blockIdx.y * T;
    const int cnt = (N - n0 < T) ? (N - n0) : T;
    float acc[T];
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i = tid; i < T * TK; i += 256) {   // T*TK = 256 时每人正好搬一个
            const int t = i / TK, kk = i % TK;
            sX[t][kk] = (n0 + t < N && k0 + kk < K) ? X[(n0 + t) * K + k0 + kk] : 0.0f;
        }
        __syncthreads();
        if (m < M) {
            for (int kk = 0; kk < TK; ++kk) {
                const float wk = (k0 + kk < K) ? W[m * K + k0 + kk] : 0.0f;
                if (FIXED) {
                    for (int t = 0; t < T; ++t) acc[t] += sX[t][kk] * wk;
                } else {
                    for (int t = 0; t < cnt; ++t) acc[t] += sX[t][kk] * wk;
                }
            }
        }
        __syncthreads();
    }
    if (m < M)
        for (int t = 0; t < cnt; ++t) Y[(n0 + t) * M + m] = acc[t];
}
template <int T, int TK, bool FIXED>
static void matmul_v4a(const float* X, const float* W, float* Y, int N, int K, int M) {
    const dim3 grid((M + 255) / 256, (N + T - 1) / T);
    matmul_v4a_kernel<T, TK, FIXED><<<grid, 256>>>(X, W, Y, N, K, M);
}

// ============================================================================
// 第 4 版：第 2 版 + 共享内存
// ============================================================================
//
// 一个块 256 个线程。每个线程负责一个输出 m 的 TN 个 token（TN 个累加器，同第 2 版）。
// 每段 32 个特征：把 X 的 TN×32 个数和 W 的 256×32 个数搬进共享内存，再从共享内存里算。
// 搬 W 时，一组 32 个线程读的是 W 同一行里连续的 32 个数。
// STRIDE：sW 里每个特征占多少格。取 257 而不是 256，是为了写入时一组线程不挤在同一个 bank。
template <int TN, int STRIDE>
__global__ void matmul_v4_kernel(const float* __restrict__ X, const float* __restrict__ W,
                                 float* __restrict__ Y, int N, int K, int M) {
    const int TK = 32, TM = 256;
    __shared__ float sX[TN * TK];
    __shared__ float sW[TK * STRIDE];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int m0 = blockIdx.x * TM, m = m0 + tid, n0 = blockIdx.y * TN;
    float acc[TN];
    for (int t = 0; t < TN; ++t) acc[t] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i = tid; i < TN * TK; i += TM) {
            const int t = i / TK, kk = i % TK;
            sX[i] = (n0 + t < N && k0 + kk < K) ? X[(n0 + t) * K + k0 + kk] : 0.0f;
        }
        for (int r = warp; r < TM; r += 8) {
            sW[lane * STRIDE + r] = (m0 + r < M && k0 + lane < K) ? W[(m0 + r) * K + k0 + lane] : 0.0f;
        }
        __syncthreads();
        for (int kk = 0; kk < TK; ++kk) {
            const float wk = sW[kk * STRIDE + tid];
            for (int t = 0; t < TN; ++t) acc[t] += sX[t * TK + kk] * wk;
        }
        __syncthreads();
    }
    if (m < M)
        for (int t = 0; t < TN; ++t)
            if (n0 + t < N) Y[(n0 + t) * M + m] = acc[t];
}
template <int TN, int STRIDE>
static void matmul_v4(const float* X, const float* W, float* Y, int N, int K, int M) {
    const dim3 grid((M + 255) / 256, (N + TN - 1) / TN);
    matmul_v4_kernel<TN, STRIDE><<<grid, 256>>>(X, W, Y, N, K, M);
}

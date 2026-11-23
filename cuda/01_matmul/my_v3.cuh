// 练习 3：共享内存分块。你写的内核放在这个文件里（练习 1、2 在 my_kernels.cuh）。
//
// 运行（在 Mac 上）：bash scripts/board.sh
#pragma once
#include <cuda_runtime.h>

// ============================================================================
// 练习 3：一个线程块合作算 Y 的一个 16×16 小块
// ============================================================================
//
// 线程的排法：每个块是 16×16 = 256 个线程，排成二维。
//   threadIdx.x（0 到 15）  我是块里横向第几个  → 对应小块里的第几个输出
//   threadIdx.y（0 到 15）  我是块里纵向第几个  → 对应小块里的第几个 token
//   blockIdx.x              这个块是横向第几块  → 它负责的输出从 blockIdx.x * 16 开始
//   blockIdx.y              这个块是纵向第几块  → 它负责的 token 从 blockIdx.y * 16 开始
//
// 每个线程还是只算一个 Y[n][m]。不同的是：896 个特征分成 56 段，每段 16 个。
// 对每一段，全块的线程先一起把这一段要用的 X 和 W 搬进共享内存，再各自从共享内存里取数来算。
//
// 这个练习假定 K 是 16 的整数倍（896 = 56 × 16）。
const int TILE = 16;

__global__ void my_v3_kernel(const float* X, const float* W, float* Y, int N, int K, int M) {
    // 共享内存：整个块的 256 个线程共用这两张 16×16 的小表。这两行已经写好，不用改。
    __shared__ float sX[TILE][TILE];   // sX[块内第几个 token][段内第几个特征]
    __shared__ float sW[TILE][TILE];   // sW[块内第几个输出][段内第几个特征]

    // 第 1 步：算出 6 个数。
    //   tx, ty  我在块里的横向、纵向序号
    //   m0, n0  这个块负责的第一个输出、第一个 token
    //   m, n    我负责的输出、token
    // 在这里写：
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int m0 = TILE * blockIdx.x;
    int n0 = TILE * blockIdx.y;
    int m = m0 + tx;
    int n = n0 + ty;


    // 第 2 步：准备累加器，初值 0。
    // 在这里写：
    float acc = 0.0f;



    // 第 3 步：外层循环，k0 从 0 开始，每次加 TILE，直到 K。k0 是"这一段的第一个特征"。
    //   循环体里依次做 3a、3b、3c、3d 四件事。
    // 在这里写循环的壳，然后把下面四件事写进循环体：

    for (int k0 = 0; k0 < K; k0 += TILE)
    {   
        if (n < N)
        {
            sX[ty][tx] = X[n*K + (k0 + tx)];
        }
        else
        { 
            sX[ty][tx] = 0.0f;
        } 

        if (m0 + ty < M)
        {
            sW[ty][tx] = W[(m0+ty)*K + (k0 + tx)];
        }
        else
        {
            sW[ty][tx] = 0.0f;
        }
    //   3a. 搬数据。我（tx, ty）负责搬两个数：
    //         sX[ty][tx] ← X 里 token n 的第 (k0 + tx) 个特征
    //         sW[ty][tx] ← W 里输出 (m0 + ty) 的第 (k0 + tx) 个权重
    //       注意越界：n >= N 时没有这个 token，往 sX 里放 0；m0 + ty >= M 时往 sW 里放 0。
    //       注意：这里不能像练习 1、2 那样让越界的线程 return，因为别的线程要用它搬的数。

    __syncthreads();
    //   3b. __syncthreads();   等全块的线程都搬完。

    for (int kk = 0; kk < TILE ; ++kk)
    {
        acc += sX[ty][kk] * sW[tx][kk];
    }
    //   3c. 内层循环 kk 从 0 到 TILE-1：
    //         累加器 += sX 里"我的 token"那一行的第 kk 个 × sW 里"我的输出"那一行的第 kk 个

    __syncthreads();
    //   3d. __syncthreads();   等全块的线程都算完，才能进入下一段去覆盖共享内存。

    // 第 4 步：把累加器写到 Y[n][m]。只有 n 和 m 都在范围内才写。
    // 在这里写：
    }
        if (n < N && m < M)
    {
        Y[n*M + m] = acc;
    }
}

static void my_v3(const float* X, const float* W, float* Y, int N, int K, int M) {
    // 第 5 步：横向要几块、纵向要几块（都是每块管 TILE 个，向上取整）。
    //         每块的线程数是二维的：dim3(TILE, TILE)。这一行已经写好。
    // 改 grid 这一行：
    int blocks_x = M / TILE;
    if (M % TILE != 0)
    {
        blocks_x = M / TILE + 1;
    }

    int blocks_y = N / TILE;
    if (N % TILE != 0)
    {
        blocks_y = N / TILE + 1;
    }

    const dim3 grid(blocks_x, blocks_y);
    const dim3 block(TILE, TILE);

    my_v3_kernel<<<grid, block>>>(X, W, Y, N, K, M);
}

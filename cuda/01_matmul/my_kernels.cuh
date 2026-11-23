// 你写的内核都放在这个文件里。
//
// 怎么用：
//   1. 按下面的步骤，把标着"在这里写"的地方补上代码
//   2. 在 Mac 上运行：bash scripts/board.sh
//      它会把代码传到板子上、编译、运行，然后打印一张表
//   3. 表里"练习 1（你写的）"那行显示"通过"就对了；显示"失败"时，表下面会告诉你第一个错在哪
//
// 现在这个文件能编译，但内核是空的，所以运行结果会是"失败"——这是正常的起点。
#pragma once
#include <cuda_runtime.h>

// ============================================================================
// 练习 1：每个线程算 Y 里的一个数
// ============================================================================
//
// 要算的东西：
//   Y[n][m] = X[n][0]*W[m][0] + X[n][1]*W[m][1] + ... + X[n][K-1]*W[m][K-1]
//
// 三个数组在内存里都是"一行接一行"排成的一长条：
//   X 有 N 行、每行 K 个数   → 第 n 行第 k 个数是 X[n*K + k]
//   W 有 M 行、每行 K 个数   → 第 m 行第 k 个数是 W[m*K + k]
//   Y 有 N 行、每行 M 个数   → 第 n 行第 m 个数是 Y[n*M + m]
//
// 线程的排法（由下面的发起函数 my_v1 决定）：
//   纵向 N 排，每排对应一个 token；横向若干块，每块 256 个线程，合起来盖住 M 个输出。
//
// 内核里可以直接用的四个内置变量：
//   blockIdx.x   我所在的块，是横向第几块（从 0 数）
//   blockIdx.y   我所在的块，是纵向第几排（从 0 数）
//   blockDim.x   每块有多少个线程（这里是 256）
//   threadIdx.x  我是块里的第几个线程（0 到 255）
__global__ void my_v1_kernel(const float* X, const float* W, float* Y, int N, int K, int M) {
    // 第 1 步：算出"我负责第几个 token"（n）和"我负责它的第几个输出"（m）。
    //         n 只看纵向；m 要用横向的块号、每块线程数、块内序号三个数拼出来。
    // 在这里写：
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y;

    // 第 2 步：横向的线程总数可能比 M 多（比如 M=5000 时要开 20 块 = 5120 个线程）。
    //         多出来的线程没有活干，必须直接 return，否则会写到 Y 的范围外面。
    // 在这里写：
    if (m >= M)
    {
        return;
    }

    // 第 3 步：准备一个 float 变量当累加器，初值 0。
    //         循环 K 次，每次把 X 第 n 行的第 k 个数 和 W 第 m 行的第 k 个数 相乘，加进累加器。
    // 在这里写：
    const float* x = X + n * K;
    const float* w = W + m * K;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
    {
        acc += x[k] * w[k]; //acc += X[n*K + k] * W[m*K + k]
    }

    // 第 4 步：把累加器的值写到 Y 的第 n 行第 m 个位置。
    // 在这里写：
    Y[n * M + m] = acc;

}

// 发起函数：在 CPU 上运行，决定线程怎么排，然后发起内核。
static void my_v1(const float* X, const float* W, float* Y, int N, int K, int M) {
    const int threads = 256;   // 每块 256 个线程
    int blocks_x = M / threads;

    // 第 5 步：算出横向要几块、纵向要几排，填进 grid(横向块数, 纵向排数)。
    //         横向：块数 × 256 要 ≥ M。注意 M 不一定是 256 的整数倍，要向上取整。
    //         纵向：每个 token 一排。
    //         现在填的 (1, 1) 是占位用的，只开了 1 块，肯定不够。
    // 改这一行：
    if (M % threads != 0)
    {
        blocks_x = M / threads + 1;
    }       
    const dim3 grid(blocks_x , N);

    my_v1_kernel<<<grid, threads>>>(X, W, Y, N, K, M);
}

// ============================================================================
// 练习 2：每个线程算同一个 m 的 4 个 token —— 每读一个权重，连用 4 次
// ============================================================================
//
// 练习 1 里，W 的第 m 行被 145 个线程各读了一遍。
// 这一版让一个线程负责"第 m 个输出、连续 4 个 token"（n0, n0+1, n0+2, n0+3）：
//   每读进一个权重，就和 4 个 token 的输入各乘一次，分别加进 4 个累加器。
//
// 线程的排法变了：纵向不再是每个 token 一排，而是每 4 个 token 一排。
//   第 0 排负责 token 0,1,2,3；第 1 排负责 token 4,5,6,7；……
// 横向和练习 1 完全一样。
//
// 注意：145 不是 4 的整数倍。最后一排只有 1 个 token 是真的（token 144），
//       另外 3 个（145,146,147）不存在，既不能读 X 的那几行，也不能写 Y 的那几行。
const int TOKENS_PER_THREAD = 16;

__global__ void my_v2_kernel(const float* X, const float* W, float* Y, int N, int K, int M) {
    // 第 1 步：算出 m（和练习 1 一样）和 n0（我负责的第一个 token）。相当于我负责的第一个token的编号。
    //         n0 要用纵向排号和 TOKENS_PER_THREAD 算出来。
    // 在这里写：
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int n0 = TOKENS_PER_THREAD*blockIdx.y;

    // 第 2 步：m 超出范围的线程直接 return（和练习 1 一样）。
    // 在这里写：
    if (m >= M)
    {
        return;
    }

    // 第 3 步：算出 cnt = 我这一排实际有几个 token。
    //         一般是 4；但最后一排可能不满。从 n0 开始到最后还剩 N - n0 个 token，
    //         cnt 取"4"和"剩下的个数"里较小的那个。
    // 在这里写：
    const int cnt = min(TOKENS_PER_THREAD , N - n0);


    // 第 4 步：准备 4 个累加器（一个长度为 4 的 float 数组），全部清零。
    // 在这里写：
    float acc[TOKENS_PER_THREAD] = {0.0f};

    
    // 第 5 步：两层循环。
    //         外层 k 从 0 到 K-1：先把 W 第 m 行的第 k 个数读进一个局部变量（只读这一次）。
    //         内层 t 从 0 到 cnt-1：第 t 个 token 是 n0 + t，
    //             把 X 第 (n0+t) 行的第 k 个数 乘上刚才那个局部变量，加进第 t 个累加器。
    // 在这里写：
    const float* x = X + n0*K;
    const float* w = W + m*K;
    for (int i = 0; i < K ;++i)
    {
        for (int t = 0; t < cnt ;++t)
        {
        acc[t] += x[K*t+i]*w[i];
        }
    }


    // 第 6 步：把 cnt 个累加器分别写到 Y 的第 (n0+t) 行第 m 个位置。
    // 在这里写：
    for (int t = 0; t < cnt; ++t)
    {
        Y[(n0 + t)*M + m] = acc[t];
    }
}

static void my_v2(const float* X, const float* W, float* Y, int N, int K, int M) {
    const int threads = 256;   // 每块 256 个线程
    int blocks_x = M / threads;
    int blocks_y = N / TOKENS_PER_THREAD;
    // 第 7 步：横向块数和练习 1 一样；纵向排数 = N 个 token 每 4 个一排，向上取整。
    //         现在的 (1, 1) 是占位用的。
    // 改这一行：
    if (M % threads != 0)
    {
        blocks_x = M / threads + 1;
    }       

    if (N % TOKENS_PER_THREAD != 0)
    {
        blocks_y = N / TOKENS_PER_THREAD + 1;
    }       
    const dim3 grid(blocks_x, blocks_y);

    my_v2_kernel<<<grid, threads>>>(X, W, Y, N, K, M);
}

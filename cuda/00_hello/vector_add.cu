// 第一个 CUDA 程序：c[i] = a[i] + b[i]
//
// 目的：
//   1. 确认 nvcc 能编译、GPU 能跑；
//   2. 打印这块 GPU 的关键参数（SM 个数、频率、L2、共享内存……），后面写算子要用；
//   3. 量出这块板子上内核实际能跑到的内存带宽，替换 tools/model_math.py 里按规格推算的带宽。
//
// 编译运行：make && ../../build/vector_add
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

// 每个 CUDA 调用都检查返回值：出错时打印文件、行号和原因，然后退出。
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(err_));                               \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

// __global__：在 GPU 上运行、由 CPU 发起的函数，叫"内核"（kernel）。
// 发起时指定 gridDim 个线程块、每块 blockDim 个线程，每个线程都执行这段代码，
// 靠 blockIdx / threadIdx 区分自己是谁、该处理哪些元素。
//
// 这里用"网格步长循环"：线程总数可以少于元素数，每个线程隔 stride 个元素处理一个。
// 相邻线程访问相邻地址（i, i+1, ...），一个 warp（32 个线程）的读写能合并成整段的内存事务。
__global__ void vector_add(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ c, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

static int attr(cudaDeviceAttr a, int dev) {
    int v = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&v, a, dev));
    return v;
}

static int print_device() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int rt = 0, drv = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    CUDA_CHECK(cudaDriverGetVersion(&drv));

    // CUDA 13 去掉了 cudaDeviceProp 里的 clockRate 等字段，频率一律用 cudaDeviceGetAttribute 读。
    const int sms = attr(cudaDevAttrMultiProcessorCount, dev);
    const int clk_khz = attr(cudaDevAttrClockRate, dev);
    const int mem_khz = attr(cudaDevAttrMemoryClockRate, dev);
    const int bus_bits = attr(cudaDevAttrGlobalMemoryBusWidth, dev);

    std::printf("== GPU\n");
    std::printf("名称                 %s\n", p.name);
    std::printf("计算能力             sm_%d%d\n", attr(cudaDevAttrComputeCapabilityMajor, dev),
                attr(cudaDevAttrComputeCapabilityMinor, dev));
    std::printf("CUDA 运行时 / 驱动   %d / %d\n", rt, drv);
    std::printf("集成显卡（和 CPU 共用内存） %s\n", attr(cudaDevAttrIntegrated, dev) ? "是" : "否");
    std::printf("SM 个数              %d\n", sms);
    std::printf("GPU 最高频率         %d MHz（驱动报的上限，不是当前频率）\n", clk_khz / 1000);
    std::printf("显存频率 / 位宽      %d MHz / %d bit\n", mem_khz / 1000, bus_bits);
    std::printf("  → 按这两个数推算的理论带宽 %.1f GB/s（DDR 每周期传两次；仅供参考）\n",
                2.0 * mem_khz * 1e3 * bus_bits / 8 / 1e9);
    std::printf("L2 缓存              %d KB\n", attr(cudaDevAttrL2CacheSize, dev) / 1024);
    std::printf("每 SM 共享内存       %d KB（每块最多 %d KB，需显式申请才能超过 48 KB）\n",
                attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) / 1024,
                attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) / 1024);
    std::printf("每 SM 寄存器         %d 个（32 bit）\n", attr(cudaDevAttrMaxRegistersPerMultiprocessor, dev));
    std::printf("每 SM 最多线程       %d；每块最多线程 %d；warp 大小 %d\n",
                attr(cudaDevAttrMaxThreadsPerMultiProcessor, dev),
                attr(cudaDevAttrMaxThreadsPerBlock, dev), attr(cudaDevAttrWarpSize, dev));
    std::printf("全局内存             %.2f GB\n", p.totalGlobalMem / 1e9);
    return sms;
}

int main(int argc, char** argv) {
    const int sms = print_device();

    // 32M 个 float，每个数组 128 MB，三个共 384 MB：远大于 L2，量到的是真正的内存带宽。
    const size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (size_t)32 << 20;
    const size_t bytes = n * sizeof(float);

    // 1. 在 CPU 内存里准备输入
    std::vector<float> ha(n), hb(n), hc(n);
    for (size_t i = 0; i < n; ++i) {
        ha[i] = (float)(i % 1000) * 0.5f;
        hb[i] = (float)(i % 777) * -0.25f + 1.0f;
    }

    // 2. 在 GPU 上分配内存，把输入拷过去。
    //    Jetson 的 CPU 和 GPU 共用同一块 LPDDR5，cudaMemcpy 在这里是"内存到内存"的拷贝，
    //    不走 PCIe；但它仍然是一次真实的拷贝，要花时间和带宽。
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dc, 0, bytes));

    // 3. 发起内核：每块 256 个线程，块数 = SM 数 × 32（每个线程处理若干元素）
    const int threads = 256;
    const int blocks = sms * 32;

    // 先空跑几次热身：第一次发起内核要加载代码、建上下文，时间不能算进去。
    for (int i = 0; i < 3; ++i) vector_add<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());      // 发起参数错了会在这里报出来
    CUDA_CHECK(cudaDeviceSynchronize()); // 内核运行时出错会在这里报出来

    // 4. 用 CUDA 事件计时：事件在 GPU 的指令流里打时间戳，只量 GPU 上的执行时间。
    const int reps = 20;
    std::vector<float> ms(reps);
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(t0));
        vector_add<<<blocks, threads>>>(da, db, dc, n);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[r], t0, t1));
    }
    CUDA_CHECK(cudaGetLastError());

    // 5. 拷回结果，和 CPU 算的逐元素对比。
    //    单次 float 加法在 CPU 和 GPU 上都按 IEEE 754 舍入到最近，结果应当逐位相同；
    //    以后写矩阵乘时累加顺序不同，就只能要求"误差在容许范围内"了。
    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    double max_abs_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float ref = ha[i] + hb[i];
        const double err = std::abs((double)hc[i] - (double)ref);
        if (hc[i] != ref) ++mismatches;
        max_abs_err = std::max(max_abs_err, err);
    }

    std::vector<float> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    const float med = sorted[reps / 2];
    const double moved = 3.0 * bytes;    // 读 a、读 b、写 c
    std::printf("\n== vector_add（%zu 个 float，%d 块 × %d 线程，%d 次）\n", n, blocks, threads, reps);
    std::printf("正确性    不一致 %zu 个，最大绝对误差 %.3g → %s\n", mismatches, max_abs_err,
                mismatches == 0 ? "通过" : "失败");
    std::printf("耗时      中位数 %.3f ms，最快 %.3f ms，最慢 %.3f ms\n", med, sorted[0], sorted[reps - 1]);
    std::printf("有效带宽  %.1f GB/s（中位数）；%.1f GB/s（最快）\n", moved / (med * 1e-3) / 1e9,
                moved / (sorted[0] * 1e-3) / 1e9);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    return mismatches == 0 ? 0 : 1;
}

#include "ck/elementwise/add.cuh"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace {

// 造一组确定性输入，在 host 上算参考结果，再和 kernel 输出逐元素比对。
// 浮点加法在 float 下对这几个量级是精确的，可以直接用 == 比较。
void run_and_check(std::size_t n) {
    std::vector<float> h_a(n), h_b(n), h_ref(n);
    for (std::size_t i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i % 100) * 0.5f - 25.0f;
        h_b[i] = static_cast<float>(i % 37) - 18.0f;
        h_ref[i] = h_a[i] + h_b[i];
    }

    const std::size_t bytes = n * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // n == 0 时 bytes 为 0，cudaMalloc(0) 的行为不值得依赖，直接跳过分配
    if (bytes > 0) {
        ASSERT_EQ(cudaMalloc(&d_a, bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_b, bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_c, bytes), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    }

    ck::vector_add(d_a, d_b, d_c, static_cast<int>(n));

    // kernel 启动是异步的，启动错误不会在这里抛出来，必须显式检查
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "kernel launch failed, n = " << n;
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "kernel execution failed, n = " << n;

    std::vector<float> h_c(n, 0.0f);
    if (bytes > 0) {
        ASSERT_EQ(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    }

    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(h_c[i], h_ref[i]) << "mismatch at index " << i << ", n = " << n;
    }

    if (d_a != nullptr) { cudaFree(d_a); }
    if (d_b != nullptr) { cudaFree(d_b); }
    if (d_c != nullptr) { cudaFree(d_c); }
}

}  // namespace

// kernel 里 threads = 256，下面几个用例专门压 256 这个边界

TEST(VectorAdd, SingleElement) {
    run_and_check(1);
}

TEST(VectorAdd, BelowBlockSize) {
    run_and_check(255);
}

TEST(VectorAdd, ExactlyOneBlock) {
    run_and_check(256);
}

// 257 是经典 off-by-one 边界：跨两个 block，第二个 block 只有一个有效线程
TEST(VectorAdd, JustOverOneBlock) {
    run_and_check(257);
}

TEST(VectorAdd, PartialLastBlock) {
    run_and_check(1000);  // 1000 = 3 * 256 + 232
}

TEST(VectorAdd, ManyBlocks) {
    run_and_check(1 << 20);
}

TEST(VectorAdd, ZeroLength) {
    run_and_check(0);
}
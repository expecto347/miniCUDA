#include "ck/elementwise/add.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

// 有效带宽按 3 次访存算：读 a、读 b、写 c
constexpr int kRounds = 5;
constexpr int kItersPerRound = 50;
constexpr int kWarmupIters = 10;

void fill(std::vector<float>& v, float base) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = base + static_cast<float>(i % 64) * 0.25f;
    }
}

// 跑一轮计时，返回单次 kernel 的平均毫秒数
double time_once(const float* d_a, const float* d_b, float* d_c, int n,
                 cudaEvent_t start, cudaEvent_t stop) {
    cudaEventRecord(start, nullptr);
    for (int i = 0; i < kItersPerRound; ++i) {
        ck::vector_add(d_a, d_b, d_c, n);
    }
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return static_cast<double>(ms) / kItersPerRound;
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "failed to query device 0\n");
        return 1;
    }
    std::printf("device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    const std::size_t sizes[] = {
        1u << 10, 1u << 12, 1u << 14, 1u << 16,
        1u << 18, 1u << 20, 1u << 22, 1u << 24,
    };

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::printf("%12s %14s %14s %12s\n", "elements", "ms/iter(min)", "ms/iter(avg)", "GB/s(min)");
    std::printf("%12s %14s %14s %12s\n", "--------", "-------------", "-------------", "---------");

    for (std::size_t n : sizes) {
        const std::size_t bytes = n * sizeof(float);

        std::vector<float> h_a(n), h_b(n), h_c(n);
        fill(h_a, 1.0f);
        fill(h_b, 2.0f);

        float* d_a = nullptr;
        float* d_b = nullptr;
        float* d_c = nullptr;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

        for (int i = 0; i < kWarmupIters; ++i) {
            ck::vector_add(d_a, d_b, d_c, static_cast<int>(n));
        }
        cudaDeviceSynchronize();

        double best = 1e30;
        double sum = 0.0;
        for (int r = 0; r < kRounds; ++r) {
            const double ms = time_once(d_a, d_b, d_c, static_cast<int>(n), start, stop);
            best = ms < best ? ms : best;
            sum += ms;
        }
        const double avg = sum / kRounds;

        // 顺手校验一次结果，避免测出个错的快
        cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) {
            if (h_c[i] != h_a[i] + h_b[i]) {
                std::fprintf(stderr, "MISMATCH at %zu for n = %zu\n", i, n);
                ok = false;
                break;
            }
        }

        const double gbps = 3.0 * static_cast<double>(bytes) / (best * 1e-3) / 1e9;
        std::printf("%12zu %14.5f %14.5f %12.1f%s\n", n, best, avg, gbps, ok ? "" : "  <-- WRONG");

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
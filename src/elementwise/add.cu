#include "ck/elementwise/add.cuh"

#include <cuda_runtime.h>

namespace ck
{
    namespace
    {
        __global__ void vector_add_kernel(const float *a, const float *b, float *c, int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i < n)
            {
                c[i] = a[i] + b[i];
            }
        }
    } // namespace

    void vector_add(const float *a, const float *b, float *c, int n)
    {
        // n <= 0 时 blocks 会算成 0，而 <<<0, threads>>> 是非法启动配置
        if (n <= 0)
        {
            return;
        }

        constexpr int threads = 256;
        int blocks = (n + threads - 1) / threads;

        vector_add_kernel<<<blocks, threads>>>(a, b, c, n);
    }
} // namespace ck

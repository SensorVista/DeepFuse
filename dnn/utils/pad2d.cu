#include "common.cuh"

#include <cuda_runtime.h>

template<typename T>
__global__ void pad2d_kernel(
    const T* __restrict__ input,    // [N, C, H_in, W_in]
    T* __restrict__ output,         // [N, C, H_out, W_out]
    int N, int C,
    int H_in, int W_in,
    int H_out, int W_out,
    int pad_y, int pad_x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_in * W_in;
    if (idx >= total) return;

    int x_in = idx % W_in;
    int y_in = (idx / W_in) % H_in;
    int c    = (idx / (W_in * H_in)) % C;
    int n    = idx / (W_in * H_in * C);

    int x_out = x_in + pad_x;
    int y_out = y_in + pad_y;

    int input_offset  = ((n * C + c) * H_in + y_in) * W_in + x_in;
    int output_offset = ((n * C + c) * H_out + y_out) * W_out + x_out;

    output[output_offset] = input[input_offset];
}

namespace dnn::utils {

inline void pad2d(
    const float* input_dev,  // [N, C, H_in, W_in]
    float* output_dev,       // [N, C, H_out, W_out]
    int N, int C,
    int H_in, int W_in,
    int H_out, int W_out,
    int pad_y, int pad_x)
{
    int total = N * C * H_in * W_in;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    pad2d_kernel<float><<<blocks, threads>>>(
        input_dev, output_dev,
        N, C,
        H_in, W_in,
        H_out, W_out,
        pad_y, pad_x
    );

    cudaDeviceSynchronize();  // or remove if batched
}

} // namespace dnn::utils

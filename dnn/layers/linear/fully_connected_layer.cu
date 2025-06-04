#include "dnn/layers/linear/fully_connected_layer.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>
#include <sstream>

namespace dnn {

// Utility for __half support (host <-> device)
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ __half to_half(float v) { return __float2half(v); }
__device__ __forceinline__ __half to_half(__half v) { return v; }

// FC forward: output = input * W^T + b
template<typename T>
__global__ void fully_connected_forward_kernel(
    const T* __restrict__ input,     // [N, in_features]
    const T* __restrict__ weights,   // [out_features, in_features] (row-major)
    const T* __restrict__ bias,      // [out_features]
    T* __restrict__ output,          // [N, out_features]
    int N,                           // Batch size
    int in_features,
    int out_features
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // output neuron idx
    int batch = blockIdx.y;

    if (row < out_features && batch < N) {
        float sum = 0.f;
        for (int i = 0; i < in_features; ++i) {
            float inp = to_float(input[batch * in_features + i]);
            float w = to_float(weights[row * in_features + i]);
            sum += inp * w;
        }
        sum += bias ? to_float(bias[row]) : 0.f;
        if constexpr (std::is_same<T, __half>::value)
            output[batch * out_features + row] = to_half(sum);
        else
            output[batch * out_features + row] = static_cast<T>(sum);
    }
}

template<typename T>
__global__ void fc_backward_input(
    const T* __restrict__ dY,     // [N, out_features]
    const T* __restrict__ W,      // [out_features, in_features]
    T* __restrict__ dX,           // [N, in_features]
    int N, int in_features, int out_features)
{
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;  // in_features
    int batch = blockIdx.y;                              // N

    if (in_idx < in_features && batch < N) {
        float sum = 0.f;
        for (int j = 0; j < out_features; ++j) {
            float dy = to_float(dY[batch * out_features + j]);
            float w = to_float(W[j * in_features + in_idx]);
            sum += dy * w;
        }
        if constexpr (std::is_same<T, __half>::value)
            dX[batch * in_features + in_idx] = to_half(sum);
        else
            dX[batch * in_features + in_idx] = static_cast<T>(sum);
    }
}

template<typename T>
__global__ void fc_backward_weights(
    const T* __restrict__ dY,     // [N, out_features]
    const T* __restrict__ X,      // [N, in_features]
    T* __restrict__ dW,           // [out_features, in_features]
    int N, int in_features, int out_features)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x; // out_features
    int in_idx = blockIdx.y * blockDim.y + threadIdx.y;  // in_features

    if (out_idx < out_features && in_idx < in_features) {
        float sum = 0.f;
        for (int n = 0; n < N; ++n) {
            float dy = to_float(dY[n * out_features + out_idx]);
            float x = to_float(X[n * in_features + in_idx]);
            sum += dy * x;
        }
        if constexpr (std::is_same<T, __half>::value)
            dW[out_idx * in_features + in_idx] = to_half(sum);
        else
            dW[out_idx * in_features + in_idx] = static_cast<T>(sum);
    }
}

template<typename T>
__global__ void fc_backward_bias(
    const T* __restrict__ dY,   // [N, out_features]
    T* __restrict__ db,         // [out_features]
    int N, int out_features)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x; // out_features
    if (out_idx < out_features) {
        float sum = 0.f;
        for (int n = 0; n < N; ++n)
            sum += to_float(dY[n * out_features + out_idx]);
        if constexpr (std::is_same<T, __half>::value)
            db[out_idx] = to_half(sum);
        else
            db[out_idx] = static_cast<T>(sum);
    }
}

template<typename T>
tensor<T> FullyConnectedLayer<T>::forward(const tensor<T>& input) {
    int N = input.shape()[0];
    tensor<T> output({ N, out_features_ });
    dim3 blockDim(256);
    dim3 gridDim((out_features_ + blockDim.x - 1) / blockDim.x, N);

    fully_connected_forward_kernel<T> << <gridDim, blockDim, 0, 0 >> > (
        input.data(),
        weights_.data(),
        bias_.data(),
        output.data(),
        N, in_features_, out_features_
        );

    utils::THROW_CUDA_EX();

    return output;
}

template<typename T>
tensor<T> FullyConnectedLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    const int N = input.shape()[0];

    // [1] Allocate gradient of input for chain rule
    tensor<T> grad_input({N, in_features_});

    // [2] Compute grad_input: dX = dY * W
    {
        dim3 blockDim(256);
        dim3 gridDim((in_features_ + blockDim.x - 1) / blockDim.x, N);
        fc_backward_input<T><<<gridDim, blockDim>>>(
            grad_output.data(),
            weights_.data(),
            grad_input.data(),
            N,
            in_features_,
            out_features_
        );
    }

    // [3] Compute grad_weights: dW = dY^T * X
    {
        dim3 blockDim(16, 16);
        dim3 gridDim(
            (out_features_ + blockDim.x - 1) / blockDim.x,
            (in_features_ + blockDim.y - 1) / blockDim.y
        );
        fc_backward_weights<T><<<gridDim, blockDim>>>(
            grad_output.data(),
            input.data(),
            grad_weights_.data(),  // directly write into persistent buffer
            N,
            in_features_,
            out_features_
        );
    }

    // [4] Compute grad_bias: db = sum over batch of dY
    {
        dim3 blockDim(256);
        dim3 gridDim((out_features_ + blockDim.x - 1) / blockDim.x);
        fc_backward_bias<T><<<gridDim, blockDim>>>(
            grad_output.data(),
            grad_bias_.data(),  // directly write into persistent buffer
            N,
            out_features_
        );
    }

    cudaDeviceSynchronize();

    utils::THROW_CUDA_EX();

    // [5] Return grad_input to propagate further
    return grad_input;
}

// Explicit template instantiations
template class FullyConnectedLayer<float>;  // FP32
// template class FullyConnectedLayer<__half>; // FP16
// template class FullyConnectedLayer<__nv_bfloat16>; // BF16

} // namespace dnn 
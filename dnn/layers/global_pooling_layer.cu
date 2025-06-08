#include "pooling_layer.cuh"
#include "../utils/common.cuh"

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

// -------------------------
// CUDA kernels fallback
// -------------------------

#ifndef ENABLE_CUDNN
template<typename T>
__global__ void global_avg_pool_forward(const T* input, T* output, int B, int C, int H, int W) {
    int bc = blockIdx.x;
    int b = bc / C;
    int c = bc % C;

    if (b >= B || c >= C) return;

    const int offset = ((b * C + c) * H * W);
    T sum = 0;
    for (int i = 0; i < H * W; ++i)
        sum += input[offset + i];

    output[b * C + c] = sum / static_cast<T>(H * W);
}

template<typename T>
__global__ void global_avg_pool_backward(const T* grad_out, T* grad_in, int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;

    if (idx < total) {
        int b = idx / (C * H * W);
        int c = (idx / (H * W)) % C;
        grad_in[idx] = grad_out[b * C + c] / static_cast<T>(H * W);
    }
}
#endif

namespace dnn {

template<typename T>
GlobalPoolingLayer<T>::GlobalPoolingLayer(PoolingType type)
    : type_(type)
#ifdef ENABLE_CUDNN
    , pool_desc_(nullptr)
    , input_desc_(nullptr)
    , output_desc_(nullptr)
#endif
{
#ifdef ENABLE_CUDNN
    utils::CHECK_CUDNN_EX(cudnnCreatePoolingDescriptor(&pool_desc_));
    utils::CHECK_CUDNN_EX(cudnnCreateTensorDescriptor(&input_desc_));
    utils::CHECK_CUDNN_EX(cudnnCreateTensorDescriptor(&output_desc_));
#endif
}

template<typename T>
GlobalPoolingLayer<T>::~GlobalPoolingLayer() {
#ifdef ENABLE_CUDNN
    if (pool_desc_) cudnnDestroyPoolingDescriptor(pool_desc_);
    if (input_desc_) cudnnDestroyTensorDescriptor(input_desc_);
    if (output_desc_) cudnnDestroyTensorDescriptor(output_desc_);
#endif
}

template<typename T>
tensor<T> GlobalPoolingLayer<T>::forward(const tensor<T>& input) {
    const auto& shape = input.shape();
    const int B = shape[0];
    const int C = shape[1];
    const int H = shape[2];
    const int W = shape[3];

    tensor<T> output({B, C});  // Output shape: [B, C]

#ifdef ENABLE_CUDNN
    cudnnPoolingMode_t mode;
    switch (type_) {
        case PoolingType::Max:
            mode = CUDNN_POOLING_MAX;
            break;
        case PoolingType::Average:
            mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            break;
        case PoolingType::MaxDeterministic:
            mode = CUDNN_POOLING_MAX_DETERMINISTIC;
            break;
        default:
            throw std::runtime_error("Unsupported pooling type");
    }

    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, cudnn_dtype<T>(), B, C, H, W));
    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, cudnn_dtype<T>(), B, C, 1, 1));

    utils::CHECK_CUDNN_EX(cudnnSetPooling2dDescriptor(
        pool_desc_,
        mode,
        CUDNN_NOT_PROPAGATE_NAN,
        H, W,
        0, 0,
        1, 1
    ));

    const float alpha = 1.0f, beta = 0.0f;
    utils::CHECK_CUDNN_EX(cudnnPoolingForward(
        utils::cudnn_handle(),
        pool_desc_,
        &alpha,
        input_desc_,
        input.data(),
        &beta,
        output_desc_,
        output.data()
    ));
#else
    // Fallback: CUDA kernel
    const int num_threads = 256;
    const int num_blocks = B * C;

    global_avg_pool_forward<<<num_blocks, num_threads>>>(
        input.data(), output.data(), B, C, H, W
    );
    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());
#endif

    return output;
}

template<typename T>
tensor<T> GlobalPoolingLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    const auto& shape = input.shape();
    const int B = shape[0];
    const int C = shape[1];
    const int H = shape[2];
    const int W = shape[3];

    tensor<T> grad_input({B, C, H, W});

#ifdef ENABLE_CUDNN
    const float alpha = 1.0f, beta = 0.0f;

    utils::CHECK_CUDNN_EX(cudnnPoolingBackward(
        utils::cudnn_handle(),
        pool_desc_,
        &alpha,
        output_desc_,
        nullptr, // y (not used)
        output_desc_,
        grad_output.data(),
        input_desc_,
        input.data(),
        &beta,
        input_desc_,
        grad_input.data()
    ));
#else
    const int total = B * C * H * W;
    const int num_threads = 256;
    const int num_blocks = (total + num_threads - 1) / num_threads;

    global_avg_pool_backward<<<num_blocks, num_threads>>>(
        grad_output.data(), grad_input.data(), B, C, H, W
    );
    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());
#endif

    return grad_input;
}

// -------------------------
// Explicit template instantiation
// -------------------------

template class GlobalPoolingLayer<float>;
template class GlobalPoolingLayer<__half>;

}  // namespace dnn

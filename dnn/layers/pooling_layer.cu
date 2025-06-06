#include "pooling_layer.cuh"
#include "../utils/common.cuh"
#include "../core/device.cuh"

#include <cuda_runtime.h>

namespace dnn {

#ifndef ENABLE_CUDNN
template<typename T>
__global__ void avg_pool_forward_2d(
    T* output, const T* input,
    int batch_size, int channels,
    int height, int width,
    int kernel_size, int stride,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_h * out_w) return;

    int b = idx / (channels * out_h * out_w);
    int c = (idx / (out_h * out_w)) % channels;
    int oh = (idx / out_w) % out_h;
    int ow = idx % out_w;

    T sum = 0.0f;
    int count = 0;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;
            if (ih < height && iw < width) {
                int in_idx = (
                    b * channels * height * width +
                    c * height * width +
                    ih * width + iw
                );
                sum += input[in_idx];
                ++count;
            }
        }
    }

    output[idx] = sum / static_cast<T>(count);
}

template<typename T>
__global__ void max_pool_forward_2d(
    T* output, const T* input,
    int batch_size, int channels,
    int height, int width,
    int kernel_size, int stride,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_h * out_w) return;

    int b = idx / (channels * out_h * out_w);
    int c = (idx / (out_h * out_w)) % channels;
    int oh = (idx / out_w) % out_h;
    int ow = idx % out_w;

    T max_val = utils::neg_infinity<T>();

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;
            if (ih < height && iw < width) {
                int in_idx = (
                    b * channels * height * width +
                    c * height * width +
                    ih * width + iw
                );
                max_val = max(max_val, input[in_idx]);
            }
        }
    }

    output[idx] = max_val;
}

template<typename T>
__global__ void avg_pool_backward_2d(
    T* grad_input, const T* grad_output,
    int batch_size, int channels,
    int height, int width,
    int kernel_size, int stride,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;

    int b = idx / (channels * height * width);
    int c = (idx / (height * width)) % channels;
    int h = (idx / width) % height;
    int w = idx % width;

    int oh = h / stride;
    int ow = w / stride;

    if (oh < out_h && ow < out_w) {
        int out_idx = (
            b * channels * out_h * out_w +
            c * out_h * out_w +
            oh * out_w + ow
        );
        T grad = grad_output[out_idx] / static_cast<T>(kernel_size * kernel_size);
        atomicAdd(&grad_input[idx], grad);
    }
}

template<typename T>
__global__ void max_pool_backward_2d(
    T* grad_input, const T* grad_output, const T* input,
    int batch_size, int channels,
    int height, int width,
    int kernel_size, int stride,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_h * out_w) return;

    int b = idx / (channels * out_h * out_w);
    int c = (idx / (out_h * out_w)) % channels;
    int oh = (idx / out_w) % out_h;
    int ow = idx % out_w;

    int h_start = oh * stride;
    int w_start = ow * stride;

    T max_val = utils::neg_infinity<T>();
    int max_idx = -1;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = h_start + kh;
            int iw = w_start + kw;
            if (ih < height && iw < width) {
                int in_idx = (
                    b * channels * height * width +
                    c * height * width +
                    ih * width + iw
                    );
                if (input[in_idx] > max_val) {
                    max_val = input[in_idx];
                    max_idx = in_idx;
                }
            }
        }
    }

    if (max_idx != -1) {
        atomicAdd(&grad_input[max_idx], grad_output[idx]);
    }
}
#endif

template<typename T>
PoolingLayer<T>::PoolingLayer(PoolingType type, int kernel_size, int stride)
    : type_(type)
    , kernel_size_(kernel_size)
    , stride_(stride)
#ifdef ENABLE_CUDNN
    , pool_desc_(nullptr)
    , input_desc_(nullptr)
    , output_desc_(nullptr)
#endif
{
    if (stride == 0 || kernel_size == 0) {
        throw std::invalid_argument("Stride and kernel_size must be non-zero.");
    }

#ifdef ENABLE_CUDNN
    utils::CHECK_CUDNN_EX(cudnnCreatePoolingDescriptor(&pool_desc_));
    utils::CHECK_CUDNN_EX(cudnnCreateTensorDescriptor(&input_desc_));
    utils::CHECK_CUDNN_EX(cudnnCreateTensorDescriptor(&output_desc_));

    cudnnPoolingMode_t mode;
    switch (type) {
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

    utils::CHECK_CUDNN_EX(cudnnSetPooling2dDescriptor(
        pool_desc_,
        mode,
        CUDNN_NOT_PROPAGATE_NAN,
        kernel_size, kernel_size,
        0, 0,
        stride, stride
    ));
#endif
}

template<typename T>
PoolingLayer<T>::~PoolingLayer() {
#ifdef ENABLE_CUDNN
    if (pool_desc_) utils::CHECK_CUDNN_EX(cudnnDestroyPoolingDescriptor(pool_desc_));
    if (input_desc_) utils::CHECK_CUDNN_EX(cudnnDestroyTensorDescriptor(input_desc_));
    if (output_desc_) utils::CHECK_CUDNN_EX(cudnnDestroyTensorDescriptor(output_desc_));
#endif
}

template<typename T>
tensor<T> PoolingLayer<T>::forward(const tensor<T>& input) {
    int batch_size = input.shape()[0];
    int channels = input.shape()[1];
    int height = input.shape()[2];
    int width = input.shape()[3];

    if (height % stride_ != 0 || width % stride_ != 0) {
        throw std::runtime_error("PoolingLayer expects divisible dimensions");
    }

    int out_h = height / stride_;
    int out_w = width / stride_;

    tensor<T> output({ batch_size, channels, out_h, out_w });

#ifdef ENABLE_CUDNN
    // Set up input and output descriptors
    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(input_desc_,
        CUDNN_TENSOR_NCHW, utils::dnn_type<T>(),
        batch_size, channels, height, width));

    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(output_desc_,
        CUDNN_TENSOR_NCHW, utils::dnn_type<T>(),
        batch_size, channels, out_h, out_w));

    const float alpha = 1.0f, beta = 0.0f;
    utils::CHECK_CUDNN_EX(cudnnPoolingForward(
        Cuda::current().cudnn(),
        pool_desc_,
        &alpha, input_desc_, input.data(),
        &beta, output_desc_, output.data()));
#else
    int size = output.size();
    int block = 256;
    int grid = (size + block - 1) / block;

    if (type_ == PoolingType::Average) {
        avg_pool_forward_2d<<<grid, block>>>(
            output.data(), input.data(),
            batch_size, channels, height, width,
            kernel_size_, stride_,
            out_h, out_w
        );
    } else {
        max_pool_forward_2d<<<grid, block>>>(
            output.data(), input.data(),
            batch_size, channels, height, width,
            kernel_size_, stride_,
            out_h, out_w
        );
    }

    utils::THROW_CUDA_EX();
#endif

    return output;
}

template<typename T>
tensor<T> PoolingLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    int batch_size = input.shape()[0];
    int channels = input.shape()[1];
    int height = input.shape()[2];
    int width = input.shape()[3];

    int out_h = height / stride_;
    int out_w = width / stride_;

    tensor<T> grad_input(input.shape());
    utils::CHECK_CUDA_EX(cudaMemset(grad_input.data(), 0, grad_input.size() * sizeof(T)));

#ifdef ENABLE_CUDNN
    // Set up input and output descriptors
    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(input_desc_,
        CUDNN_TENSOR_NCHW, utils::dnn_type<T>(),
        batch_size, channels, height, width));

    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(output_desc_,
        CUDNN_TENSOR_NCHW, utils::dnn_type<T>(),
        batch_size, channels, out_h, out_w));

    const float alpha = 1.0f, beta = 0.0f;
    utils::CHECK_CUDNN_EX(cudnnPoolingBackward(
        Cuda::current().cudnn(),
        pool_desc_,
        &alpha, output_desc_, grad_output.data(),
        input_desc_, input.data(),
        &beta, input_desc_, grad_input.data()));
#else
    int size = grad_input.size();
    int block = 256;
    int grid = (size + block - 1) / block;

    if (type_ == PoolingType::Average) {
        avg_pool_backward_2d<<<grid, block>>>(
            grad_input.data(), grad_output.data(),
            batch_size, channels, height, width,
            kernel_size_, stride_,
            out_h, out_w
        );
    } else {
        max_pool_backward_2d<<<grid, block>>>(
            grad_input.data(), grad_output.data(), input.data(),
            batch_size, channels, height, width,
            kernel_size_, stride_,
            out_h, out_w
        );
    }

    utils::THROW_CUDA_EX();
#endif

    return grad_input;
}

// Explicit template instantiations
template class PoolingLayer<float>;  // FP32
// template class PoolingLayer<__half>; // FP16
// template class PoolingLayer<__nv_bfloat16>; // BF16

} // namespace dnn 
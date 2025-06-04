#include "dnn/layers/pooling/avg_pool_layer.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>

namespace dnn {

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
AvgPoolLayer<T>::AvgPoolLayer(int kernel_size, int stride) 
    : kernel_size_(kernel_size), stride_(stride) {}

template<typename T>
tensor<T> AvgPoolLayer<T>::forward(const tensor<T>& input) {
    int batch_size = input.shape()[0];
    int channels = input.shape()[1];
    int height = input.shape()[2];
    int width = input.shape()[3];

    if (height % stride_ != 0 || width % stride_ != 0) {
        throw std::runtime_error("AvgPoolLayer expects divisible dimensions");
    }

    int out_h = height / stride_;
    int out_w = width / stride_;

    tensor<T> output({ batch_size, channels, out_h, out_w });
    int size = output.size();

    int block = 256;
    int grid = (size + block - 1) / block;

    avg_pool_forward_2d << <grid, block >> > (
        output.data(), input.data(),
        batch_size, channels, height, width,
        static_cast<int>(kernel_size_), static_cast<int>(stride_),
        out_h, out_w
        );

    utils::THROW_CUDA_EX();

    return output;
}

template<typename T>
tensor<T> AvgPoolLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    int batch_size = input.shape()[0];
    int channels = input.shape()[1];
    int height = input.shape()[2];
    int width = input.shape()[3];

    int out_h = height / stride_;
    int out_w = width / stride_;

    tensor<T> grad_input(input.shape());
    utils::CHECK_CUDA_EX(cudaMemset(grad_input.data(), 0, grad_input.size() * sizeof(T)));

    int size = grad_input.size();
    int block = 256;
    int grid = (size + block - 1) / block;

    avg_pool_backward_2d<<<grid, block >>> (
        grad_input.data(), grad_output.data(),
        batch_size, channels, height, width,
        kernel_size_, stride_,
        out_h, out_w
        );

    utils::THROW_CUDA_EX();

    return grad_input;
}

// Explicit template instantiations
template class AvgPoolLayer<float>;  // FP32
// template class AvgPoolLayer<__half>; // FP16
// template class AvgPoolLayer<__nv_bfloat16>; // BF16

} // namespace dnn 
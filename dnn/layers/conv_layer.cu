#include "conv_layer.cuh"
#include "../utils/common.cuh"
#include "../core/device.cuh"
#include "../core/tensor.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>
#include <sstream>

using namespace std;

#ifndef ENABLE_CUDNN
// CUDA kernels for non-cuDNN implementation
template<typename T>
__global__ void conv2d_forward(
    T* output,           // [batch_size, out_channels, out_height, out_width]
    const T* input,      // [batch_size, in_channels, in_height, in_width]
    const T* weight,     // [out_channels, in_channels, kernel_size, kernel_size]
    const T* bias,       // [out_channels]
    const uint8_t* connection_mask, // [out_channels, in_channels]
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_height * out_width;
    if (idx >= total) return;

    // Compute output indices
    int b      = idx / (out_channels * out_height * out_width);
    int c_out  = (idx / (out_height * out_width)) % out_channels;
    int h_out  = (idx / out_width) % out_height;
    int w_out  = idx % out_width;

    float acc = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        // Check connectivity mask
        if (connection_mask && connection_mask[c_out * in_channels + c_in] == 0) {
            continue;
        }

        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = h_out * stride + ky - padding;
                int in_x = w_out * stride + kx - padding;

                if (in_y < 0 || in_y >= in_height || in_x < 0 || in_x >= in_width)
                    continue;

                int in_idx = ((b * in_channels + c_in) * in_height + in_y) * in_width + in_x;
                int w_idx  = ((c_out * in_channels + c_in) * kernel_size + ky) * kernel_size + kx;

                if constexpr (std::is_same<T, half>::value) {
                    acc += __half2float(input[in_idx]) * __half2float(weight[w_idx]);
                } else {
                    acc += static_cast<float>(input[in_idx]) * static_cast<float>(weight[w_idx]);
                }
            }
        }
    }

    if constexpr (std::is_same<T, half>::value) {
        if (bias) acc += __half2float(bias[c_out]);
        output[idx] = __float2half(acc);
    } else {
        if (bias) acc += static_cast<float>(bias[c_out]);
        output[idx] = static_cast<T>(acc);
    }
}

template<typename T>
__global__ void conv2d_backward(
    T* grad_input, 
    const T* grad_output, 
    const T* weight,
    const uint8_t* connection_mask,
    int N, 
    int C_in, 
    int C_out,
    int H_in, 
    int W_in,
    int H_out, 
    int W_out,
    int K, 
    int stride, 
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H_in * W_in;
    if (idx >= total) return;

    int w = idx % W_in;
    int h = (idx / W_in) % H_in;
    int c_in = (idx / (H_in * W_in)) % C_in;
    int n = idx / (C_in * H_in * W_in);

    float grad = 0.0f;
    for (int c_out = 0; c_out < C_out; ++c_out) {
        // Check connectivity mask
        if (connection_mask && connection_mask[c_out * C_in + c_in] == 0) {
            continue;
        }

        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                int out_y = (h + padding - ky) / stride;
                int out_x = (w + padding - kx) / stride;

                if (out_y < 0 || out_y >= H_out || (h + padding - ky) % stride != 0) continue;
                if (out_x < 0 || out_x >= W_out || (w + padding - kx) % stride != 0) continue;

                int out_idx = ((n * C_out + c_out) * H_out + out_y) * W_out + out_x;
                int w_idx = ((c_out * C_in + c_in) * K + ky) * K + kx;

                grad += static_cast<float>(grad_output[out_idx]) * static_cast<float>(weight[w_idx]);
            }
        }
    }
    grad_input[idx] = static_cast<T>(grad);
}

template<typename T>
__global__ void conv_weight_grad_2d(
    const T* input, const T* grad_output, T* grad_weight,
    const uint8_t* connection_mask,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int K, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * K * K;
    if (idx >= total) return;

    int kx = idx % K;
    int ky = (idx / K) % K;
    int c_in = (idx / (K * K)) % C_in;
    int c_out = idx / (C_in * K * K);

    // Check connectivity mask
    if (connection_mask && connection_mask[c_out * C_in + c_in] == 0) {
        grad_weight[idx] = 0.0f; // No connection, so gradient is zero
        return;
    }

    float grad = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                int in_y = h * stride - padding + ky;
                int in_x = w * stride - padding + kx;
                if (in_y < 0 || in_y >= H_in || in_x < 0 || in_x >= W_in) continue;

                int in_idx = ((n * C_in + c_in) * H_in + in_y) * W_in + in_x;
                int out_idx = ((n * C_out + c_out) * H_out + h) * W_out + w;

                grad += static_cast<float>(input[in_idx]) * static_cast<float>(grad_output[out_idx]);
            }
        }
    }
    grad_weight[idx] = static_cast<T>(grad);
}

template<typename T>
__global__ void conv_bias_grad_2d(
    const T* grad_output, T* grad_bias,
    int N, int C_out, int H_out, int W_out
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C_out) return;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H_out; ++h)
            for (int w = 0; w < W_out; ++w)
                sum += static_cast<float>(grad_output[((n * C_out + c) * H_out + h) * W_out + w]);

    grad_bias[c] = static_cast<T>(sum);
}
#endif

namespace dnn {

template<typename T>
ConvLayer<T>::ConvLayer(int in_channels, int out_channels, const std::vector<int>& kernel_size, int stride, int padding, const std::vector<std::vector<bool>>& connection_table /* = {} */)
    : LayerWeightBias<T>(
        tensor<T>({out_channels, in_channels, kernel_size[0], kernel_size[1]}),
        tensor<T>({1, out_channels, 1, 1}),
        tensor<T>({out_channels, in_channels, kernel_size[0], kernel_size[1]}),
        tensor<T>({1, out_channels, 1, 1})
    )
    , in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , connection_mask_dev_({out_channels, in_channels})
    , use_sparse_connectivity_(connection_table.size() > 0)
#ifdef ENABLE_CUDNN
    , filter_desc_(nullptr)
    , conv_desc_(nullptr)
    , input_desc_(nullptr)
    , output_desc_(nullptr)
#endif
{
    if (use_sparse_connectivity_) {
        if (connection_table.size() != out_channels || connection_table[0].size() != in_channels) {
            throw std::runtime_error("Connection table dimensions do not match in/out channels.");
        }
        std::vector<uint8_t> host_connection_mask(out_channels * in_channels);
        for (int i = 0; i < out_channels; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                host_connection_mask[i * in_channels + j] = connection_table[i][j] ? 1 : 0;
            }
        }
        connection_mask_dev_.upload(host_connection_mask.data());
    }
    
    initialize_weights();
}

template<typename T>
ConvLayer<T>::~ConvLayer() {
#ifdef ENABLE_CUDNN
    if (filter_desc_) utils::CHECK_CUDNN_EX(cudnnDestroyFilterDescriptor(filter_desc_));
    if (conv_desc_) utils::CHECK_CUDNN_EX(cudnnDestroyConvolutionDescriptor(conv_desc_));
    if (input_desc_) utils::CHECK_CUDNN_EX(cudnnDestroyTensorDescriptor(input_desc_));
    if (output_desc_) utils::CHECK_CUDNN_EX(cudnnDestroyTensorDescriptor(output_desc_));
#endif
}

template<typename T>
tensor<T> ConvLayer<T>::forward(const tensor<T>& input) {
    // Input dimensions (assumes NCHW: [batch, channels, height, width])
    int batch_size   = input.shape()[0];
    int input_height = input.shape()[2];
    int input_width  = input.shape()[3];

    // Kernel size
    int kH = kernel_size_[0];
    int kW = kernel_size_[1];

    // Calculate output dimensions with padding
    int output_height = (input_height + 2 * padding_ - kH) / stride_ + 1;
    int output_width  = (input_width  + 2 * padding_ - kW) / stride_ + 1;

    // Validate input dimensions
    if (input_height < kH || input_width < kW) {
        std::stringstream ss;
        ss << "Input dimensions [" << input_height << "x" << input_width 
           << "] are smaller than kernel size [" << kH << "x" << kW << "]";
        throw std::runtime_error(ss.str());
    }

    // Validate output dimensions
    if (output_height == 0 || output_width == 0) {
        std::stringstream ss;
        ss << "Invalid output dimensions in ConvLayer: "
           << "input=[" << input_height << "x" << input_width << "], "
           << "kernel=[" << kH << "x" << kW << "], "
           << "stride=" << stride_ << ", "
           << "padding=" << padding_ << ", "
           << "output=[" << output_height << "x" << output_width << "]. "
           << "Try increasing padding to at least " << (kH - 1) / 2;
        throw std::runtime_error(ss.str());
    }

    std::vector<int> output_shape = {
        batch_size,
        out_channels_,
        output_height,
        output_width
    };

    tensor<T> output(output_shape);

#ifdef ENABLE_CUDNN
    // Set up cuDNN descriptors
    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(input_desc_,
        CUDNN_TENSOR_NCHW, utils::dnn_type<T>(),
        batch_size, in_channels_, input_height, input_width));

    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(output_desc_,
        CUDNN_TENSOR_NCHW, utils::dnn_type<T>(),
        batch_size, out_channels_, output_height, output_width));

    utils::CHECK_CUDNN_EX(cudnnSetFilter4dDescriptor(filter_desc_,
        utils::dnn_type<T>(), CUDNN_TENSOR_NCHW,
        out_channels_, in_channels_, kH, kW));

    utils::CHECK_CUDNN_EX(cudnnSetConvolution2dDescriptor(conv_desc_,
        padding_, padding_, stride_, stride_, 1, 1,
        CUDNN_CROSS_CORRELATION, utils::dnn_type<T>()));

    // Get the best algorithm for forward convolution
    cudnnConvolutionFwdAlgo_t fwd_algo;
    utils::CHECK_CUDNN_EX(cudnnGetConvolutionForwardAlgorithm(
        Cuda::current().cudnn(),
        input_desc_, filter_desc_, conv_desc_, output_desc_,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo));

    // Get workspace size
    size_t workspace_bytes = 0;
    utils::CHECK_CUDNN_EX(cudnnGetConvolutionForwardWorkspaceSize(
        Cuda::current().cudnn(),
        input_desc_, filter_desc_, conv_desc_, output_desc_,
        fwd_algo, &workspace_bytes));

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        utils::CHECK_CUDA_EX(cudaMalloc(&workspace, workspace_bytes));
    }

    // Perform forward convolution
    const float alpha = 1.0f, beta = 0.0f;
    utils::CHECK_CUDNN_EX(cudnnConvolutionForward(
        Cuda::current().cudnn(),
        &alpha, input_desc_, input.data(),
        filter_desc_, this->weights_.data(),
        conv_desc_, fwd_algo,
        workspace, workspace_bytes,
        &beta, output_desc_, output.data()));

    // Add bias
    utils::CHECK_CUDNN_EX(cudnnAddTensor(
        Cuda::current().cudnn(),
        &alpha, this->bias_.desc(), this->bias_.data(),
        &alpha, output_desc_, output.data()));

    // Clean up workspace
    if (workspace) {
        cudaFree(workspace);
    }
#else
    int total = batch_size * out_channels_ * output_height * output_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_forward<T><<<blocks, threads>>>(
        output.data(),
        input.data(),
        this->weights_.data(),
        this->bias_.data(),
        use_sparse_connectivity_ ? connection_mask_dev_.data() : nullptr,
        batch_size,
        in_channels_,
        out_channels_,
        input_height,
        input_width,
        output_height,
        output_width,
        kH,
        stride_,
        padding_
    );

    utils::THROW_CUDA_EX();
#endif

    return output;
}

template<typename T>
tensor<T> ConvLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    const int N = input.shape()[0];
    const int C_in = input.shape()[1];
    const int H_in = input.shape()[2];
    const int W_in = input.shape()[3];
    const int C_out = grad_output.shape()[1];
    const int H_out = grad_output.shape()[2];
    const int W_out = grad_output.shape()[3];
    const int K = kernel_size_[0];

    tensor<T> grad_input(input.shape());
    grad_input.zero();

#ifdef ENABLE_CUDNN
    auto& cuda = Cuda::current();
    auto handle_dnn = cuda.cudnn();

    // Get the best algorithms for backward passes
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;

    utils::CHECK_CUDNN_EX(cudnnGetConvolutionBackwardDataAlgorithm(
        handle_dnn, filter_desc_, grad_output.desc(), conv_desc_, grad_input.desc(),
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwd_data_algo));

    utils::CHECK_CUDNN_EX(cudnnGetConvolutionBackwardFilterAlgorithm(
        handle_dnn, input.desc(), grad_output.desc(), conv_desc_, filter_desc_,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwd_filter_algo));

    // Get workspace sizes
    size_t workspace_data_bytes = 0, workspace_filter_bytes = 0;
    utils::CHECK_CUDNN_EX(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle_dnn, filter_desc_, grad_output.desc(), conv_desc_, grad_input.desc(),
        bwd_data_algo, &workspace_data_bytes));

    utils::CHECK_CUDNN_EX(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle_dnn, input.desc(), grad_output.desc(), conv_desc_, filter_desc_,
        bwd_filter_algo, &workspace_filter_bytes));

    // Allocate workspace
    size_t workspace_bytes = std::max(workspace_data_bytes, workspace_filter_bytes);
    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        utils::CHECK_CUDA_EX(cudaMalloc(&workspace, workspace_bytes));
    }

    const float alpha = 1.0f, beta = 0.0f;

    // Backward data
    utils::CHECK_CUDNN_EX(cudnnConvolutionBackwardData(
        handle_dnn,
        &alpha, filter_desc_, this->weights_.data(),
        grad_output.desc(), grad_output.data(),
        conv_desc_, bwd_data_algo,
        workspace, workspace_bytes,
        &beta, grad_input.desc(), grad_input.data()));

    // Backward filter
    utils::CHECK_CUDNN_EX(cudnnConvolutionBackwardFilter(
        handle_dnn,
        &alpha, input.desc(), input.data(),
        grad_output.desc(), grad_output.data(),
        conv_desc_, bwd_filter_algo,
        workspace, workspace_bytes,
        &beta, this->grad_weights_.desc(), this->grad_weights_.data()));

    // Backward bias
    utils::CHECK_CUDNN_EX(cudnnConvolutionBackwardBias(
        handle_dnn,
        &alpha, grad_output.desc(), grad_output.data(),
        &beta, this->grad_bias_.desc(), this->grad_bias_.data()));

    // Clean up workspace
    if (workspace) {
        cudaFree(workspace);
    }
#else
    // Grad input
    {
        int total = N * C_in * H_in * W_in;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        conv2d_backward<T><<<blocks, threads>>>(
            grad_input.data(), grad_output.data(), this->weights_.data(),
            use_sparse_connectivity_ ? connection_mask_dev_.data() : nullptr,
            N, C_in, C_out, H_in, W_in, H_out, W_out, K, stride_, padding_);
    }

    // Grad weights
    {
        int total = C_out * C_in * K * K;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        conv_weight_grad_2d<T><<<blocks, threads>>>(
            input.data(), grad_output.data(), this->grad_weights_.data(),
            use_sparse_connectivity_ ? connection_mask_dev_.data() : nullptr,
            N, C_in, C_out, H_in, W_in, H_out, W_out, K, stride_, padding_);
    }

    // Grad bias
    {
        int threads = 256;
        int blocks = (C_out + threads - 1) / threads;
        conv_bias_grad_2d<T><<<blocks, threads>>>(
            grad_output.data(), this->grad_bias_.data(),
            N, C_out, H_out, W_out);
    }

    cudaDeviceSynchronize();
    utils::THROW_CUDA_EX();
#endif

    return grad_input;
}

template<typename T>
void ConvLayer<T>::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    // Xavier/Glorot initialization
    float limit = std::sqrt(6.0f / (in_channels_ * kernel_size_[0] * kernel_size_[1] + out_channels_ * kernel_size_[0] * kernel_size_[1])); // Adjusted for Glorot uniform
    
    std::vector<T> host_weights(this->weights_.size());
    std::vector<T> host_bias(this->bias_.size());

    if constexpr (std::is_same<T, __half>::value) {
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (int i = 0; i < host_weights.size(); ++i) {
            host_weights[i] = __float2half(dist(gen));
        }
        std::fill(host_bias.begin(), host_bias.end(), __float2half(0.0f));
    } else {
        std::uniform_real_distribution<T> dist(-limit, limit);
        for (int i = 0; i < host_weights.size(); ++i) {
            host_weights[i] = dist(gen);
        }
        std::fill(host_bias.begin(), host_bias.end(), static_cast<T>(0.0f));
    }

    this->weights_.upload(host_weights.data());
    this->bias_.upload(host_bias.data());
}


// Explicit template instantiations
template class ConvLayer<float>;  // FP32
template class ConvLayer<__half>; // FP16
// template class ConvLayer<__nv_bfloat16>; // BF16

} // namespace dnn 
#include "fully_connected_layer.cuh"
#include "../utils/common.cuh"
#include "../core/device.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>
#include <sstream>

namespace dnn {

#ifndef ENABLE_CUDNN
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
            float inp = utils::to_float(input[batch * in_features + i]);
            float w = utils::to_float(weights[row * in_features + i]);
            sum += inp * w;
        }
        sum += bias ? utils::to_float(bias[row]) : 0.f;
        if constexpr (std::is_same<T, __half>::value)
            output[batch * out_features + row] = utils::to_half(sum);
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
            float dy = utils::to_float(dY[batch * out_features + j]);
            float w = utils::to_float(W[j * in_features + in_idx]);
            sum += dy * w;
        }
        if constexpr (std::is_same<T, __half>::value)
            dX[batch * in_features + in_idx] = utils::to_half(sum);
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
            float dy = utils::to_float(dY[n * out_features + out_idx]);
            float x = utils::to_float(X[n * in_features + in_idx]);
            sum += dy * x;
        }
        if constexpr (std::is_same<T, __half>::value)
            dW[out_idx * in_features + in_idx] = utils::to_half(sum);
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
            sum += utils::to_float(dY[n * out_features + out_idx]);
        if constexpr (std::is_same<T, __half>::value)
            db[out_idx] = utils::to_half(sum);
        else
            db[out_idx] = static_cast<T>(sum);
    }
}
#endif

template<typename TT>
FullyConnectedLayer<TT>::FullyConnectedLayer(int in_features, int out_features, bool training_enabled)
    : LayerWeightBias<TT>(
        tensor<TT>({ out_features, in_features }),
        tensor<TT>({ 1, out_features, 1, 1 }),
        tensor<TT>({ out_features, in_features }),
        tensor<TT>({ 1, out_features, 1, 1 }),
        training_enabled), 
    in_features_(in_features), 
    out_features_(out_features)
#ifdef ENABLE_CUDNN
    , filter_desc_(nullptr)
    , conv_desc_(nullptr)
    , input_desc_(nullptr)
    , output_desc_(nullptr)
#endif
{
#ifdef ENABLE_CUDNN
    utils::CHECK_CUDNN_EX(cudnnCreateFilterDescriptor(&filter_desc_));
    utils::CHECK_CUDNN_EX(cudnnCreateConvolutionDescriptor(&conv_desc_));
    utils::CHECK_CUDNN_EX(cudnnCreateTensorDescriptor(&input_desc_));
    utils::CHECK_CUDNN_EX(cudnnCreateTensorDescriptor(&output_desc_));

    // Set up filter descriptor for weights
    utils::CHECK_CUDNN_EX(cudnnSetFilter4dDescriptor(filter_desc_, 
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_features_, in_features_, 1, 1));

    // Set up convolution descriptor
    utils::CHECK_CUDNN_EX(cudnnSetConvolution2dDescriptor(conv_desc_,
        0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#endif

    initialize_weights();

#ifdef ENABLE_CUDNN
utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(bias_.desc(),
    CUDNN_TENSOR_NCHW, utils::dnn_type<TT>(), 1, out_features_, 1, 1));
utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(grad_bias_.desc(),
    CUDNN_TENSOR_NCHW, utils::dnn_type<TT>(), 1, out_features_, 1, 1));
#endif    
}

template<typename TT>
FullyConnectedLayer<TT>::~FullyConnectedLayer() {
#ifdef ENABLE_CUDNN
    if (filter_desc_) utils::CHECK_CUDNN_EX(cudnnDestroyFilterDescriptor(filter_desc_));
    if (conv_desc_) utils::CHECK_CUDNN_EX(cudnnDestroyConvolutionDescriptor(conv_desc_));
    if (input_desc_) utils::CHECK_CUDNN_EX(cudnnDestroyTensorDescriptor(input_desc_));
    if (output_desc_) utils::CHECK_CUDNN_EX(cudnnDestroyTensorDescriptor(output_desc_));
#endif
}

template<typename TT>
tensor<TT> FullyConnectedLayer<TT>::forward(const tensor<TT>& input) {
    if (this->training_enabled_) {
        input_cache_ = input.clone();
    }
#ifdef ENABLE_CUDNN
    int N = input.shape()[0];
    tensor<TT> output({ N, out_features_ });

    // Set up input and output descriptors
    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(input_desc_,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, in_features_, 1, 1));
    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(output_desc_,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, out_features_, 1, 1));

    const float alpha = 1.0f, beta = 0.0f;
    auto handle = Cuda::current().cudnn();

    // Forward pass using cuDNN convolution
    utils::CHECK_CUDNN_EX(cudnnConvolutionForward(handle,
        &alpha, input_desc_, input.data(),
        filter_desc_, this->weights_.data(),
        conv_desc_, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        nullptr, 0, &beta, output_desc_, output.data()));

    // Add bias
    utils::CHECK_CUDNN_EX(cudnnAddTensor(handle,
        &alpha, this->bias_.desc(), this->bias_.data(),
        &alpha, output_desc_, output.data()));

    return output;
#else
    int N = input.shape()[0];
    tensor<TT> output({ N, out_features_ });
    dim3 blockDim(256);
    dim3 gridDim((out_features_ + blockDim.x - 1) / blockDim.x, N);

    fully_connected_forward_kernel<TT><<<gridDim, blockDim>>>(
        input.data(),
        this->weights_.data(),
        this->bias_.data(),
        output.data(),
        N, in_features_, out_features_
    );

    utils::THROW_CUDA_EX();

    return output;
#endif
}

template<typename TT>
tensor<TT> FullyConnectedLayer<TT>::backward(const tensor<TT>& grad_output) {
    const tensor<TT>* used_input = nullptr;
    if (this->training_enabled_) {
        if (!input_cache_) throw std::runtime_error("FullyConnectedLayer: input_cache_ not set for backward");
        used_input = &(*input_cache_);
    } else {
        throw std::runtime_error("FullyConnectedLayer: backward called without input argument in stateless mode");
    }
    const int N = used_input->shape()[0];

#ifdef ENABLE_CUDNN
    auto& cuda = Cuda::current();
    auto handle_blas = cuda.cublas();
    auto handle_dnn = cuda.cudnn();

    int batch_size = used_input->shape()[0];

    tensor<TT> grad_input({ batch_size, in_features_ });

    // grad_input = grad_output * weights^T
    utils::CHECK_CUBLAS_EX(cublasGemmEx(
        handle_blas,
        CUBLAS_OP_T, CUBLAS_OP_N,  // A^T * B
        in_features_, batch_size, out_features_,
        utils::one<TT>(),
        this->weights_.data(), this->weights_.blas_type(), in_features_,       // lda
        grad_output.data(), grad_output.blas_type(), out_features_, // ldb
        utils::zero<TT>(),
        grad_input.data(), grad_input.blas_type(), in_features_,    // ldc
        utils::compute_type<TT>(), CUBLAS_GEMM_DEFAULT
    ));

    // grad_weights = grad_output^T * input
    utils::CHECK_CUBLAS_EX(cublasGemmEx(
        handle_blas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_features_, in_features_, batch_size,
        utils::one<TT>(),
        grad_output.data(), grad_output.blas_type(), batch_size,       // lda
        used_input->data(), used_input->blas_type(), in_features_,                 // ldb
        utils::zero<TT>(),
        this->grad_weights_.data(), this->grad_weights_.blas_type(), in_features_, // ldc
        utils::compute_type<TT>(), CUBLAS_GEMM_DEFAULT
    ));

    // grad_bias = reduce_sum(grad_output, axis=0)
    cudnnReduceTensorDescriptor_t reduce_desc;
    utils::CHECK_CUDNN_EX(cudnnCreateReduceTensorDescriptor(&reduce_desc));
    utils::CHECK_CUDNN_EX(cudnnSetReduceTensorDescriptor(
        reduce_desc,
        CUDNN_REDUCE_TENSOR_ADD,
        utils::dnn_type<TT>(),
        CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES
    ));

    size_t workspace_bytes = 0;
    utils::CHECK_CUDNN_EX(cudnnGetReductionWorkspaceSize(
        handle_dnn,
        reduce_desc,
        grad_output.desc(),
        this->grad_bias_.desc(),
        &workspace_bytes
    ));

    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        utils::CHECK_CUDA_EX(cudaMalloc(&workspace, workspace_bytes));
    }

    utils::CHECK_CUDNN_EX(cudnnReduceTensor(
        handle_dnn,
        reduce_desc,
        nullptr, 0,
        workspace, workspace_bytes,
        utils::one<TT>(),
        grad_output.desc(), grad_output.data(),
        utils::zero<TT>(),
        this->grad_bias_.desc(), this->grad_bias_.data()
    ));

    if (workspace) {
        cudaFree(workspace);
    }

    cudnnDestroyReduceTensorDescriptor(reduce_desc);

    return grad_input;
#else
    tensor<TT> grad_input({N, in_features_});

    // [2] Compute grad_input: dX = dY * W
    {
        dim3 blockDim(256);
        dim3 gridDim((in_features_ + blockDim.x - 1) / blockDim.x, N);
        fc_backward_input<TT><<<gridDim, blockDim>>>(
            grad_output.data(),
            this->weights_.data(),
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
        fc_backward_weights<TT><<<gridDim, blockDim>>>(
            grad_output.data(),
            used_input->data(),
            this->grad_weights_.data(),
            N,
            in_features_,
            out_features_
        );
    }

    // [4] Compute grad_bias: db = sum over batch of dY
    {
        dim3 blockDim(256);
        dim3 gridDim((out_features_ + blockDim.x - 1) / blockDim.x);
        fc_backward_bias<TT><<<gridDim, blockDim>>>(
            grad_output.data(),
            this->grad_bias_.data(),
            N,
            out_features_
        );
    }

    cudaDeviceSynchronize();
    utils::THROW_CUDA_EX();

    return grad_input;
#endif
}

template<typename TT>
void FullyConnectedLayer<TT>::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    // Xavier/Glorot initialization
    float limit = std::sqrt(6.0f / (in_features_ + out_features_)); // Adjusted for Glorot uniform
    
    std::vector<TT> host_weights(this->weights_.size());
    std::vector<TT> host_bias(this->bias_.size());

    if constexpr (std::is_same<TT, __half>::value) {
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (int i = 0; i < host_weights.size(); ++i) {
            host_weights[i] = __float2half(dist(gen));
        }
        std::fill(host_bias.begin(), host_bias.end(), __float2half(0.0f));
    } else {
        std::uniform_real_distribution<TT> dist(-limit, limit);
        for (int i = 0; i < host_weights.size(); ++i) {
            host_weights[i] = dist(gen);
        }
        std::fill(host_bias.begin(), host_bias.end(), static_cast<TT>(0.0f));
    }

    this->weights_.upload(host_weights.data());
    this->bias_.upload(host_bias.data());
}

template<typename TT>
void FullyConnectedLayer<TT>::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&in_features_), sizeof(in_features_));
    out.write(reinterpret_cast<const char*>(&out_features_), sizeof(out_features_));
    weights_.save(out);
    bias_.save(out);
}

template<typename TT>
void FullyConnectedLayer<TT>::load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&in_features_), sizeof(in_features_));
    in.read(reinterpret_cast<char*>(&out_features_), sizeof(out_features_));
    weights_.load(in);
    bias_.load(in);
}

// Explicit template instantiations
template class FullyConnectedLayer<float>;  // FP32
template class FullyConnectedLayer<__half>; // FP16
// template class FullyConnectedLayer<__nv_bfloat16>; // BF16

} // namespace dnn 
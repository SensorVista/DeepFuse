#include "flatten_layer.cuh"
#include "../utils/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <stdexcept>

namespace dnn {

template<typename T>
__global__ void flatten_forward_kernel(T* output, const T* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            output[idx] = input[idx];
        } else if constexpr (std::is_same_v<T, __half>) {
            output[idx] = input[idx];
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            output[idx] = input[idx];
        }
    }
}

template<typename T>
__global__ void flatten_backward_kernel(T* grad_input, const T* grad_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            grad_input[idx] = grad_output[idx];
        } else if constexpr (std::is_same_v<T, __half>) {
            grad_input[idx] = grad_output[idx];
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            grad_input[idx] = grad_output[idx];
        }
    }
}

template<typename T>
FlattenLayer<T>::FlattenLayer(bool training_enabled)
    : Layer<T>(training_enabled) {}

template<typename T>
tensor<T> FlattenLayer<T>::forward(const tensor<T>& input) {
    if (this->training_enabled_) {
        input_cache_ = input.clone();
    }
    int batch_size = input.shape()[0];
    int flat_dim   = input.size() / batch_size;
    tensor<T> output({batch_size, static_cast<int>(flat_dim)});
    int block_size = 256;
    int num_blocks = (input.size() + block_size - 1) / block_size;
    flatten_forward_kernel<T><<<num_blocks, block_size>>>(
        output.data(), input.data(), input.size());
    utils::THROW_CUDA_EX();
    return output;
}

template<typename T>
tensor<T> FlattenLayer<T>::backward(const tensor<T>& grad_output) {
    if (this->training_enabled_) {
        if (!input_cache_.has_value()) {
            throw std::runtime_error("FlattenLayer: input_cache_ is empty in backward().");
        }
    }
    const tensor<T>& input = this->training_enabled_ ? input_cache_.value() : grad_output; // fallback for stateless
    tensor<T> grad_input(input.shape());
    int size = grad_output.size();
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    flatten_backward_kernel<T><<<num_blocks, block_size>>>(grad_input.data(), grad_output.data(), size);
    utils::THROW_CUDA_EX();
    return grad_input;
}

// Explicit template instantiations
template class FlattenLayer<float>;  // FP32
template class FlattenLayer<__half>; // FP16
template class FlattenLayer<__nv_bfloat16>; // BF16

} // namespace dnn 
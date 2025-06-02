#include "dnn/layers/activation/tanh_layer.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>
#include <iostream>

namespace lenet5 {

template<typename T>
__global__ void tanh_forward_kernel(T* output, const T* input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = static_cast<T>(1.7159f * tanh(2.0f/3.0f * input[idx]));
    }
}

template<typename T>
__global__ void tanh_backward_kernel(T* grad_input, const T* grad_output, const T* input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // The output of the forward pass is A * tanh(S * x)
        // The derivative with respect to x is A * S * (1 - tanh(S * x)^2)
        // We have grad_output (dLoss/dOutput) and we need dLoss/dInput = dLoss/dOutput * dOutput/dInput
        T tanh_sx = tanh(2.0f/3.0f * input[idx]);
        T activation_derivative = static_cast<T>(1.7159f * (2.0f/3.0f) * (1.0f - tanh_sx * tanh_sx));
        grad_input[idx] = grad_output[idx] * activation_derivative;
    }
}

template<typename T>
tensor<T> TanhLayer<T>::forward(const tensor<T>& input) {
    tensor<T> output(input.shape());
    size_t size = input.size();
    if (size == 0) 
        return output;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
        
    tanh_forward_kernel<<<num_blocks, block_size>>>(output.data(), input.data(), size);

    utils::THROW_CUDA_EX();

    return output;
}

template<typename T>
tensor<T> TanhLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    tensor<T> grad_input(grad_output.shape());
    size_t size = grad_output.size();
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    tanh_backward_kernel<<<num_blocks, block_size>>>(
        grad_input.data(), grad_output.data(), input.data(), size);

    utils::THROW_CUDA_EX();

    return grad_input;
}

// Explicit template instantiations
template class TanhLayer<float>;  // FP32
// template class TanhLayer<__half>; // FP16
// template class TanhLayer<__nv_bfloat16>; // BF16

} // namespace lenet5 
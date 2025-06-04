#include "relu_layer.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>

namespace dnn {

// CUDA kernel for ReLU forward pass
template<typename T>
__global__ void relu_forward_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(static_cast<T>(0.0f), input[idx]);
    }
}

// CUDA kernel for ReLU backward pass
template<typename T>
__global__ void relu_backward_kernel(const T* grad_output, const T* input, T* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > static_cast<T>(0.0f)) ? grad_output[idx] : static_cast<T>(0.0f);
    }
}

// Forward pass
template<typename T>
tensor<T> ReLULayer<T>::forward(const tensor<T>& input) {
    tensor<T> output(input.shape());
    int total_size = input.size();

    if (total_size == 0) return output;

    const int BLOCK_SIZE = 256;
    int num_blocks = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    relu_forward_kernel<<<num_blocks, BLOCK_SIZE>>>(input.data(), output.data(), total_size);
    
    utils::THROW_CUDA_EX();

    return output;
}

// Backward pass
template<typename T>
tensor<T> ReLULayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    tensor<T> grad_input(input.shape());
    int total_size = input.size();

    if (total_size == 0) return grad_input;

    const int BLOCK_SIZE = 256;
    int num_blocks = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    relu_backward_kernel<<<num_blocks, BLOCK_SIZE>>>(grad_output.data(), input.data(), grad_input.data(), total_size);
    utils::THROW_CUDA_EX();

    return grad_input;
}

// Explicit template instantiations
template class ReLULayer<float>;

} // namespace dnn 
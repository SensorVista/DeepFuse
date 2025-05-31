#include "dnn/layers/activation/tanh_layer.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>
#include <iostream>

namespace lenet5 {

template<typename T>
__global__ void tanh_forward_kernel(T* output, const T* input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanh(input[idx]);
    }
}

template<typename T>
__global__ void tanh_backward_kernel(T* grad_input, const T* grad_output, const T* input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T tanh_x = tanh(input[idx]);
        grad_input[idx] = grad_output[idx] * (1.0f - tanh_x * tanh_x);
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
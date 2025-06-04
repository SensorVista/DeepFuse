#include "dnn/layers/reshape/flatten_layer.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace dnn {

template<typename T>
__global__ void flatten_forward_kernel(T* output, const T* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

template<typename T>
__global__ void flatten_backward_kernel(T* grad_input, const T* grad_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx];
    }
}

template<typename T>
tensor<T> FlattenLayer<T>::forward(const tensor<T>& input) {
    int batch_size = input.shape()[0];
    int flat_dim   = input.size() / batch_size;
    tensor<T> output({batch_size, static_cast<int>(flat_dim)});

    int block_size = 256;
    int num_blocks = (input.size() + block_size - 1) / block_size;
    flatten_forward_kernel<<<num_blocks, block_size>>>(
        output.data(), input.data(), input.size());

    utils::THROW_CUDA_EX();

    return output;
}

template<typename T>
tensor<T> FlattenLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    tensor<T> grad_input(input.shape());
    int size = grad_output.size();
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    flatten_backward_kernel<<<num_blocks, block_size>>>(grad_input.data(), grad_output.data(), size);

    utils::THROW_CUDA_EX();

    return grad_input;
}

// Explicit template instantiations
template class FlattenLayer<float>;  // FP32
// template class FlattenLayer<__half>; // FP16
// template class FlattenLayer<__nv_bfloat16>; // BF16

} // namespace dnn 
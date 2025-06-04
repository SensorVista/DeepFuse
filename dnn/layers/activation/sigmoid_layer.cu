#include "sigmoid_layer.cuh"
#include "../../utils/common.cuh"

#include <cuda_runtime.h>
#include <iostream>

template<typename T>
__global__ void sigmoid_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = T(1.0) / (T(1.0) + exp(-input[idx]));
    }
}

template<typename T>
__global__ void sigmoid_derivative_kernel(const T* grad_output, const T* input, T* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T sigmoid_x = T(1.0) / (T(1.0) + exp(-input[idx]));
        grad_input[idx] = grad_output[idx] * sigmoid_x * (T(1.0) - sigmoid_x);
    }
}

namespace dnn {

template<typename T>
tensor<T> SigmoidLayer<T>::forward(const tensor<T>& input) {
    // Sigmoid: f(x) = 1 / (1 + e^(-x))
    tensor<T> output(input.shape());
    
    // Launch CUDA kernel for sigmoid activation
    dim3 block(256);
    dim3 grid((input.size() + block.x - 1) / block.x);
    
    sigmoid_kernel<<<grid, block>>>(input.data(), output.data(), input.size());
    cudaDeviceSynchronize();
    
    return output;
}

template<typename T>
tensor<T> SigmoidLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    // Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
    tensor<T> grad_input(grad_output.shape());
    
    // Launch CUDA kernel for sigmoid derivative
    dim3 block(256);
    dim3 grid((grad_output.size() + block.x - 1) / block.x);
    
    sigmoid_derivative_kernel<<<grid, block>>>(grad_output.data(), input.data(), grad_input.data(), grad_output.size());
    cudaDeviceSynchronize();
    
    return grad_input;
}

// Explicit template instantiations
template class SigmoidLayer<float>;  // FP32
// template class SigmoidLayer<__half>; // FP16
// template class SigmoidLayer<__nv_bfloat16>; // BF16

} // namespace dnn

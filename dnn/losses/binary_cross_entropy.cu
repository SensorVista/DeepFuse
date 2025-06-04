#include "dnn/losses/binary_cross_entropy.cuh"

#include <cuda_runtime.h>

#include "dnn/utils/common.cuh"

namespace dnn {

template<typename T>
__global__ void binary_cross_entropy_kernel(const T* predictions, const T* targets, T* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // BCE = -(y * log(p) + (1-y) * log(1-p))
        // Add small epsilon to avoid log(0)
        const T epsilon = T(1e-7);
        T p = max(min(predictions[idx], T(1.0) - epsilon), epsilon);
        T y = targets[idx];
        loss[idx] = -(y * log(p) + (T(1.0) - y) * log(T(1.0) - p));
    }
}

template<typename T>
__global__ void binary_cross_entropy_gradient_kernel(const T* predictions, const T* targets, T* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const T epsilon = T(1e-7);  // avoid div by 0
        T p = max(min(predictions[idx], T(1.0) - epsilon), epsilon);
        T y = targets[idx];
        gradient[idx] = (p - y) / (p * (1 - p));
    }
}

template<typename T>
T BinaryCrossEntropyLoss<T>::compute(const tensor<T>& predictions, const tensor<T>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }

    tensor<T> loss({predictions.size()});
    
    // Launch kernel to compute loss for each element
    dim3 block(256);
    dim3 grid((predictions.size() + block.x - 1) / block.x);
    
    binary_cross_entropy_kernel<<<grid, block>>>(predictions.data(), targets.data(), loss.data(), predictions.size());
    
    cudaDeviceSynchronize();

    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());

    // Compute mean loss
    T total_loss = 0;
    std::vector<T> host_loss(loss.size());
    loss.download(host_loss.data());
    
    for (int i = 0; i < loss.size(); ++i) {
        total_loss += host_loss[i];
    }
    
    return total_loss / static_cast<T>(loss.size());
}

template<typename T>
tensor<T> BinaryCrossEntropyLoss<T>::compute_gradient(const tensor<T>& predictions, const tensor<T>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }

    tensor<T> gradient(predictions.shape());
    
    // Launch kernel to compute gradient for each element
    dim3 block(256);
    dim3 grid((predictions.size() + block.x - 1) / block.x);
    
    binary_cross_entropy_gradient_kernel<<<grid, block>>>(predictions.data(), targets.data(), gradient.data(), predictions.size());
    
    cudaDeviceSynchronize();
    
    return gradient;
}

// Explicit template instantiations
template class BinaryCrossEntropyLoss<float>;  // FP32
// template class BinaryCrossEntropyLoss<__half>; // FP16
// template class BinaryCrossEntropyLoss<__nv_bfloat16>; // BF16

} // namespace dnn

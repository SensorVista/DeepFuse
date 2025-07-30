#include "binary_cross_entropy.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <type_traits>
#include "dnn/utils/common.cuh"

namespace dnn {

template<typename T>
__global__ void binary_cross_entropy_kernel(const T* predictions, const T* targets, T* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // BCE = -(y * log(p) + (1-y) * log(1-p))
    // Add small epsilon to avoid log(0)
    if (idx < size) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            const T epsilon = T(1e-7);
            T p = fmax(fmin(predictions[idx], T(1.0) - epsilon), epsilon);
            T y = targets[idx];
            loss[idx] = -(y * log(p) + (T(1.0) - y) * log(T(1.0) - p));
        } else if constexpr (std::is_same_v<T, __half>) {
            float epsilon = 1e-7f;
            float p = fmaxf(fminf(__half2float(predictions[idx]), 1.0f - epsilon), epsilon);
            float y = __half2float(targets[idx]);
            loss[idx] = __float2half(-(y * logf(p) + (1.0f - y) * logf(1.0f - p)));
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float epsilon = 1e-7f;
            float p = fmaxf(fminf(__bfloat162float(predictions[idx]), 1.0f - epsilon), epsilon);
            float y = __bfloat162float(targets[idx]);
            loss[idx] = __float2bfloat16(-(y * logf(p) + (1.0f - y) * logf(1.0f - p)));
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            float epsilon = 1e-7f;
            float p = fmaxf(fminf(static_cast<float>(predictions[idx]), 1.0f - epsilon), epsilon);
            float y = static_cast<float>(targets[idx]);
            loss[idx] = static_cast<T>(-(y * logf(p) + (1.0f - y) * logf(1.0f - p)));
        }
    }
}

template<typename T>
__global__ void binary_cross_entropy_gradient_kernel(const T* predictions, const T* targets, T* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            const T epsilon = T(1e-7);
            T p = fmax(fmin(predictions[idx], T(1.0) - epsilon), epsilon);
            T y = targets[idx];
            gradient[idx] = (p - y) / (p * (T(1.0) - p));
        } else if constexpr (std::is_same_v<T, __half>) {
            float epsilon = 1e-7f;
            float p = fmaxf(fminf(__half2float(predictions[idx]), 1.0f - epsilon), epsilon);
            float y = __half2float(targets[idx]);
            gradient[idx] = __float2half((p - y) / (p * (1.0f - p)));
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float epsilon = 1e-7f;
            float p = fmaxf(fminf(__bfloat162float(predictions[idx]), 1.0f - epsilon), epsilon);
            float y = __bfloat162float(targets[idx]);
            gradient[idx] = __float2bfloat16((p - y) / (p * (1.0f - p)));
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            float epsilon = 1e-7f;
            float p = fmaxf(fminf(static_cast<float>(predictions[idx]), 1.0f - epsilon), epsilon);
            float y = static_cast<float>(targets[idx]);
            gradient[idx] = static_cast<T>((p - y) / (p * (1.0f - p)));
        }
    }
}

template<typename T>
T BinaryCrossEntropyLoss<T>::compute(const tensor<T>& predictions, const tensor<T>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    tensor<T> loss({predictions.size()});
    dim3 block(256);
    dim3 grid((predictions.size() + block.x - 1) / block.x);
    binary_cross_entropy_kernel<T><<<grid, block>>>(predictions.data(), targets.data(), loss.data(), predictions.size());
    cudaDeviceSynchronize();
    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());
    float total_loss = 0.0f;
    std::vector<T> host_loss(loss.size());
    loss.download(host_loss.data());
    for (int i = 0; i < loss.size(); ++i) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            total_loss += static_cast<float>(host_loss[i]);
        } else if constexpr (std::is_same_v<T, __half>) {
            total_loss += __half2float(host_loss[i]);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            total_loss += __bfloat162float(host_loss[i]);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            total_loss += static_cast<float>(host_loss[i]);
        }
    }
    return static_cast<T>(total_loss / static_cast<float>(loss.size()));
}

template<typename T>
tensor<T> BinaryCrossEntropyLoss<T>::compute_gradient(const tensor<T>& predictions, const tensor<T>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    tensor<T> gradient(predictions.shape());
    dim3 block(256);
    dim3 grid((predictions.size() + block.x - 1) / block.x);
    binary_cross_entropy_gradient_kernel<T><<<grid, block>>>(predictions.data(), targets.data(), gradient.data(), predictions.size());
    cudaDeviceSynchronize();
    return gradient;
}

// Explicit template instantiations

template class BinaryCrossEntropyLoss<float>;
template class BinaryCrossEntropyLoss<__half>;
template class BinaryCrossEntropyLoss<__nv_bfloat16>;
template class BinaryCrossEntropyLoss<int8_t>;
template class BinaryCrossEntropyLoss<uint8_t>;

} // namespace dnn

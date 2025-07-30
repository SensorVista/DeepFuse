#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <type_traits>

namespace dnn::utils {

// Templated CUDA kernel for gradient clipping
// Specialize for each type as needed

template<typename T>
__global__ void clip_grad_norm_kernel(T* grad, T norm, T max_norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            grad[idx] = grad[idx] * (max_norm / norm);
        } else if constexpr (std::is_same_v<T, __half>) {
            grad[idx] = __hmul(grad[idx], __hdiv(__float2half(static_cast<float>(max_norm)), __float2half(static_cast<float>(norm))));
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            grad[idx] = __hmul(grad[idx], __hdiv(__float2bfloat16(static_cast<float>(max_norm)), __float2bfloat16(static_cast<float>(norm))));
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            // For int8/uint8, do integer math (may lose precision)
            grad[idx] = static_cast<T>(static_cast<float>(grad[idx]) * (static_cast<float>(max_norm) / static_cast<float>(norm)));
        }
    }
}

template<typename T>
void clip_grad_norm(std::vector<tensor<T>*>& gradients, T max_norm) {
    using std::is_same_v;
    using std::sqrt;
    using std::vector;
    using std::abs;
    float total_norm = 0.0f;
    for (auto* grad_tensor : gradients) {
        vector<T> host_grad(grad_tensor->size());
        grad_tensor->download(host_grad.data());
        float sum_sq = 0.0f;
        for (T val : host_grad) {
            if constexpr (is_same_v<T, float> || is_same_v<T, double>) {
                sum_sq += static_cast<float>(val) * static_cast<float>(val);
            } else if constexpr (is_same_v<T, __half>) {
                float fval = __half2float(val);
                sum_sq += fval * fval;
            } else if constexpr (is_same_v<T, __nv_bfloat16>) {
                float fval = __bfloat162float(val);
                sum_sq += fval * fval;
            } else if constexpr (is_same_v<T, int8_t> || is_same_v<T, uint8_t>) {
                sum_sq += static_cast<float>(val) * static_cast<float>(val);
            }
        }
        total_norm += sum_sq;
    }
    total_norm = sqrt(total_norm);
    if (total_norm > static_cast<float>(max_norm)) {
        for (auto* grad_tensor : gradients) {
            int size = grad_tensor->size();
            int block_size = 256;
            int num_blocks = (size + block_size - 1) / block_size;
            clip_grad_norm_kernel<T><<<num_blocks, block_size>>>(
                grad_tensor->data(), static_cast<T>(total_norm), max_norm, size);
            CHECK_CUDA_EX(cudaDeviceSynchronize());
        }
    }
}

// Explicit instantiations

template void clip_grad_norm<float>(std::vector<tensor<float>*>&, float);
template void clip_grad_norm<__half>(std::vector<tensor<__half>*>&, __half);
template void clip_grad_norm<__nv_bfloat16>(std::vector<tensor<__nv_bfloat16>*>&, __nv_bfloat16);
template void clip_grad_norm<int8_t>(std::vector<tensor<int8_t>*>&, int8_t);
template void clip_grad_norm<uint8_t>(std::vector<tensor<uint8_t>*>&, uint8_t);

} // namespace dnn::utils
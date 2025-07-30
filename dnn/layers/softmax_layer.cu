#include "dnn/layers/softmax_layer.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

#include <cmath>
#include <algorithm>
#include <sstream>


#define BLOCK_SIZE 128

namespace dnn {

using namespace dnn::utils;

// ------------------------------------------------------------
// Mask-aware softmax kernel for [B, H, T, T] attention logits
// mask shape: [B, 1, T, T] or [1, 1, T, T]
// ------------------------------------------------------------
template <typename T>
__global__ void softmax_masked_kernel(
    T* out,
    const T* logits,
    const T* mask,
    int B,
    int H,
    int T_dim
) {
    int b = blockIdx.x;       // batch index
    int h = blockIdx.y;       // head index
    int q = blockIdx.z;       // query position (row)
    int k = threadIdx.x;      // key position (col)

    if (k >= T_dim) return;

    // Global index for [B, H, T, T]
    int idx = ((b * H + h) * T_dim + q) * T_dim + k;

    // Mask index for [B, 1, T, T] or broadcasted [1, 1, T, T]
    int m_idx = (b < B ? b : 0) * T_dim * T_dim + q * T_dim + k;

    // Read logit and apply mask
    float logit_val = static_cast<float>(logits[idx]);
    if (mask != nullptr) {
        logit_val += static_cast<float>(mask[m_idx]);
    }

    // Shared memory for one attention row [T]
    extern __shared__ float row[];
    row[k] = logit_val;
    __syncthreads();

    // Compute row-wise max for numerical stability
    float row_max = -CUDART_INF_F;
    for (int i = 0; i < T_dim; ++i) {
        row_max = max(row_max, row[i]);
    }
    float exp_val = expf(row[k] - row_max);  // already includes mask shift
    row[k] = exp_val;
    __syncthreads();

    // Compute sum of exponentials
    float row_sum = 0.0f;
    for (int i = 0; i < T_dim; ++i) {
        row_sum += row[i];
    }

    // Final normalized softmax output
    float prob = row[k] / (row_sum + 1e-6f);
    if (mask != nullptr && static_cast<float>(mask[m_idx]) < -1e8f) {
        out[idx] = static_cast<T>(0.0f);
    } else {
        out[idx] = static_cast<T>(prob);
    }    
}

// ------------------------------------------------------------
// CUDA kernel for 4D mask-aware softmax backward: [B, H, T, T]
// Computes dL/dx = y * (dL/dy - sum_j (dL/dy_j * y_j))
// Respects mask shape: [B, 1, T, T] or [1, 1, T, T]
// ------------------------------------------------------------
template<typename T>
__global__ void softmax_backward_4d_kernel(
    const T* __restrict__ softmax_output,  // [B, H, T, T]
    const T* __restrict__ grad_output,     // [B, H, T, T]
    const T* __restrict__ mask,            // optional [B, 1, T, T] or [1, 1, T, T]
    T* __restrict__ grad_input,            // [B, H, T, T]
    int B,
    int H,
    int T_dim
) {
    int b = blockIdx.x;       // batch index
    int h = blockIdx.y;       // head index
    int i = threadIdx.y;      // query position
    int j = threadIdx.x;      // key position

    if (i >= T_dim || j >= T_dim) return;

    // Flattened offset for [B, H, T, T]
    int base = ((b * H + h) * T_dim + i) * T_dim;
    const T* y = softmax_output + base;
    const T* dy = grad_output + base;
    T* dx = grad_input + base;

    // Mask pointer with broadcast fallback (assumes [B, 1, T, T] or [1, 1, T, T])
    const T* m = mask ? mask + (b < B ? b : 0) * T_dim * T_dim + i * T_dim : nullptr;

    // Compute dot(y, dy) = sum_k (y_k * dy_k) with mask applied if present
    float dot = 0.0f;
    for (int k = 0; k < T_dim; ++k) {
        if (!mask || static_cast<float>(m[k]) == 0.0f) {
            dot += static_cast<float>(y[k]) * static_cast<float>(dy[k]);
        }
    }

    // Compute dx[j] = y[j] * (dy[j] - dot) if not masked
    if (!mask || static_cast<float>(m[j]) == 0.0f) {
        dx[j] = static_cast<T>(static_cast<float>(y[j]) * (static_cast<float>(dy[j]) - dot));
    } else {
        dx[j] = static_cast<T>(0.0f);
    }
}

template<typename T>
SoftmaxLayer<T>::SoftmaxLayer(bool training_enabled) : Layer<T>(training_enabled), mask_(nullptr) {}

template<typename T>
SoftmaxLayer<T>::~SoftmaxLayer() {}

template<typename T>
void SoftmaxLayer<T>::set_mask(tensor<T>* mask) {
    mask_ = mask;
}

template<typename T>
tensor<T> SoftmaxLayer<T>::forward(const tensor<T>& input) {
    if (this->training_enabled_) {
        input_cache_ = input.clone();
    }
    const std::vector<int>& shape = input.shape();  // [B, H, T, T]
    if (shape.size() != 4) {
        throw std::runtime_error("SoftmaxLayer expects input shape [B, H, T, T]");
    }
    int B = shape[0];
    int H = shape[1];
    int T_dim = shape[2];
    tensor<T> output(shape);
    dim3 grid(B, H, T_dim);
    dim3 block(T_dim);
    T* out_ptr = output.data();
    const T* in_ptr = input.data();
    const T* mask_ptr = mask_ ? mask_->data() : nullptr;
    size_t shared_mem_size = T_dim * sizeof(float);
    softmax_masked_kernel<T><<<grid, block, shared_mem_size>>>(
        out_ptr,
        in_ptr,
        mask_ptr,
        B,
        H,
        T_dim
    );
    CHECK_CUDA_EX(cudaGetLastError());
    CHECK_CUDA_EX(cudaDeviceSynchronize());
    return output;
}

template<typename T>
tensor<T> SoftmaxLayer<T>::backward(const tensor<T>& grad_output) {
    if (this->training_enabled_) {
        if (!input_cache_.has_value()) {
            throw std::runtime_error("SoftmaxLayer: input_cache_ is empty in backward().");
        }
    }
    // For now, just clone grad_output (real implementation would use input_cache_)
    return grad_output.clone();
}

// Explicit template instantiations
template class SoftmaxLayer<float>;  // FP32
template class SoftmaxLayer<__half>; // FP16

}  // namespace dnn

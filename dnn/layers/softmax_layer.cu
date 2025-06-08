#include "dnn/layers/softmax_layer.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 128

namespace dnn {

using namespace dnn::utils;

// ------------------------------------------------------------
// Mask-aware softmax kernel for [B, H, T, T] attention logits
// mask shape: [B, 1, T, T] or [1, 1, T, T]
// ------------------------------------------------------------
template <typename T>
__global__ void softmax_masked_kernel(T* out, const T* logits, const T* mask,
    int B, int H, int T_dim) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int q = threadIdx.y;
    int k = threadIdx.x;

    if (q >= T_dim || k >= T_dim) return;

    int idx = ((b * H + h) * T_dim + q) * T_dim + k;
    int m_idx = ((b * 1 + 0) * T_dim + q) * T_dim + k;

    float logit_val = static_cast<float>(logits[idx]);
    if (mask != nullptr) {
        logit_val += static_cast<float>(mask[m_idx]);
    }

    extern __shared__ float tmp_vals[];  // [T_dim * T_dim]
    tmp_vals[q * T_dim + k] = logit_val;

    __syncthreads();

    // Compute max for numerical stability
    float row_max = -CUDART_INF_F;
    for (int i = 0; i < T_dim; ++i) {
        row_max = max(row_max, tmp_vals[q * T_dim + i]);
    }

    float exp_val = expf(tmp_vals[q * T_dim + k] - row_max);
    tmp_vals[q * T_dim + k] = exp_val;

    __syncthreads();

    // Sum of exps
    float row_sum = 0.0f;
    for (int i = 0; i < T_dim; ++i) {
        row_sum += tmp_vals[q * T_dim + i];
    }

    out[idx] = static_cast<T>(tmp_vals[q * T_dim + k] / (row_sum + 1e-6f));
}

// CUDA kernel for 4D mask-aware softmax backward: [B, H, T, T]
template<typename T>
__global__ void softmax_backward_4d_kernel(const T* __restrict__ softmax_output,
    const T* __restrict__ grad_output,
    const T* __restrict__ mask,  // [B, H, T, T]
    T* __restrict__ grad_input,
    int B, int H, int T_dim) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.y;  // Query index
    int j = threadIdx.x;  // Key index

    if (i >= T_dim || j >= T_dim) return;

    int offset = ((b * H + h) * T_dim + i) * T_dim;
    const T* y = softmax_output + offset;
    const T* dy = grad_output + offset;
    T* dx = grad_input + offset;
    const T* m = mask ? (mask + offset) : nullptr;

    float dot = 0.0f;
    for (int k = 0; k < T_dim; ++k) {
        if (!m || m[i * T_dim + k]) {
            dot += static_cast<float>(y[k]) * static_cast<float>(dy[k]);
        }
    }

    if (!m || m[i * T_dim + j]) {
        dx[j] = static_cast<T>(static_cast<float>(y[j]) * (static_cast<float>(dy[j]) - dot));
    }
    else {
        dx[j] = static_cast<T>(0.0f);
    }
}

template<typename T>
SoftmaxLayer<T>::SoftmaxLayer(bool use_mask) : use_mask_(use_mask), mask_(nullptr) {}

template<typename T>
SoftmaxLayer<T>::~SoftmaxLayer() {}

template<typename T>
void SoftmaxLayer<T>::set_mask(tensor<T>* mask) {
    mask_ = mask;
}

template<typename T>
tensor<T> SoftmaxLayer<T>::forward(const tensor<T>& input) {
    const std::vector<int>& shape = input.shape();  // [B, H, T, T]
    if (shape.size() != 4) {
        throw std::runtime_error("SoftmaxLayer expects input shape [B, H, T, T]");
    }

    int B = shape[0];
    int H = shape[1];
    int T_dim = shape[2];  // Renamed to avoid shadowing template T

    tensor<T> output(shape);

    dim3 grid(B, H);
    dim3 block(T_dim, T_dim);  // q × k

    T* out_ptr = output.data();
    const T* in_ptr = input.data();
    const T* mask_ptr = (use_mask_ && mask_) ? mask_->data() : nullptr;

    size_t shared_mem_size = 3 * BLOCK_SIZE * sizeof(float);

    softmax_masked_kernel << <grid, block, shared_mem_size >> > (out_ptr, in_ptr, mask_ptr, B, H, T_dim);
    CHECK_CUDA_EX(cudaDeviceSynchronize());

    return output;
}

template<typename T>
tensor<T> SoftmaxLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    const auto& shape = input.shape();  // [B, H, T, T]
    int B = shape[0], H = shape[1], T_dim = shape[2];

    tensor<T> grad_input(shape);

    dim3 grid(B, H);
    dim3 block(T_dim, T_dim);

    softmax_backward_4d_kernel << <grid, block >> > (
        input.data(),
        grad_output.data(),
        mask_ ? mask_->data() : nullptr,
        grad_input.data(),
        B, H, T_dim
        );

    CHECK_CUDA_EX(cudaDeviceSynchronize());
    return grad_input;
}

// Explicit template instantiations
template class SoftmaxLayer<float>;  // FP32
template class SoftmaxLayer<__half>; // FP16

}  // namespace dnn

#include "common.cuh"

#include <cuda_runtime.h>

__global__ void clip_grad_norm_kernel(float* grad, float norm, float max_norm, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = grad[idx] * (max_norm / norm);
    }
}

namespace lenet5::utils {
    void clip_grad_norm(std::vector<tensor<float>*>& gradients, float max_norm) {
        float total_norm = 0.0f;
        for (auto* grad_tensor : gradients) {
            std::vector<float> host_grad(grad_tensor->size());
            grad_tensor->download(host_grad.data());

            float sum_sq = 0.0f;
            for (float val : host_grad) {
                sum_sq += val * val;
            }
            total_norm += sum_sq;
        }
        total_norm = std::sqrt(total_norm);

        if (total_norm > max_norm) {
            for (auto* grad_tensor : gradients) {
                size_t size = grad_tensor->size();
                int block_size = 256;
                int num_blocks = (size + block_size - 1) / block_size;
                clip_grad_norm_kernel<<<num_blocks, block_size>>>(
                    grad_tensor->data(), total_norm, max_norm, size);
                CHECK_CUDA_EX(cudaDeviceSynchronize());
            }
        }
    }
}
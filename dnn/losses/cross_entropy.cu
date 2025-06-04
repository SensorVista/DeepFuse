#include "cross_entropy.cuh"
#include "../utils/common.cuh"

#include <cuda_runtime.h>

namespace dnn {

template<typename T>
__global__ void cross_entropy_log_softmax_kernel(T* loss, const T* logits, const T* targets, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        const T* logit_row = logits + idx * num_classes;
        const T* target_row = targets + idx * num_classes;

        // 1. Find max logit
        T max_logit = logit_row[0];
        for (int j = 1; j < num_classes; ++j)
            if (logit_row[j] > max_logit)
                max_logit = logit_row[j];

        // 2. Compute log(sum(exp(logits - max)))
        T sum_exp = 0;
        for (int j = 0; j < num_classes; ++j)
            sum_exp += expf(logit_row[j] - max_logit);

        T log_sum_exp = logf(sum_exp + 1e-7f);  // epsilon added here

        // 3. Final loss = - SUM target * (logit - max - log_sum)
        T total = 0;
        for (int j = 0; j < num_classes; ++j)
            total += target_row[j] * (logit_row[j] - max_logit - log_sum_exp);

        loss[idx] = -total;
    }
}

template<typename T>
__global__ void cross_entropy_gradient_logit_kernel(T* grad, const T* logits, const T* targets, int batch_size, int num_classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        const T* logit_row = logits + i * num_classes;
        const T* target_row = targets + i * num_classes;
        T* grad_row = grad + i * num_classes;

        // Find max logit for numerical stability
        T max_logit = logit_row[0];
        for (int j = 1; j < num_classes; ++j) {
            if (logit_row[j] > max_logit)
                max_logit = logit_row[j];
        }

        // Compute softmax denominator
        T sum_exp = 0;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += expf(logit_row[j] - max_logit);
        }

        // Compute softmax and subtract target
        for (int j = 0; j < num_classes; ++j) {
            T softmax_j = expf(logit_row[j] - max_logit) / (sum_exp + 1e-7f);
            grad_row[j] = softmax_j - target_row[j];
        }
    }
}

template<typename T>
T CrossEntropyLoss<T>::compute(const tensor<T>& logits, const tensor<T>& targets) {
    int batch_size = logits.shape()[0];
    int num_classes = logits.shape()[1];

    tensor<T> losses({ batch_size });

    int block_size = 128;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    cross_entropy_log_softmax_kernel << <num_blocks, block_size >> > (
        losses.data(),
        logits.data(),
        targets.data(),
        batch_size,
        num_classes
        );

    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());

    std::vector<T> host_losses(batch_size);
    losses.download(host_losses.data());

    T sum = 0;
    for (T v : host_losses) sum += v;
    return sum / batch_size;
}


template<typename T>
tensor<T> CrossEntropyLoss<T>::compute_gradient(const tensor<T>& logits, const tensor<T>& targets) {
    int batch_size = logits.shape()[0];
    int num_classes = logits.shape()[1];
    tensor<T> grad(logits.shape());

    int block_size = 128;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    cross_entropy_gradient_logit_kernel << <num_blocks, block_size >> > (
        grad.data(),
        logits.data(),
        targets.data(),
        batch_size,
        num_classes
        );

    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());
    return grad;
}

// Explicit template instantiations
template class CrossEntropyLoss<float>;  // FP32
// template class CrossEntropyLoss<__half>; // FP16
// template class CrossEntropyLoss<__nv_bfloat16>; // BF16

} // namespace dnn 
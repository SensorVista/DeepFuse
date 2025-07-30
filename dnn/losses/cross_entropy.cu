#include "cross_entropy.cuh"
#include "../utils/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <type_traits>

namespace dnn {

template<typename T>
__global__ void cross_entropy_log_softmax_kernel(T* loss, const T* logits, const T* targets, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        const T* logit_row = logits + idx * num_classes;
        const T* target_row = targets + idx * num_classes;
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            T max_logit = logit_row[0];
            for (int j = 1; j < num_classes; ++j)
                if (logit_row[j] > max_logit)
                    max_logit = logit_row[j];
            T sum_exp = 0;
            for (int j = 0; j < num_classes; ++j)
                sum_exp += expf(logit_row[j] - max_logit);
            T log_sum_exp = logf(sum_exp + 1e-7f);
            T total = 0;
            for (int j = 0; j < num_classes; ++j)
                total += target_row[j] * (logit_row[j] - max_logit - log_sum_exp);
            loss[idx] = -total;
        } else if constexpr (std::is_same_v<T, __half>) {
            float max_logit = __half2float(logit_row[0]);
            for (int j = 1; j < num_classes; ++j) {
                float val = __half2float(logit_row[j]);
                if (val > max_logit) max_logit = val;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j)
                sum_exp += expf(__half2float(logit_row[j]) - max_logit);
            float log_sum_exp = logf(sum_exp + 1e-7f);
            float total = 0.0f;
            for (int j = 0; j < num_classes; ++j)
                total += __half2float(target_row[j]) * (__half2float(logit_row[j]) - max_logit - log_sum_exp);
            loss[idx] = __float2half(-total);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float max_logit = __bfloat162float(logit_row[0]);
            for (int j = 1; j < num_classes; ++j) {
                float val = __bfloat162float(logit_row[j]);
                if (val > max_logit) max_logit = val;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j)
                sum_exp += expf(__bfloat162float(logit_row[j]) - max_logit);
            float log_sum_exp = logf(sum_exp + 1e-7f);
            float total = 0.0f;
            for (int j = 0; j < num_classes; ++j)
                total += __bfloat162float(target_row[j]) * (__bfloat162float(logit_row[j]) - max_logit - log_sum_exp);
            loss[idx] = __float2bfloat16(-total);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            float max_logit = static_cast<float>(logit_row[0]);
            for (int j = 1; j < num_classes; ++j) {
                float val = static_cast<float>(logit_row[j]);
                if (val > max_logit) max_logit = val;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j)
                sum_exp += expf(static_cast<float>(logit_row[j]) - max_logit);
            float log_sum_exp = logf(sum_exp + 1e-7f);
            float total = 0.0f;
            for (int j = 0; j < num_classes; ++j)
                total += static_cast<float>(target_row[j]) * (static_cast<float>(logit_row[j]) - max_logit - log_sum_exp);
            loss[idx] = static_cast<T>(-total);
        }
    }
}

template<typename T>
__global__ void cross_entropy_gradient_logit_kernel(T* grad, const T* logits, const T* targets, int batch_size, int num_classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        const T* logit_row = logits + i * num_classes;
        const T* target_row = targets + i * num_classes;
        T* grad_row = grad + i * num_classes;
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            T max_logit = logit_row[0];
            for (int j = 1; j < num_classes; ++j) {
                if (logit_row[j] > max_logit)
                    max_logit = logit_row[j];
            }
            T sum_exp = 0;
            for (int j = 0; j < num_classes; ++j) {
                sum_exp += expf(logit_row[j] - max_logit);
            }
            for (int j = 0; j < num_classes; ++j) {
                T softmax_j = expf(logit_row[j] - max_logit) / (sum_exp + 1e-7f);
                grad_row[j] = softmax_j - target_row[j];
            }
        } else if constexpr (std::is_same_v<T, __half>) {
            float max_logit = __half2float(logit_row[0]);
            for (int j = 1; j < num_classes; ++j) {
                float val = __half2float(logit_row[j]);
                if (val > max_logit) max_logit = val;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j)
                sum_exp += expf(__half2float(logit_row[j]) - max_logit);
            for (int j = 0; j < num_classes; ++j) {
                float softmax_j = expf(__half2float(logit_row[j]) - max_logit) / (sum_exp + 1e-7f);
                grad_row[j] = __float2half(softmax_j - __half2float(target_row[j]));
            }
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float max_logit = __bfloat162float(logit_row[0]);
            for (int j = 1; j < num_classes; ++j) {
                float val = __bfloat162float(logit_row[j]);
                if (val > max_logit) max_logit = val;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j)
                sum_exp += expf(__bfloat162float(logit_row[j]) - max_logit);
            for (int j = 0; j < num_classes; ++j) {
                float softmax_j = expf(__bfloat162float(logit_row[j]) - max_logit) / (sum_exp + 1e-7f);
                grad_row[j] = __float2bfloat16(softmax_j - __bfloat162float(target_row[j]));
            }
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            float max_logit = static_cast<float>(logit_row[0]);
            for (int j = 1; j < num_classes; ++j) {
                float val = static_cast<float>(logit_row[j]);
                if (val > max_logit) max_logit = val;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j)
                sum_exp += expf(static_cast<float>(logit_row[j]) - max_logit);
            for (int j = 0; j < num_classes; ++j) {
                float softmax_j = expf(static_cast<float>(logit_row[j]) - max_logit) / (sum_exp + 1e-7f);
                grad_row[j] = static_cast<T>(softmax_j - static_cast<float>(target_row[j]));
            }
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
    cross_entropy_log_softmax_kernel<T><<<num_blocks, block_size>>>(
        losses.data(),
        logits.data(),
        targets.data(),
        batch_size,
        num_classes
    );
    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());
    float sum = 0.0f;
    std::vector<T> host_losses(batch_size);
    losses.download(host_losses.data());
    for (int i = 0; i < batch_size; ++i) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            sum += static_cast<float>(host_losses[i]);
        } else if constexpr (std::is_same_v<T, __half>) {
            sum += __half2float(host_losses[i]);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            sum += __bfloat162float(host_losses[i]);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            sum += static_cast<float>(host_losses[i]);
        }
    }
    return static_cast<T>(sum / static_cast<float>(batch_size));
}

template<typename T>
tensor<T> CrossEntropyLoss<T>::compute_gradient(const tensor<T>& logits, const tensor<T>& targets) {
    int batch_size = logits.shape()[0];
    int num_classes = logits.shape()[1];
    tensor<T> grad(logits.shape());
    int block_size = 128;
    int num_blocks = (batch_size + block_size - 1) / block_size;
    cross_entropy_gradient_logit_kernel<T><<<num_blocks, block_size>>>(
        grad.data(),
        logits.data(),
        targets.data(),
        batch_size,
        num_classes
    );
    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());
    return grad;
}

template<typename T>
T CrossEntropyLoss<T>::compute(const tensor<T>& logits, const tensor<int>& target_indices) {
    int num_classes = logits.shape().back();
    auto one_hot = utils::to_one_hot<T>(target_indices, num_classes);
    return compute(logits, one_hot);
}

template<typename T>
tensor<T> CrossEntropyLoss<T>::compute_gradient(const tensor<T>& logits, const tensor<int>& target_indices) {
    int num_classes = logits.shape().back();
    auto one_hot = utils::to_one_hot<T>(target_indices, num_classes);
    return compute_gradient(logits, one_hot);
}

// Explicit template instantiations

template class CrossEntropyLoss<float>;
template class CrossEntropyLoss<__half>;
template class CrossEntropyLoss<__nv_bfloat16>;
template class CrossEntropyLoss<int8_t>;
template class CrossEntropyLoss<uint8_t>;

} // namespace dnn 
#include "adam_optimizer.cuh"
#include "../utils/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <type_traits>

namespace dnn {

template<typename T>
__global__ void adam_update_kernel(
    T* param, const T* grad,
    T* m, T* v,
    T beta1, T beta2,
    T learning_rate,
    T epsilon,
    T weight_decay,
    int timestep,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            T g = grad[idx] + weight_decay * param[idx];
            m[idx] = beta1 * m[idx] + (T(1.0) - beta1) * g;
            v[idx] = beta2 * v[idx] + (T(1.0) - beta2) * g * g;
            T m_hat = m[idx] / (T(1.0) - pow(beta1, timestep));
            T v_hat = v[idx] / (T(1.0) - pow(beta2, timestep));
            param[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        } else if constexpr (std::is_same_v<T, __half>) {
            float g = __half2float(grad[idx]) + __half2float(weight_decay) * __half2float(param[idx]);
            float m_val = __half2float(m[idx]);
            float v_val = __half2float(v[idx]);
            float beta1_f = __half2float(beta1);
            float beta2_f = __half2float(beta2);
            float lr = __half2float(learning_rate);
            float eps = __half2float(epsilon);
            float wd = __half2float(weight_decay);
            m_val = beta1_f * m_val + (1.0f - beta1_f) * g;
            v_val = beta2_f * v_val + (1.0f - beta2_f) * g * g;
            float m_hat = m_val / (1.0f - powf(beta1_f, timestep));
            float v_hat = v_val / (1.0f - powf(beta2_f, timestep));
            param[idx] = __float2half(__half2float(param[idx]) - lr * m_hat / (sqrtf(v_hat) + eps));
            m[idx] = __float2half(m_val);
            v[idx] = __float2half(v_val);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float g = __bfloat162float(grad[idx]) + __bfloat162float(weight_decay) * __bfloat162float(param[idx]);
            float m_val = __bfloat162float(m[idx]);
            float v_val = __bfloat162float(v[idx]);
            float beta1_f = __bfloat162float(beta1);
            float beta2_f = __bfloat162float(beta2);
            float lr = __bfloat162float(learning_rate);
            float eps = __bfloat162float(epsilon);
            float wd = __bfloat162float(weight_decay);
            m_val = beta1_f * m_val + (1.0f - beta1_f) * g;
            v_val = beta2_f * v_val + (1.0f - beta2_f) * g * g;
            float m_hat = m_val / (1.0f - powf(beta1_f, timestep));
            float v_hat = v_val / (1.0f - powf(beta2_f, timestep));
            param[idx] = __float2bfloat16(__bfloat162float(param[idx]) - lr * m_hat / (sqrtf(v_hat) + eps));
            m[idx] = __float2bfloat16(m_val);
            v[idx] = __float2bfloat16(v_val);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            float g = static_cast<float>(grad[idx]) + static_cast<float>(weight_decay) * static_cast<float>(param[idx]);
            float m_val = static_cast<float>(m[idx]);
            float v_val = static_cast<float>(v[idx]);
            float beta1_f = static_cast<float>(beta1);
            float beta2_f = static_cast<float>(beta2);
            float lr = static_cast<float>(learning_rate);
            float eps = static_cast<float>(epsilon);
            float wd = static_cast<float>(weight_decay);
            m_val = beta1_f * m_val + (1.0f - beta1_f) * g;
            v_val = beta2_f * v_val + (1.0f - beta2_f) * g * g;
            float m_hat = m_val / (1.0f - powf(beta1_f, timestep));
            float v_hat = v_val / (1.0f - powf(beta2_f, timestep));
            param[idx] = static_cast<T>(static_cast<float>(param[idx]) - lr * m_hat / (sqrtf(v_hat) + eps));
            m[idx] = static_cast<T>(m_val);
            v[idx] = static_cast<T>(v_val);
        }
    }
}

template<typename T>
void AdamOptimizer<T>::step() {
    ++timestep_;
    for (int i = 0; i < this->parameters_.size(); ++i) {
        auto* param = this->parameters_[i];
        auto* grad = this->gradients_[i];
        auto* m_buf = m_.at(param).get();
        auto* v_buf = v_.at(param).get();
        int size = param->size();
        int block_size = 256;
        int num_blocks = (size + block_size - 1) / block_size;
        adam_update_kernel<T><<<num_blocks, block_size>>>(
            param->data(),
            grad->data(),
            m_buf->data(),
            v_buf->data(),
            beta1_,
            beta2_,
            learning_rate_,
            epsilon_,
            weight_decay_,
            timestep_,
            size
        );
        utils::THROW_CUDA_EX();
    }
}

template<typename T>
void AdamOptimizer<T>::zero_grad() {
    for (auto* grad : this->gradients_) {
        grad->fill(0);
    }
}

template<typename T>
void AdamOptimizer<T>::update_parameters(const std::vector<tensor<T>*>& parameters, const std::vector<tensor<T>*>& gradients) {
    Optimizer<T>::update_parameters(parameters, gradients);
    for (auto* param : parameters) {
        if (m_.find(param) == m_.end()) {
            auto m_buf = std::make_unique<tensor<T>>(param->shape());
            auto v_buf = std::make_unique<tensor<T>>(param->shape());
            m_buf->fill(0);
            v_buf->fill(0);
            m_.emplace(param, std::move(m_buf));
            v_.emplace(param, std::move(v_buf));
        }
    }
}

// Explicit instantiations

template class AdamOptimizer<float>;
template class AdamOptimizer<__half>;
template class AdamOptimizer<__nv_bfloat16>;
template class AdamOptimizer<int8_t>;
template class AdamOptimizer<uint8_t>;

} // namespace dnn

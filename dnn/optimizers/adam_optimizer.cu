#include "adam_optimizer.cuh"
#include "../utils/common.cuh"
#include <cuda_runtime.h>

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
        T g = grad[idx] + weight_decay * param[idx];

        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0 - beta1) * g;

        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0 - beta2) * g * g;

        // Compute bias-corrected first and second moment
        T m_hat = m[idx] / (1.0 - pow(beta1, timestep));
        T v_hat = v[idx] / (1.0 - pow(beta2, timestep));

        // Update parameters
        param[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
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

        adam_update_kernel<<<num_blocks, block_size>>>(
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
// template class AdamOptimizer<__half>;

} // namespace dnn

#include "sgd_optimizer.cuh"
#include "../utils/common.cuh"

#include <cuda_runtime.h>

namespace dnn {

template<typename T>
__global__ void sgd_update_kernel(
    T* param,           // Parameter to update
    const T* grad,      // Gradient
    T* velocity,        // Velocity buffer
    T learning_rate,    // Learning rate
    T momentum,         // Momentum coefficient
    T weight_decay,     // Weight decay coefficient
    int size            // Size of the parameter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply weight decay
        T grad_with_decay = grad[idx] + weight_decay * param[idx];
        
        // Update velocity
        velocity[idx] = momentum * velocity[idx] - learning_rate * grad_with_decay;
        
        // Update parameter
        param[idx] += velocity[idx];
    }
}

// param = param + velocity
// velocity = momentum * velocity - lr * (grad + weight_decay * param)

template<typename T>
void SGDOptimizer<T>::step() {
    for (int i = 0; i < this->parameters_.size(); ++i) {
        auto* param = this->parameters_[i];
        auto* grad = this->gradients_[i];
        auto* velocity = velocities_.at(param).get();

        int size = param->size();
        int block_size = 256;
        int num_blocks = (size + block_size - 1) / block_size;
    
        sgd_update_kernel<<<num_blocks, block_size>>>(
            param->data(),
            grad->data(),
            velocity->data(),
            learning_rate_,
            momentum_,
            weight_decay_,
            size
        );

        utils::THROW_CUDA_EX();
    }
}

template<typename T>
void SGDOptimizer<T>::zero_grad() {
    for (auto* grad : this->gradients_) {
        grad->fill(0);
    }
}

template<typename T>
void SGDOptimizer<T>::update_parameters(const std::vector<tensor<T>*>& parameters, const std::vector<tensor<T>*>& gradients) {
    Optimizer<T>::update_parameters(parameters, gradients);

    for (auto* param : parameters) {
        if (velocities_.find(param) == velocities_.end()) {
            auto velocity = std::make_unique<tensor<T>>(param->shape());
            velocity->fill(0);
            velocities_.emplace(param, std::move(velocity));
        }
    }
}

// Explicit template instantiations
template class SGDOptimizer<float>;  // FP32
// template class SGDOptimizer<__half>;

} // namespace dnn 
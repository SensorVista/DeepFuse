#pragma once

#include "dnn/optimizers/optimizer.cuh"
#include <memory>
#include <unordered_map>

namespace dnn {

// Adam optimizer maintains per-parameter first and second moments (m, v) of gradients
template<typename T>
class AdamOptimizer : public Optimizer<T> {
public:
    AdamOptimizer(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8, T weight_decay = 0.0)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2),
          epsilon_(epsilon), weight_decay_(weight_decay), timestep_(0) {}

    void step() override;
    void zero_grad() override;
    void update_parameters(
        const std::vector<tensor<T>*>& parameters,
        const std::vector<tensor<T>*>& gradients) override;

    T learning_rate() const override { return learning_rate_; }
    void set_learning_rate(T new_lr) override { learning_rate_ = new_lr; }

private:
    T learning_rate_, beta1_, beta2_, epsilon_, weight_decay_;
    int timestep_;

    std::unordered_map<tensor<T>*, std::unique_ptr<tensor<T>>> m_; // First moment
    std::unordered_map<tensor<T>*, std::unique_ptr<tensor<T>>> v_; // Second moment
};

} // namespace dnn

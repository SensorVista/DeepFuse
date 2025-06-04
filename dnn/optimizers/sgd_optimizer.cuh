#pragma once

#include "dnn/optimizers/optimizer.cuh"

namespace dnn {

// SGD (Stochastic Gradient Descent) optimizer efficiently minimizes a loss function and improves 
// model accuracy by iteratively updating the model's parameters based on the gradients calculated 
// from random subsets of the training data

template<typename T>
class SGDOptimizer : public Optimizer<T> {
public:
    SGDOptimizer(T learning_rate = 0.01, T momentum = 0.0, T weight_decay = 0.0)
        : learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {}

    void step() override;
    void zero_grad() override;
    void update_parameters(
        const std::vector<tensor<T>*>& parameters,
        const std::vector<tensor<T>*>& gradients) override;

    T learning_rate() const override { return learning_rate_; }
    void set_learning_rate(T new_lr) override { learning_rate_ = new_lr; }

private:
    T learning_rate_;
    T momentum_;
    T weight_decay_;
    std::vector<tensor<T>> velocities_;
};

} // namespace dnn 

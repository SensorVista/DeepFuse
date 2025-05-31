#pragma once

#include "dnn/core/tensor.cuh"
#include <vector>
#include <memory>

namespace lenet5 {

template<typename T>
class Optimizer {
public:
    virtual ~Optimizer() = default;

    // Perform optimization step
    virtual void step() = 0;
    virtual void zero_grad() = 0;

    virtual std::vector<tensor<T>*> parameters() { return parameters_; }
    virtual std::vector<tensor<T>*> gradients() { return gradients_; }

    // Set parameters to optimize
    virtual void update_parameters(
        const std::vector<tensor<T>*>& parameters,
        const std::vector<tensor<T>*>& gradients) {
        parameters_ = parameters;
        gradients_ = gradients;
    }

protected:
    std::vector<tensor<T>*> parameters_;       // model parameters (weights/biases)
    std::vector<tensor<T>*> gradients_;        // gradients for those parameters
};

} // namespace lenet5 
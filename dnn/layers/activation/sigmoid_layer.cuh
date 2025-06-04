#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

namespace dnn {

// Sigmoid activation layer
// f(x) = 1 / (1 + e^(-x))
// Derivative: f'(x) = f(x) * (1 - f(x))

template<typename T>
class SigmoidLayer : public Layer<T> {
public:

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    const char* name() const override { return "Sigmoid"; }
};

template class SigmoidLayer<float>;

} // namespace dnn 
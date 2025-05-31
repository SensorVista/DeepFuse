#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

namespace lenet5 {

// Tanh activation layer
// f(x) = tanh(x)
// Derivative: f'(x) = 1 - tanh(x)^2

template<typename T>
class TanhLayer : public Layer<T> {
public:
    const char* name() const override { return "Tanh"; }

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;
};

} // namespace lenet5 
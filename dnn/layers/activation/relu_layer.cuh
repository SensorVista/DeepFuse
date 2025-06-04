#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

namespace dnn {

template<typename T>
class ReLULayer : public Layer<T> {
public:
    ReLULayer() = default;
    ~ReLULayer() = default;

    const char* name() const override { return "ReLU"; }

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;
};

} // namespace dnn 
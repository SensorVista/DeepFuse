#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

namespace dnn {

template<typename T>
class FlattenLayer : public Layer<T> {
public:
    std::string name() const override { return "Flatten"; }
    
    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

};

} // namespace dnn 
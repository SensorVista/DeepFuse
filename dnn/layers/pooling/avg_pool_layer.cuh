#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

namespace dnn {

template<typename T>
class AvgPoolLayer : public Layer<T> {
public:
    AvgPoolLayer(int kernel_size, int stride);

    const char* name() const override { return "AvgPool"; }

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;
    
private:
    int kernel_size_;
    int stride_;
};

} // namespace dnn 
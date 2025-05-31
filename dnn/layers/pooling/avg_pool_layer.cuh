#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

namespace lenet5 {

template<typename T>
class AvgPoolLayer : public Layer<T> {
public:
    AvgPoolLayer(size_t kernel_size, size_t stride);

    const char* name() const override { return "AvgPool"; }

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;
    
private:
    size_t kernel_size_;
    size_t stride_;
};

} // namespace lenet5 
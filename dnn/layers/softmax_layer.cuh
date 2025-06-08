#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#include <vector>

#include <string>

namespace dnn {

template<typename T>
class SoftmaxLayer : public Layer<T> {
public:
    SoftmaxLayer(bool use_mask = false);
    ~SoftmaxLayer();

    // Optional external additive mask: shape [B, 1, T, T] or [1, 1, T, T]
    void set_mask(tensor<T>* mask); // Non-owning

    // Forward pass over [B, H, T, T]
    tensor<T> forward(const tensor<T>& input) override;

    // Gradient of softmax (placeholder for now)
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    std::string name() const override { return "Softmax"; }

private:
    bool use_mask_;
    tensor<T>* mask_;  // Non-owning pointer to external mask
};

}  // namespace dnn

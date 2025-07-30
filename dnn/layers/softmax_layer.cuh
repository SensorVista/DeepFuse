#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#include <vector>

#include <string>
#include <optional>
#include <iostream>

namespace dnn {

template<typename T>
class SoftmaxLayer : public Layer<T> {
public:
    SoftmaxLayer(bool training_enabled = false);
    ~SoftmaxLayer();

    // Optional external additive mask: shape [B, 1, T, T] or [1, 1, T, T]
    void set_mask(tensor<T>* mask); // Non-owning

    // Forward pass over [B, H, T, T]
    tensor<T> forward(const tensor<T>& input) override;

    // Gradient of softmax (placeholder for now)
    tensor<T> backward(const tensor<T>& grad_output) override;

    void save(std::ostream& out) const override {}
    void load(std::istream& in) override {}

    std::string name() const override { return "Softmax"; }

private:
    tensor<T>* mask_;  // Non-owning pointer to external mask

    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<T>> input_cache_;
};

}  // namespace dnn

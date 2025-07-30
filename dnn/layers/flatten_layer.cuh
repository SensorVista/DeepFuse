#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"
#include <optional>

namespace dnn {

template<typename T>
class FlattenLayer : public Layer<T> {
public:
    explicit FlattenLayer(bool training_enabled = false);
    std::string name() const override { return "Flatten"; }
    
    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output) override;

    void save(std::ostream& out) const override {}
    void load(std::istream& in) override {}

private:
    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<T>> input_cache_;
};

} // namespace dnn 
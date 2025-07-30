#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#include <string>
#include <vector>
#include <optional>
#include <iostream>

namespace dnn {

template<typename T>
class LayerNormLayer : public Layer<T> {
public:
    LayerNormLayer(int norm_size, float epsilon = 1e-5f, bool affine = true, bool training_enabled = false);
    ~LayerNormLayer() override = default;

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output) override;

    std::vector<tensor<T>*> parameters() override;
    std::vector<tensor<T>*> gradients() override;

    std::string name() const override { return "LayerNorm"; }

    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

private:
    int norm_size_;
    float epsilon_;
    bool affine_;

    tensor<T> gamma_;
    tensor<T> beta_;
    tensor<T> grad_gamma_;
    tensor<T> grad_beta_;

    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<T>> input_cache_;
};

}  // namespace dnn

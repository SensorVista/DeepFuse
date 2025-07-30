#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"
#include "dnn/layers/conv_layer.cuh"
#include "dnn/layers/batch_norm_layer.cuh"
#include "dnn/layers/activation_layer.cuh"
#include "dnn/layers/residual_add_layer.cuh"

#include <memory>
#include <vector>
#include <optional>
#include <iostream>

namespace dnn {

template<typename T>
class ResidualBlock : public Layer<T> {
public:
    ResidualBlock(int in_channels, int out_channels, int stride = 1, bool bottleneck = false, bool training_enabled = false);
    ~ResidualBlock() override = default;

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output) override;

    std::vector<tensor<T>*> parameters() override;
    std::vector<tensor<T>*> gradients() override;

    std::string name() const override { return bottleneck_ ? "ResidualBottleneck" : "ResidualBlock"; }

    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

private:
    bool bottleneck_;
    bool use_projection_;
    int in_channels_, out_channels_, stride_;

    std::vector<std::unique_ptr<BaseLayer>> main_path_;
    std::unique_ptr<ConvLayer<T>> projection_;
    ResidualAddLayer<T> add_;

    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<T>> input_cache_;
};

} // namespace dnn

#pragma once
#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#include <iostream>
#include <random>

namespace lenet5 {

template<typename T>
class ConvLayer : public Layer<T> {
public:
    ConvLayer(size_t in_channels, size_t out_channels, const std::vector<size_t>& kernel_size, size_t stride, size_t padding)
        : in_channels_(in_channels)
        , out_channels_(out_channels)
        , kernel_size_(kernel_size)
        , stride_(stride)
        , padding_(padding)
        , weights_({out_channels, in_channels, kernel_size[0], kernel_size[1]})
        , bias_({out_channels})
        , grad_weights_({out_channels, in_channels, kernel_size[0], kernel_size[1]})
        , grad_bias_({out_channels}) {
        initialize_weights();
    }

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    std::vector<tensor<T>*> parameters() override { return {&weights_, &bias_}; }
    std::vector<tensor<T>*> gradients() override { return {&grad_weights_, &grad_bias_}; }

    const char* name() const override { return "Conv"; }

private:
    size_t in_channels_;
    size_t out_channels_;
    std::vector<size_t> kernel_size_;
    size_t stride_;
    size_t padding_;
    tensor<T> weights_;
    tensor<T> bias_;
    tensor<T> grad_weights_;
    tensor<T> grad_bias_;

    void initialize_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        // Xavier/Glorot initialization
        float limit = std::sqrt(6.0f / (in_channels_ * kernel_size_[0] * kernel_size_[1] + out_channels_ * kernel_size_[0] * kernel_size_[1])); // Adjusted for Glorot uniform
        std::uniform_real_distribution<T> dist(-limit, limit);

        std::vector<T> host_weights(weights_.size());
        std::vector<T> host_bias(bias_.size());
        for (size_t i = 0; i < host_weights.size(); ++i) {
            host_weights[i] = dist(gen);
        }
        std::fill(host_bias.begin(), host_bias.end(), static_cast<T>(0.0f));
        weights_.upload(host_weights.data());
        bias_.upload(host_bias.data());
    }
};

} // namespace lenet5 
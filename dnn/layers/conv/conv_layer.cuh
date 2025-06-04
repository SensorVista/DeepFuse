#pragma once
#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#include <iostream>
#include <random>

namespace dnn {

template<typename T>
class ConvLayer : public Layer<T> {
public:
    ConvLayer(int in_channels, int out_channels, const std::vector<int>& kernel_size, int stride, int padding, const std::vector<std::vector<bool>>& connection_table = {})
        : in_channels_(in_channels)
        , out_channels_(out_channels)
        , kernel_size_(kernel_size)
        , stride_(stride)
        , padding_(padding)
        , weights_({out_channels, in_channels, kernel_size[0], kernel_size[1]})
        , bias_({out_channels})
        , grad_weights_({out_channels, in_channels, kernel_size[0], kernel_size[1]})
        , grad_bias_({out_channels})
        , connection_mask_dev_({out_channels, in_channels})
        , use_sparse_connectivity_(connection_table.size() > 0) {
        if (use_sparse_connectivity_) {
            if (connection_table.size() != out_channels || connection_table[0].size() != in_channels) {
                throw std::runtime_error("Connection table dimensions do not match in/out channels.");
            }
            std::vector<uint8_t> host_connection_mask(out_channels * in_channels);
            for (int i = 0; i < out_channels; ++i) {
                for (int j = 0; j < in_channels; ++j) {
                    host_connection_mask[i * in_channels + j] = connection_table[i][j] ? 1 : 0;
                }
            }
            connection_mask_dev_.upload(host_connection_mask.data());
        }
        initialize_weights();
    }

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    std::vector<tensor<T>*> parameters() override { return {&weights_, &bias_}; }
    std::vector<tensor<T>*> gradients() override { return {&grad_weights_, &grad_bias_}; }

    const char* name() const override { return "Conv"; }

private:
    int in_channels_;
    int out_channels_;
    std::vector<int> kernel_size_;
    int stride_;
    int padding_;
    tensor<T> weights_;
    tensor<T> bias_;
    tensor<T> grad_weights_;
    tensor<T> grad_bias_;
    tensor<uint8_t> connection_mask_dev_; // Flattened mask for device
    bool use_sparse_connectivity_;

    void initialize_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        float limit = std::sqrt(6.0f / (in_channels_ * kernel_size_[0] * kernel_size_[1] + out_channels_ * kernel_size_[0] * kernel_size_[1]));
        std::uniform_real_distribution<T> dist(-limit, limit);

        std::vector<T> host_weights(weights_.size());
        std::vector<T> host_bias(bias_.size());
        for (int i = 0; i < host_weights.size(); ++i) {
            host_weights[i] = dist(gen);
        }
        std::fill(host_bias.begin(), host_bias.end(), static_cast<T>(0.0f));
        weights_.upload(host_weights.data());
        bias_.upload(host_bias.data());
    }
};

} // namespace dnn 
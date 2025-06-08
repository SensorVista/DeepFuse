#pragma once

#include "dnn/core/tensor.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace dnn {

template<typename TO, typename TI>
class LayerAsymmetric {
public:
    virtual ~LayerAsymmetric() = default;

    // Forward pass
    virtual tensor<TO> forward(const tensor<TI>& input) = 0;

    // Backward pass
    virtual tensor<TO> backward(const tensor<TO>& grad_output, const tensor<TI>& input) = 0;

    // Get layer parameters
    virtual std::vector<tensor<TO>*> parameters() { return {}; };

    // Get layer gradients
    virtual std::vector<tensor<TO>*> gradients() { return {}; };

    // Get layer name
    virtual std::string name() const = 0;

protected:
    // Helper function to check input shape
    bool check_input_shape(const tensor<TI>& input, const std::vector<int>& expected_shape) const {
        return input.shape() == expected_shape;
    }
};

template<typename TO, typename TI>
class LayerWeightBiasAsymmetric : public LayerAsymmetric<TO, TI> {
public:
    LayerWeightBiasAsymmetric(tensor<TO>&& weights, tensor<TO>&& bias, tensor<TO>&& grad_weights, tensor<TO>&& grad_bias)
        : weights_(std::move(weights)), bias_(std::move(bias)), grad_weights_(std::move(grad_weights)), grad_bias_(std::move(grad_bias)) {}
    virtual ~LayerWeightBiasAsymmetric() = default;

    // Get layer weights
    virtual tensor<TO>* weights() { return &weights_; };

    // Get layer bias
    virtual tensor<TO>* bias() { return &bias_; };

    // Get layer weights
    virtual tensor<TO>* grad_weights() { return &grad_weights_; };

    // Get layer bias
    virtual tensor<TO>* grad_bias() { return &grad_bias_; }; 

    // Initialize weights and bias
    virtual void initialize_weights() = 0;

    std::vector<tensor<TO>*> parameters() override { return {&weights_, &bias_}; }
    std::vector<tensor<TO>*> gradients() override { return {&grad_weights_, &grad_bias_}; }

protected:
    tensor<TO> weights_;
    tensor<TO> bias_;
    tensor<TO> grad_weights_;
    tensor<TO> grad_bias_;

};

template<typename T>
using Layer = LayerAsymmetric<T, T>;

template<typename T>
using LayerWeightBias = LayerWeightBiasAsymmetric<T, T>;


} // namespace dnn 
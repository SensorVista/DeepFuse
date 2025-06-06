#pragma once

#include "dnn/core/tensor.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <string>
#include <vector>
#include <memory>

namespace dnn {

template<typename T>
class Layer {
public:
    virtual ~Layer() = default;

    // Forward pass
    virtual tensor<T> forward(const tensor<T>& input) = 0;

    // Backward pass
    virtual tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) = 0;

    // Get layer parameters
    virtual std::vector<tensor<T>*> parameters() { return {}; };

    // Get layer gradients
    virtual std::vector<tensor<T>*> gradients() { return {}; };

    // Get layer name
    virtual std::string name() const = 0;

protected:
    // Helper function to check input shape
    bool check_input_shape(const tensor<T>& input, const std::vector<int>& expected_shape) const {
        return input.shape() == expected_shape;
    }
};

// Forward declarations for supported template types
template class Layer<float>;  // FP32
// template class Layer<__half>; // FP16
// template class Layer<__nv_bfloat16>; // BF16


template<typename T>
class LayerWeightBias : public Layer<T> {
public:
    LayerWeightBias(tensor<T>&& weights, tensor<T>&& bias, tensor<T>&& grad_weights, tensor<T>&& grad_bias)
        : weights_(std::move(weights)), bias_(std::move(bias)), grad_weights_(std::move(grad_weights)), grad_bias_(std::move(grad_bias)) {}
    virtual ~LayerWeightBias() = default;

    // Get layer weights
    virtual tensor<T>* weights() { return &weights_; };

    // Get layer bias
    virtual tensor<T>* bias() { return &bias_; };

    // Get layer weights
    virtual tensor<T>* grad_weights() { return &grad_weights_; };

    // Get layer bias
    virtual tensor<T>* grad_bias() { return &grad_bias_; }; 

    // Initialize weights and bias
    virtual void initialize_weights() = 0;

    std::vector<tensor<T>*> parameters() override { return {&weights_, &bias_}; }
    std::vector<tensor<T>*> gradients() override { return {&grad_weights_, &grad_bias_}; }

protected:
    tensor<T> weights_;
    tensor<T> bias_;
    tensor<T> grad_weights_;
    tensor<T> grad_bias_;

};

} // namespace dnn 
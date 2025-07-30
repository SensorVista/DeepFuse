#pragma once

#include "dnn/core/tensor.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <iostream>

namespace dnn {

class BaseLayer {
public:
    virtual ~BaseLayer() = default;

    // Get layer name
    virtual std::string name() const = 0;

    // Save layer metadata and weights
    virtual void save(std::ostream& out) const = 0;
    // Load layer metadata and weights
    virtual void load(std::istream& in) = 0;
};

template<typename TTO, typename TTI>
class LayerAsymmetric : public BaseLayer {
public:
    explicit LayerAsymmetric(bool training_enabled = false) : training_enabled_(training_enabled) {}
    virtual ~LayerAsymmetric() = default;

    // Forward pass
    virtual tensor<TTO> forward(const tensor<TTI>& input) = 0;

    // Backward pass (input is now handled by per-layer cache)
    virtual tensor<TTO> backward(const tensor<TTO>& grad_output) = 0;

    // Get layer parameters
    virtual std::vector<tensor<TTO>*> parameters() { return {}; }

    // Get layer gradients
    virtual std::vector<tensor<TTO>*> gradients() { return {}; }

    // Getter for training_enabled
    bool training_enabled() const { return training_enabled_; }

protected:
    bool training_enabled_;
};

template<typename TTO, typename TTI>
class LayerWeightBiasAsymmetric : public LayerAsymmetric<TTO, TTI> {
public:
    LayerWeightBiasAsymmetric(tensor<TTO>&& weights, tensor<TTO>&& bias, tensor<TTO>&& grad_weights, tensor<TTO>&& grad_bias, bool training_enabled = false)
        : LayerAsymmetric<TTO, TTI>(training_enabled), weights_(std::move(weights)), bias_(std::move(bias)), grad_weights_(std::move(grad_weights)), grad_bias_(std::move(grad_bias)) {}
    virtual ~LayerWeightBiasAsymmetric() = default;

    // Get layer weights
    tensor<TTO>* weights() { return &weights_; };

    // Get layer bias
    tensor<TTO>* bias() { return &bias_; };

    // Get layer gradient weights
    tensor<TTO>* grad_weights() { return &grad_weights_; };

    // Get layer gradient bias
    tensor<TTO>* grad_bias() { return &grad_bias_; }; 

    // Initialize weights and bias
    virtual void initialize_weights() = 0;

    std::vector<tensor<TTO>*> parameters() override { return {&weights_, &bias_}; }
    std::vector<tensor<TTO>*> gradients() override { return {&grad_weights_, &grad_bias_}; }

    // Backward pass (input is now handled by per-layer cache)
    tensor<TTO> backward(const tensor<TTO>& grad_output) override = 0;

protected:
    tensor<TTO> weights_;
    tensor<TTO> bias_;
    tensor<TTO> grad_weights_;
    tensor<TTO> grad_bias_;
};

template<typename TT>
using Layer = LayerAsymmetric<TT, TT>;

template<typename TT>
using LayerWeightBias = LayerWeightBiasAsymmetric<TT, TT>;

} // namespace dnn 
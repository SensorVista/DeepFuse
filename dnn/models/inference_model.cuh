#pragma once

#include "../core/layer.cuh"
#include "../losses/loss.cuh"
#include "../optimizers/optimizer.cuh"
#include <vector>
#include <memory>

namespace dnn {

// InferenceModel class
// Defines the forward pass through the network

template<typename T>
class InferenceModel {
public:
    explicit InferenceModel() = default;
    virtual ~InferenceModel() = default;

    // Forward pass through the network
    virtual tensor<T> forward(const tensor<T>& input) {
        tensor<T> current(input.shape());
        current.copy_from(input);
        for (const auto& layer : layers_) {
            current = layer->forward(current);  // move assignment
        }
        return current;
    }

    // Get current loss value
    T loss() const { return current_loss_; } 

    // Get all parameters from all layers
    virtual std::vector<tensor<T>*> parameters() {
        std::vector<tensor<T>*> params;
        for (const auto& layer : layers_) {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    // Save model weights
    virtual void save_weights(const std::string& path) const {
        // Implementation for saving model weights
    }

    // Load model weights
    virtual void load_weights(const std::string& path) {
        // Implementation for loading model weights
    }

protected:
    std::vector<std::unique_ptr<Layer<T>>> layers_;
    T current_loss_ = 0;
};

} // namespace dnn 
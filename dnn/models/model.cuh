#pragma once

#include <dnn/core/layer.cuh>
#include <dnn/losses/loss.cuh>
#include <dnn/optimizers/optimizer.cuh>

#include <vector>
#include <memory>
#include <string>

namespace dnn {

// Model class for Inference
// Defines the forward pass through the network

template<typename T>
class Model {
public:
    explicit Model(bool training_enabled = false) : training_enabled_(training_enabled) {}
    virtual ~Model() = default;

    // Get all parameters from all layers
    virtual std::string name() const { return "DefaultNet"; }

    // Get all layers in order (non-owning pointers, do not delete/copy)
    virtual std::vector<BaseLayer*> layers() = 0;

    // Forward pass through the network
    virtual tensor<T> forward(const tensor<T>& input);

    // Get current loss value
    T loss() const { return current_loss_; } 

    // Get all parameters from all layers
    virtual std::vector<tensor<T>*> parameters();

    // Getter for training_enabled
    bool training_enabled() const { return training_enabled_; }

    // Save model metadata and weights
    virtual void save(const std::string& path) const = 0;

protected:
    T current_loss_ = 0;
    bool training_enabled_;
};

} // namespace dnn 
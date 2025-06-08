#pragma once

#include "../core/layer.cuh"
#include "../losses/loss.cuh"
#include "../optimizers/optimizer.cuh"
#include <vector>
#include <memory>
#include <string>

namespace dnn {

// Storage format types for model serialization
enum class StorageType {
    ONNX     // ONNX format
};

// InferenceModel class
// Defines the forward pass through the network

template<typename T>
class Model {
public:
    explicit Model() = default;
    virtual ~Model() = default;

    // Forward pass through the network
    virtual tensor<T> forward(const tensor<T>& input);

    // Get current loss value
    T loss() const { return current_loss_; } 

    // Get all parameters from all layers
    virtual std::vector<tensor<T>*> parameters();

    // Save model weights and architecture
    virtual void save(const std::string& path, StorageType type = StorageType::ONNX) const;

    // Load model from file
    static Model<T>* load(const std::string& path, StorageType type = StorageType::ONNX);

    // Add a layer to the model
    void add_layer(std::unique_ptr<Layer<T>> layer) {
        layers_.push_back(std::move(layer));
    }

    // Get all layers
    const std::vector<std::unique_ptr<Layer<T>>>& layers() const { return layers_; }

protected:
    std::vector<std::unique_ptr<Layer<T>>> layers_;
    T current_loss_ = 0;
};

} // namespace dnn 
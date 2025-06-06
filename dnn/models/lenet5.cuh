#pragma once

#include <dnn/layers/conv_layer.cuh>
#include <dnn/layers/fully_connected_layer.cuh>
#include <dnn/layers/activation_layer.cuh>
#include <dnn/layers/pooling_layer.cuh>
#include <dnn/layers/flatten_layer.cuh>
#include <dnn/losses/cross_entropy.cuh>
#include <dnn/models/training_model.cuh>
#include <dnn/optimizers/sgd_optimizer.cuh>

#include <memory>

namespace dnn {

template<typename T>
class LeNet5 : public TrainingModel<T> {
public:
    explicit LeNet5(T learning_rate = 0.01, T momentum = 0.9) : TrainingModel<T>() {
        // Per Yann LeCun's paper, the C3 layer uses a sparse connectivity mask
        // This is a 16x6 matrix that specifies which input maps are connected to each output map
        // The matrix is defined as follows:
        const std::vector<std::vector<bool>> c3_connection_table = {
            {1, 0, 0, 0, 0, 1},  // Output map 0 connects to input maps 0 and 5
            {0, 1, 0, 0, 1, 0},  // Output map 1: inputs 1 and 4
            {0, 0, 1, 1, 0, 0},  // etc.
            {1, 1, 0, 0, 0, 0},
            {0, 1, 1, 0, 0, 0},
            {0, 0, 1, 1, 0, 0},
            {0, 0, 0, 1, 1, 0},
            {0, 0, 0, 0, 1, 1},
            {1, 0, 0, 0, 1, 0},
            {0, 1, 0, 1, 0, 0},
            {0, 0, 1, 0, 1, 0},
            {0, 0, 0, 1, 0, 1},
            {1, 0, 1, 0, 0, 0},
            {0, 1, 0, 0, 0, 1},
            {1, 0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0, 1}
        };

        // Create LeNet-5 architecture
        // Conv1: 1x32x32 -> 6x28x28
        this->layers_.push_back(std::make_unique<ConvLayer<T>>(1, 6, std::vector<int>{5, 5}, 1, 0));
        // Tanh activation after Conv1
        this->layers_.push_back(std::make_unique<ActivationLayer<T>>(ActivationType::Tanh));
        // Pool1: 2x2 average pooling (stride 2)
        this->layers_.push_back(std::make_unique<PoolingLayer<T>>(PoolingType::Average, 2, 2));
        // Conv2: 6x14x14 -> 16x10x10
        this->layers_.push_back(std::make_unique<ConvLayer<T>>(6, 16, std::vector<int>{5, 5}, 1, 0, c3_connection_table));
        // Tanh activation after Conv2
        this->layers_.push_back(std::make_unique<ActivationLayer<T>>(ActivationType::Tanh));
        // Pool2: 2x2 average pooling (stride 2)
        this->layers_.push_back(std::make_unique<PoolingLayer<T>>(PoolingType::Average, 2, 2));
        // Conv3: 16x5x5 -> 120x1x1
        this->layers_.push_back(std::make_unique<ConvLayer<T>>(16, 120, std::vector<int>{5, 5}, 1, 0));
        // Tanh activation after Conv3
        this->layers_.push_back(std::make_unique<ActivationLayer<T>>(ActivationType::Tanh));
        // Flatten: convert 3D tensor to 1D vector
        this->layers_.push_back(std::make_unique<FlattenLayer<T>>());
        // FC1 (F6): Fully connected, 84 units
        this->layers_.push_back(std::make_unique<FullyConnectedLayer<T>>(120, 84));
        // Tanh activation after FC1
        this->layers_.push_back(std::make_unique<ActivationLayer<T>>(ActivationType::Tanh));
        // FC2 (output): Fully connected, 10 units
        this->layers_.push_back(std::make_unique<FullyConnectedLayer<T>>(84, 10));

        // Initialize with default optimizer and loss function
        this->set_optimizer(std::make_unique<SGDOptimizer<T>>(learning_rate, momentum, 0.0f));
        this->set_loss(std::make_unique<CrossEntropyLoss<T>>());
    }
};

} // namespace dnn 
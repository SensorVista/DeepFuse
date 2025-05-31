#pragma once

#include "training_model.cuh"
#include "../layers/conv/conv_layer.cuh"
#include "../layers/linear/fully_connected_layer.cuh"
#include "../layers/activation/tanh_layer.cuh"
#include "../layers/pooling/avg_pool_layer.cuh"
#include "../layers/reshape/flatten_layer.cuh"
#include "../losses/cross_entropy.cuh"
#include "../optimizers/sgd_optimizer.cuh"
#include <memory>

namespace lenet5 {

template<typename T>
class LeNet5 : public TrainingModel<T> {
public:
    explicit LeNet5(T learning_rate = 0.01f, T momentum = 0.0f, T weight_decay = 0.0f) : TrainingModel<T>() {
        // Create LeNet-5 architecture
        // Conv1: 1x32x32 -> 6x28x28
        this->layers_.push_back(std::make_unique<ConvLayer<T>>(1, 6, std::vector<size_t>{5, 5}, 1, 0));
        // Tanh activation after Conv1
        this->layers_.push_back(std::make_unique<TanhLayer<T>>());
        // Pool1: 2x2 average pooling (stride 2)
        this->layers_.push_back(std::make_unique<AvgPoolLayer<T>>(2, 2));
        // Conv2: 6x14x14 -> 16x10x10
        this->layers_.push_back(std::make_unique<ConvLayer<T>>(6, 16, std::vector<size_t>{5, 5}, 1, 0));
        // Tanh activation after Conv2
        this->layers_.push_back(std::make_unique<TanhLayer<T>>());
        // Pool2: 2x2 average pooling (stride 2)
        this->layers_.push_back(std::make_unique<AvgPoolLayer<T>>(2, 2));
        // Conv3: 16x5x5 -> 120x1x1
        this->layers_.push_back(std::make_unique<ConvLayer<T>>(16, 120, std::vector<size_t>{5, 5}, 1, 0));
        // Tanh activation after Conv3
        this->layers_.push_back(std::make_unique<TanhLayer<T>>());
        // Flatten: convert 3D tensor to 1D vector
        this->layers_.push_back(std::make_unique<FlattenLayer<T>>());
        // FC1 (F6): Fully connected, 84 units
        this->layers_.push_back(std::make_unique<FullyConnectedLayer<T>>(120, 84));
        // Tanh activation after FC1
        this->layers_.push_back(std::make_unique<TanhLayer<T>>());
        // FC2 (output): Fully connected, 10 units
        this->layers_.push_back(std::make_unique<FullyConnectedLayer<T>>(84, 10));

        // Set default loss and optimizer
        this->set_loss(std::make_unique<CrossEntropyLoss<T>>());
        this->set_optimizer(std::make_unique<SGDOptimizer<T>>(learning_rate, momentum, weight_decay));
    }
};

} // namespace lenet5 
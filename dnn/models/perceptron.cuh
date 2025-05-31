#pragma once

#include "training_model.cuh"
#include "../layers/linear/fully_connected_layer.cuh"
#include "../layers/activation/tanh_layer.cuh"
#include "../layers/activation/sigmoid_layer.cuh"
#include "../losses/binary_cross_entropy.cuh"
#include "../optimizers/sgd_optimizer.cuh"
#include <memory>

namespace lenet5 {

template<typename T>
class Perceptron : public TrainingModel<T> {
public:
    Perceptron(size_t input_size, size_t hidden_size, size_t output_size, T learning_rate = 0.01f) 
        : TrainingModel<T>() {
        // Create MLP architecture
        // Input -> Hidden layer
        this->layers_.push_back(std::make_unique<FullyConnectedLayer<T>>(input_size, hidden_size));
        this->layers_.push_back(std::make_unique<TanhLayer<T>>());
        
        // Hidden -> Output layer
        this->layers_.push_back(std::make_unique<FullyConnectedLayer<T>>(hidden_size, output_size));
        this->layers_.push_back(std::make_unique<SigmoidLayer<T>>());

        // Set default loss and optimizer
        this->set_loss(std::make_unique<BinaryCrossEntropyLoss<T>>());
        this->set_optimizer(std::make_unique<SGDOptimizer<T>>(learning_rate, 0.9f));
    }
};

} // namespace lenet5 

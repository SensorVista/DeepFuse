#pragma once

#include <dnn/layers/activation_layer.cuh>
#include <dnn/losses/binary_cross_entropy.cuh>
#include <dnn/layers/fully_connected_layer.cuh>
#include <dnn/optimizers/sgd_optimizer.cuh>
#include <dnn/models/training_model.cuh>

#include <memory>

namespace dnn {

template<typename T>
class Perceptron : public TrainingModel<T> {
public:
    Perceptron(int input_size, int hidden_size, int output_size, T learning_rate = 0.01, T momentum = 0.9) 
        : TrainingModel<T>() {
        // Create MLP architecture
        // Input -> Hidden layer
        this->layers_.push_back(std::make_unique<FullyConnectedLayer<T>>(input_size, hidden_size));
        this->layers_.push_back(std::make_unique<ActivationLayer<T>>(ActivationType::Tanh));
        
        // Hidden -> Output layer
        this->layers_.push_back(std::make_unique<FullyConnectedLayer<T>>(hidden_size, output_size));
        this->layers_.push_back(std::make_unique<ActivationLayer<T>>(ActivationType::Sigmoid));

        // Set loss and optimizer
        this->set_loss(std::make_unique<dnn::BinaryCrossEntropyLoss<T>>());
        this->set_optimizer(std::make_unique<dnn::SGDOptimizer<T>>(learning_rate, momentum));
    }
};

} // namespace dnn 

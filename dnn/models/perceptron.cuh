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
    Perceptron(int input_size, int hidden_size, int output_size, T learning_rate = 0.01, T momentum = 0.9, bool training_enabled = false);

    void save(const std::string& path) const override;
    static std::unique_ptr<Perceptron<T>> load(const std::string& path, bool training_enabled);

    std::vector<BaseLayer*> layers() override;

private:
    FullyConnectedLayer<T> fc1_;
    ActivationLayer<T> act1_;
    FullyConnectedLayer<T> fc2_;
    ActivationLayer<T> act2_;
};

} // namespace dnn 

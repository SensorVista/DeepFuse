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
    static const std::vector<std::vector<bool>>& c3_connection_table() {
        static const std::vector<std::vector<bool>> table = {
            {1, 0, 0, 0, 0, 1},
            {0, 1, 0, 0, 1, 0},
            {0, 0, 1, 1, 0, 0},
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
        return table;
    }

    explicit LeNet5(T learning_rate = 0.01, T momentum = 0.9, bool training_enabled = false)
        : TrainingModel<T>(training_enabled),
          conv1_(1, 6, std::vector<int>{5, 5}, 1, 0, std::vector<std::vector<bool>>{}, training_enabled),
          act1_(ActivationType::Tanh, training_enabled),
          pool1_(PoolingType::Average, 2, 2, training_enabled),
          conv2_(6, 16, std::vector<int>{5, 5}, 1, 0, c3_connection_table(), training_enabled),
          act2_(ActivationType::Tanh, training_enabled),
          pool2_(PoolingType::Average, 2, 2, training_enabled),
          conv3_(16, 120, std::vector<int>{5, 5}, 1, 0, std::vector<std::vector<bool>>{}, training_enabled),
          act3_(ActivationType::Tanh, training_enabled),
          flatten_(training_enabled),
          fc1_(120, 84, training_enabled),
          act4_(ActivationType::Tanh, training_enabled),
          fc2_(84, 10, training_enabled)
    {
        this->set_optimizer(std::make_unique<SGDOptimizer<T>>(learning_rate, momentum, 0.0f));
        this->set_loss(std::make_unique<CrossEntropyLoss<T>>());
    }

    void save(const std::string& path) const override;
    static std::unique_ptr<LeNet5<T>> load(const std::string& path, bool training_enabled);

    std::vector<BaseLayer*> layers() override {
        return { &conv1_, &act1_, &pool1_, &conv2_, &act2_, &pool2_, &conv3_, &act3_, &flatten_, &fc1_, &act4_, &fc2_ };
    }

private:
    ConvLayer<T> conv1_;
    ActivationLayer<T> act1_;
    PoolingLayer<T> pool1_;
    ConvLayer<T> conv2_;
    ActivationLayer<T> act2_;
    PoolingLayer<T> pool2_;
    ConvLayer<T> conv3_;
    ActivationLayer<T> act3_;
    FlattenLayer<T> flatten_;
    FullyConnectedLayer<T> fc1_;
    ActivationLayer<T> act4_;
    FullyConnectedLayer<T> fc2_;
};

} // namespace dnn 
#pragma once

#include <dnn/models/training_model.cuh>
#include <dnn/layers/conv_layer.cuh>
#include <dnn/layers/batch_norm_layer.cuh>
#include <dnn/layers/activation_layer.cuh>
#include <dnn/layers/residual_block.cuh>
#include <dnn/layers/pooling_layer.cuh>
#include <dnn/layers/fully_connected_layer.cuh>
#include <dnn/losses/cross_entropy.cuh>
#include <dnn/optimizers/sgd_optimizer.cuh>

namespace dnn {

/// ResNet-32 deepens ResNet-20 with 5 blocks per stage.
/// Improves accuracy while maintaining manageable size.
/// Ideal for medium-scale CIFAR experiments.

template<typename T>
class ResNet32 : public TrainingModel<T> {
public:
    explicit ResNet32(int num_classes = 10, T lr = 0.1, T momentum = 0.9) {
        const int C = 16;

        this->layers_.push_back(std::make_unique<ConvLayer<T>>(3, C, std::vector<int>{3, 3}, 1, 1));
        this->layers_.push_back(std::make_unique<BatchNormLayer<T>>(C));
        this->layers_.push_back(std::make_unique<ActivationLayer<T>>(ActivationType::ReLU));

        for (int i = 0; i < 5; ++i) this->layers_.push_back(std::make_unique<ResidualBlock<T>>(C, C, 1, false));
        this->layers_.push_back(std::make_unique<ResidualBlock<T>>(C, C * 2, 2, false));
        for (int i = 1; i < 5; ++i) this->layers_.push_back(std::make_unique<ResidualBlock<T>>(C * 2, C * 2, 1, false));
        this->layers_.push_back(std::make_unique<ResidualBlock<T>>(C * 2, C * 4, 2, false));
        for (int i = 1; i < 5; ++i) this->layers_.push_back(std::make_unique<ResidualBlock<T>>(C * 4, C * 4, 1, false));

        this->layers_.push_back(std::make_unique<GlobalAvgPoolLayer<T>>());
        this->layers_.push_back(std::make_unique<FullyConnectedLayer<T>>(C * 4, num_classes));

        this->set_loss(std::make_unique<CrossEntropyLoss<T>>());
        this->set_optimizer(std::make_unique<SGDOptimizer<T>>(lr, momentum, 0.0f));
    }
};

}

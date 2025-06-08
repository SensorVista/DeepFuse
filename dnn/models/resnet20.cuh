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

/// ResNet-20 architecture for CIFAR-10.
/// Uses 3 blocks per stage with basic 3×3 residual units.
/// Lightweight design suitable for low-resource training.

template<typename T>
class ResNet20 : public TrainingModel<T> {
public:
    explicit ResNet20(int num_classes = 10, T learning_rate = 0.1, T momentum = 0.9) : TrainingModel<T>() {
        const int input_channels = 3;
        const int base_channels = 16;

        // Initial Conv + BN + ReLU
        this->layers_.push_back(std::make_unique<ConvLayer<T>>(input_channels, base_channels, std::vector<int>{3, 3}, 1, 1));
        this->layers_.push_back(std::make_unique<BatchNormLayer<T>>(base_channels));
        this->layers_.push_back(std::make_unique<ActivationLayer<T>>(ActivationType::ReLU));

        // 3 × Residual Blocks at 16 channels
        for (int i = 0; i < 3; ++i) {
            this->layers_.push_back(std::make_unique<ResidualBlock<T>>(base_channels, base_channels, 1, false));
        }

        // 3 × Residual Blocks at 32 channels (stride 2 on first)
        this->layers_.push_back(std::make_unique<ResidualBlock<T>>(base_channels, base_channels * 2, 2, false));
        for (int i = 1; i < 3; ++i) {
            this->layers_.push_back(std::make_unique<ResidualBlock<T>>(base_channels * 2, base_channels * 2, 1, false));
        }

        // 3 × Residual Blocks at 64 channels (stride 2 on first)
        this->layers_.push_back(std::make_unique<ResidualBlock<T>>(base_channels * 2, base_channels * 4, 2, false));
        for (int i = 1; i < 3; ++i) {
            this->layers_.push_back(std::make_unique<ResidualBlock<T>>(base_channels * 4, base_channels * 4, 1, false));
        }

        // Global average pool and final FC layer
        this->layers_.push_back(std::make_unique<GlobalAvgPoolLayer<T>>());
        this->layers_.push_back(std::make_unique<FullyConnectedLayer<T>>(base_channels * 4, num_classes));

        // Loss and optimizer
        this->set_loss(std::make_unique<CrossEntropyLoss<T>>());
        this->set_optimizer(std::make_unique<SGDOptimizer<T>>(learning_rate, momentum, 0.0f));
    }
};

} // namespace dnn

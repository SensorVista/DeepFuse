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
    explicit ResNet20(int num_classes = 10, T learning_rate = 0.1, T momentum = 0.9, bool training_enabled = false)
        : TrainingModel<T>(training_enabled),
          conv1_(3, 16, std::vector<int>{3, 3}, 1, 1, std::vector<std::vector<bool>>{}, training_enabled),
          bn1_(16, 1e-5f, 0.9f, true, training_enabled),
          act1_(ActivationType::ReLU, training_enabled),
          res_stage1_(),
          res_stage2_(),
          res_stage3_(),
          global_pool_(PoolingType::Average, training_enabled),
          fc_(16 * 4, num_classes, training_enabled)
    {
        // Stage 1: 3 residual blocks (16, 16, 1)
        for (int i = 0; i < 3; ++i)
            res_stage1_.emplace_back(std::make_unique<ResidualBlock<T>>(16, 16, 1, false, training_enabled));
        // Stage 2: 1 downsample block (16, 32, 2), then 2 (32, 32, 1)
        res_stage2_.emplace_back(std::make_unique<ResidualBlock<T>>(16, 32, 2, false, training_enabled));
        for (int i = 1; i < 3; ++i)
            res_stage2_.emplace_back(std::make_unique<ResidualBlock<T>>(32, 32, 1, false, training_enabled));
        // Stage 3: 1 downsample block (32, 64, 2), then 2 (64, 64, 1)
        res_stage3_.emplace_back(std::make_unique<ResidualBlock<T>>(32, 64, 2, false, training_enabled));
        for (int i = 1; i < 3; ++i)
            res_stage3_.emplace_back(std::make_unique<ResidualBlock<T>>(64, 64, 1, false, training_enabled));
        this->set_loss(std::make_unique<CrossEntropyLoss<T>>());
        this->set_optimizer(std::make_unique<SGDOptimizer<T>>(learning_rate, momentum, 0.0f));
    }

    void save(const std::string& path) const override;
    static std::unique_ptr<ResNet20<T>> load(const std::string& path, bool training_enabled);

    std::vector<BaseLayer*> layers() override {
        std::vector<BaseLayer*> result = { &conv1_, &bn1_, &act1_ };
        for (auto& block : res_stage1_) result.push_back(block.get());
        for (auto& block : res_stage2_) result.push_back(block.get());
        for (auto& block : res_stage3_) result.push_back(block.get());
        result.push_back(&global_pool_);
        result.push_back(&fc_);
        return result;
    }

private:
    ConvLayer<T> conv1_;
    BatchNormLayer<T> bn1_;
    ActivationLayer<T> act1_;
    std::vector<std::unique_ptr<ResidualBlock<T>>> res_stage1_;
    std::vector<std::unique_ptr<ResidualBlock<T>>> res_stage2_;
    std::vector<std::unique_ptr<ResidualBlock<T>>> res_stage3_;
    GlobalPoolingLayer<T> global_pool_;
    FullyConnectedLayer<T> fc_;
};

} // namespace dnn

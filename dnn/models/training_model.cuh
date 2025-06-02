#pragma once

#include "../core/layer.cuh"
#include "../losses/loss.cuh"
#include "../optimizers/optimizer.cuh"
#include "inference_model.cuh"
#include "../utils/common.cuh"

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>

namespace lenet5 {

// TrainingModel class inherits from InferenceModel
// Adds support loss function, optimizers to the model and training methods

template<typename T>
class TrainingModel : public InferenceModel<T> {
public:
    explicit TrainingModel(bool enable_training = false) : InferenceModel<T>() {}

    virtual ~TrainingModel() = default;

    // Training step
    virtual void train_step(const tensor<T>& input, const tensor<T>& target) {
        // Zero gradients before forward pass
        optimizer_->zero_grad();

        // Cache inputs/outputs for each layer
        std::vector<tensor<T>> layer_inputs;
        tensor<T> current(input.shape());
        current.copy_from(input);
        layer_inputs.push_back(std::move(current));

        // Forward pass
        for (const auto& layer : this->layers_) {
            //std::cout << layer->name() << " Forward" << std::endl;
            tensor<T> output = layer->forward(layer_inputs.back());  // value return, moveable
            layer_inputs.push_back(std::move(output));  // Cache output for next layer
        }

        cudaDeviceSynchronize();
        
        // Compute loss
        this->current_loss_ = loss_->compute(layer_inputs.back(), target);

        // Compute loss gradient
        tensor<T> current_grad = loss_->compute_gradient(layer_inputs.back(), target);

        // Backward pass
        for (int i = static_cast<int>(this->layers_.size()) - 1; i >= 0; --i) {
            current_grad = this->layers_[i]->backward(current_grad, layer_inputs[i]);
        }

        cudaDeviceSynchronize();
        
        // Clip gradients
        lenet5::utils::clip_grad_norm(optimizer_->gradients(), 1.0f);

        // Update parameters
        optimizer_->step();
    }

    // Set optimizer
    virtual void set_optimizer(std::unique_ptr<Optimizer<T>> optimizer) {
        optimizer_ = std::move(optimizer);

        // Update trainable parameters
        std::vector<tensor<T>*> params, grads;
        for (const auto& layer : this->layers_) {
            for (auto* p : layer->parameters()) {
                params.push_back(p);
            }
            for (auto* g : layer->gradients()) {
                grads.push_back(g);
            }
        }
        optimizer_->update_parameters(params, grads);
    }

    // Set loss function
    virtual void set_loss(std::unique_ptr<Loss<T>> loss) {
        loss_ = std::move(loss);
    }

    // Expose optimizer's learning rate methods
    T learning_rate() const { return optimizer_->learning_rate(); }
    void set_learning_rate(T new_lr) { optimizer_->set_learning_rate(new_lr); }

protected:
    std::unique_ptr<Loss<T>> loss_;
    std::unique_ptr<Optimizer<T>> optimizer_;

};

} // namespace lenet5 

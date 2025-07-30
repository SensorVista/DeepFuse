#pragma once

#include "../core/layer.cuh"
#include "../losses/loss.cuh"
#include "../optimizers/optimizer.cuh"
#include "model.cuh"
#include "../utils/common.cuh"

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>

namespace dnn {

// TrainingModel class inherits from InferenceModel
// Adds support loss function, optimizers to the model and training methods

template<typename T>
class TrainingModel : public Model<T> {
public:
    explicit TrainingModel(bool training_enabled = false) : Model<T>(training_enabled) {
    }

    virtual ~TrainingModel() = default;

    // Training step
    virtual void train_step(const tensor<T>& input, const tensor<T>& target) {
        // Zero gradients before forward pass
        optimizer_->zero_grad();

        // Forward pass
        tensor<T> current = input.clone();
        for (auto* layer : this->layers()) {
            auto typed_layer = dynamic_cast<Layer<T>*>(layer);
            if (!typed_layer) {
                throw std::runtime_error("Layer type mismatch in forward pass");
            }
            current = typed_layer->forward(current);
        }

        cudaDeviceSynchronize();
        
        // Compute loss
        this->current_loss_ = loss_->compute(current, target);

        // Compute loss gradient
        tensor<T> current_grad = loss_->compute_gradient(current, target);

        // Backward pass
        auto layers_vec = this->layers();
        for (int i = static_cast<int>(layers_vec.size()) - 1; i >= 0; --i) {
            auto typed_layer = dynamic_cast<Layer<T>*>(layers_vec[i]);
            if (!typed_layer) {
                throw std::runtime_error("Layer type mismatch in backward pass");
            }
            current_grad = typed_layer->backward(current_grad);
        }

        cudaDeviceSynchronize();
        
        // Collect gradients for clipping
        std::vector<tensor<T>*> all_grads;
        for (auto* layer : this->layers()) {
            auto typed_layer = dynamic_cast<Layer<T>*>(layer);
            if (typed_layer) {
                auto layer_grads = typed_layer->gradients();
                all_grads.insert(all_grads.end(), layer_grads.begin(), layer_grads.end());
            }
        }

        // Clip gradients
        dnn::utils::clip_grad_norm(all_grads, T(1.0));

        // Update parameters
        optimizer_->step();
    }

    // Set optimizer
    virtual void set_optimizer(std::unique_ptr<Optimizer<T>> optimizer) {
        optimizer_ = std::move(optimizer);

        // Update trainable parameters
        std::vector<tensor<T>*> params, grads;
        for (auto* layer : this->layers()) {
            auto typed_layer = dynamic_cast<Layer<T>*>(layer);
            if (typed_layer) {
                auto layer_params = typed_layer->parameters();
                auto layer_grads = typed_layer->gradients();
                params.insert(params.end(), layer_params.begin(), layer_params.end());
                grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
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

} // namespace dnn 

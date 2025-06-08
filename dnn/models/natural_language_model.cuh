#pragma once

#include "training_model.cuh"
#include "../tokens/tokenizer.cuh"

#include <memory>

namespace dnn {

template<typename T>
class NaturalLanguageModel : public TrainingModel<T> {
public:
    NaturalLanguageModel(std::shared_ptr<Tokenizer> tokenizer,
                         std::unique_ptr<Loss<T>> loss,
                         std::unique_ptr<Optimizer<T>> optimizer)
        : tokenizer_(std::move(tokenizer)) {
        this->set_loss(std::move(loss));
        this->set_optimizer(std::move(optimizer));
    }

    tensor<T> forward_from_text(const std::string& text) {
        std::vector<int> tokens = tokenizer_->encode(text, true);
        tensor<T> input = to_tensor(tokens);  // Convert to 1×N tensor
        return this->forward(input);
    }

    void train_from_text(const std::string& text) {
        std::vector<int> tokens = tokenizer_->encode(text, true);
        std::vector<int> input_tokens(tokens.begin(), tokens.end() - 1);
        std::vector<int> target_tokens(tokens.begin() + 1, tokens.end());

        tensor<T> input = to_tensor(input_tokens);
        tensor<T> target = to_tensor(target_tokens);
        this->train_step(input, target);
    }

protected:
    std::shared_ptr<Tokenizer> tokenizer_;

    tensor<T> to_tensor(const std::vector<int>& ids) const {
        tensor<T> result({1, static_cast<int>(ids.size()), 1});
        std::vector<T> host(ids.begin(), ids.end());
        result.upload(host.data());
        return result;
    }
};

}  // namespace dnn

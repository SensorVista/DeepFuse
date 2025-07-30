#pragma once

#include "training_model.cuh"
#include "../tokens/tokenizer.cuh"

#include <memory>

namespace dnn {

template<typename T>
class NaturalLanguageModel : public TrainingModel<T> {
public:
    NaturalLanguageModel(std::shared_ptr<Tokenizer> tokenizer,
                         int max_seq_len = 1024,
                         bool training_enabled = false)  // Default GPT-2 context length
        : TrainingModel<T>(training_enabled),
          tokenizer_(std::move(tokenizer)),
          max_seq_len_(max_seq_len),
          attention_mask_({1, 1, max_seq_len, max_seq_len})  // Initialize with shape [B, 1, T, T]
    {        
        // Initialize mask with ones
        std::vector<T> host_mask(attention_mask_.size(), static_cast<T>(1.0f));
        attention_mask_.upload(host_mask.data());
    }

    tensor<T> forward_from_text(const std::string& text) {
        std::vector<int> tokens = tokenizer_->encode(text, true);
        tensor<T> input = to_tensor(tokens);  // Convert to [B, T] tensor
        return this->forward(input);
    }

    void train_from_text(const std::string& text) {
        // Encode text with special tokens
        std::vector<int> tokens = tokenizer_->encode(text, true);
        
        // Split into input and target sequences
        // Input: [0, 1, 2, ..., n-1]
        // Target: [1, 2, 3, ..., n]
        std::vector<int> input_tokens(tokens.begin(), tokens.end() - 1);
        std::vector<int> target_tokens(tokens.begin() + 1, tokens.end());

        // Convert to tensors [B, T]
        tensor<T> input = to_tensor(input_tokens);
        tensor<T> target = to_tensor(target_tokens);

        // Set attention mask for this sequence length
        set_attention_mask(input_tokens.size());

        // Perform training step
        this->train_step(input, target);
    }

    const Tokenizer* tokenizer() const { return tokenizer_.get(); }

    // Set attention mask for a specific sequence length
    void set_attention_mask(int seq_len) {
        if (seq_len > max_seq_len_) {
            throw std::runtime_error("Sequence length exceeds maximum context length");
        }
        
        // Reshape existing tensor
        attention_mask_.reshape({1, 1, seq_len, seq_len});  // [B, 1, T, T]
        
        // Reinitialize with ones
        std::vector<T> host_mask(attention_mask_.size(), static_cast<T>(1.0f));
        attention_mask_.upload(host_mask.data());
    }

    // Get current attention mask
    const tensor<T>& attention_mask() const { return attention_mask_; }

protected:
    std::shared_ptr<Tokenizer> tokenizer_;
    int max_seq_len_;
    tensor<T> attention_mask_;  // [B, 1, T, T]

    tensor<T> to_tensor(const std::vector<int>& ids) const {
        // Convert to [B, T] tensor
        tensor<T> result({1, static_cast<int>(ids.size())});
        std::vector<T> host(ids.begin(), ids.end());
        result.upload(host.data());
        return result;
    }
};

}  // namespace dnn

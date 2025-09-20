#pragma once

#include "dnn/core/tensor.cuh"
#include "dnn/core/layer.cuh"
#include <optional>

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace dnn {

/// TokenEmbedding layer
/// Maps integer token IDs to learned vector embeddings.
/// Input: [B, T] int tensor (token IDs)
/// Output: [B, T, D] float/half tensor (embedded tokens)

template<typename T>
class TokenEmbedding : public LayerWeightBiasAsymmetric<T, int> {
public:
    TokenEmbedding(int vocab_size, int embedding_dim, int max_seq_len = 1024, bool training_enabled = false);

    tensor<T> forward(const tensor<int>& input) override;
    tensor<T> backward(const tensor<T>& grad_output) override;

    std::string name() const override { return "TokenEmbedding"; }

    void initialize_weights();

    std::vector<tensor<T>*> parameters() override { return {&this->weights_, &this->bias_}; }
    std::vector<tensor<T>*> gradients() override { return {&this->grad_weights_, &this->grad_bias_}; }

    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

private:
    int vocab_size_;
    int embedding_dim_;
    int max_seq_len_;

    std::optional<tensor<int>> input_cache_;
};

} // namespace dnn

#pragma once

#include "dnn/core/tensor.cuh"
#include "dnn/core/layer.cuh"

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace dnn {

/// TokenEmbedding layer
/// Maps integer token IDs to learned vector embeddings.
/// Input: [B, T] int tensor (token IDs)
/// Output: [B, T, D] float/half tensor (embedded tokens)

template<typename T>
class TokenEmbedding : public Layer<T> {
public:
    TokenEmbedding(int vocab_size, int embedding_dim)
        : vocab_size_(vocab_size), embedding_dim_(embedding_dim),
          embeddings_(tensor<T>({vocab_size_, embedding_dim_})),
          grad_embeddings_(tensor<T>({vocab_size_, embedding_dim_})) {
        initialize_embeddings();
    }

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    std::vector<tensor<T>*> parameters() override { return { &embeddings_ }; }
    std::vector<tensor<T>*> gradients() override { return { &grad_embeddings_ }; }

    std::string name() const override { return "TokenEmbedding"; }

    tensor<T>& weights() { return embeddings_; }
    tensor<T>& grads() { return grad_embeddings_; }

private:
    void initialize_embeddings();

    int vocab_size_;
    int embedding_dim_;
    tensor<T> embeddings_;
    tensor<T> grad_embeddings_;
};

} // namespace dnn

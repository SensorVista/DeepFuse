#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#include <string>
#include <vector>
#include <stdexcept>

namespace dnn {

template<typename T>
class EmbeddingLayer : public LayerWeightBiasAsymmetric<T, int> {
public:
    EmbeddingLayer(int num_tokens, int embedding_dim);
    ~EmbeddingLayer() override = default;

    // Forward pass: convert input indices to embeddings
    tensor<T> forward(const tensor<int>& input_indices) override;

    // Backward pass: compute gradients for embedding table
    tensor<T> backward(const tensor<T>& grad_output, const tensor<int>& input_indices) override;

    // Initialize weights with small random values
    void initialize_weights() override;

    // Access embedding table
    tensor<T>* weights() override { return &this->weights_; }
    tensor<T>* bias() override { return &this->bias_; }
    tensor<T>* grad_weights() override { return &this->grad_weights_; }
    tensor<T>* grad_bias() override { return &this->grad_bias_; }

private:
    int num_tokens_;
    int embedding_dim_;
};

} // namespace dnn

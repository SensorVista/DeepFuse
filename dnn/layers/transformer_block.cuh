#pragma once

#include "dnn/layers/layer_norm_layer.cuh"
#include "dnn/layers/multi_head_attention.cuh"
#include "dnn/layers/fully_connected_layer.cuh"
#include "dnn/core/tensor.cuh"

#include <string>

namespace dnn {

template <typename T>
class TransformerBlock : public Layer<T> {
public:
    TransformerBlock(int embed_dim, int num_heads, int mlp_hidden_dim, bool use_mask = false);
    ~TransformerBlock();

    // Set attention mask [B, 1, T, T]
    void set_mask(tensor<T>* mask);

    // Forward pass: input [B, T, E]
    tensor<T> forward(const tensor<T>& input) override;

    // Backward pass: gradient of output
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    std::string name() const override { return "TransformerBlock"; }

private:
    int embed_dim_;
    int num_heads_;
    int mlp_hidden_dim_;

    LayerNormLayer<T> norm1_;
    LayerNormLayer<T> norm2_;
    MultiHeadAttentionLayer<T> attn_;
    FullyConnectedLayer<T> mlp_fc1_;
    FullyConnectedLayer<T> mlp_fc2_;

    tensor<T> residual1_;
    tensor<T> residual2_;
    tensor<T> post_attn_;
    tensor<T> post_mlp_;
};

}  // namespace dnn

#pragma once

#include "dnn/layers/layer_norm_layer.cuh"
#include "dnn/layers/fully_connected_layer.cuh"
#include "dnn/layers/multi_head_attention.cuh"

#include <string>
#include <optional>
#include <iostream>

namespace dnn {

template <typename T>
class TransformerBlock : public Layer<T> {
public:
    TransformerBlock(int embed_dim, int num_heads, int mlp_hidden_dim, bool training_enabled = false);
    ~TransformerBlock();

    // Set attention mask [B, 1, T, T]
    void set_mask(dnn::tensor<T>* mask);

    // Forward pass: input [B, T, E]
    dnn::tensor<T> forward(const dnn::tensor<T>& input) override;

    // Backward pass: gradient of output
    dnn::tensor<T> backward(const dnn::tensor<T>& grad_output) override;

    std::string name() const override { return "TransformerBlock"; }

    // Weight initialization methods
    void initialize_all_weights() {
        attn_.initialize_weights();
        mlp_fc1_.initialize_weights();
        mlp_fc2_.initialize_weights();
    }

    void save(std::ostream& out) const override {
        out.write(reinterpret_cast<const char*>(&embed_dim_), sizeof(embed_dim_));
        out.write(reinterpret_cast<const char*>(&num_heads_), sizeof(num_heads_));
        out.write(reinterpret_cast<const char*>(&mlp_hidden_dim_), sizeof(mlp_hidden_dim_));
        norm1_.save(out);
        norm2_.save(out);
        attn_.save(out);
        mlp_fc1_.save(out);
        mlp_fc2_.save(out);
    }
    void load(std::istream& in) override {
        in.read(reinterpret_cast<char*>(&embed_dim_), sizeof(embed_dim_));
        in.read(reinterpret_cast<char*>(&num_heads_), sizeof(num_heads_));
        in.read(reinterpret_cast<char*>(&mlp_hidden_dim_), sizeof(mlp_hidden_dim_));
        norm1_.load(in);
        norm2_.load(in);
        attn_.load(in);
        mlp_fc1_.load(in);
        mlp_fc2_.load(in);
    }

private:
    int embed_dim_;
    int num_heads_;
    int mlp_hidden_dim_;

    LayerNormLayer<T> norm1_;
    LayerNormLayer<T> norm2_;
    MultiHeadAttentionLayer<T> attn_;
    FullyConnectedLayer<T> mlp_fc1_;
    FullyConnectedLayer<T> mlp_fc2_;

    dnn::tensor<T> residual1_;
    dnn::tensor<T> residual2_;
    dnn::tensor<T> post_attn_;
    dnn::tensor<T> post_mlp_;

    dnn::tensor<T> normed2_unshaped_;
    dnn::tensor<T> mlp_hidden_;

    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<T>> input_cache_;
};

}  // namespace dnn

#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"
#include "dnn/layers/fully_connected_layer.cuh"
#include "dnn/layers/softmax_layer.cuh"
#include <optional>

namespace dnn {

// Multi-head attention layer from "Attention Is All You Need"
// Input shape:  [B, T, E] - batch, sequence length, embedding dim
// Output shape: [B, T, E] - same as input
// Internal shapes:
//   Q,K,V: [B, T, E] -> [B, T, H, D] -> [B*H, T, D]
//   Attention: [B*H, T, T] -> [B, H, T, T]
template<typename TT>
class MultiHeadAttentionLayer : public Layer<TT> {
public:
    MultiHeadAttentionLayer(int embed_dim, int num_heads, bool training_enabled = false);
    ~MultiHeadAttentionLayer() override;

    tensor<TT> forward(const tensor<TT>& input) override;
    tensor<TT> backward(const tensor<TT>& grad_output) override;

    void set_mask(tensor<TT>* mask); // Optional attention mask [B, 1, T, T]

    std::string name() const override { return "MultiHeadAttention"; }

    // Get layer configuration
    int embed_dim() const { return embed_dim_; }
    int num_heads() const { return num_heads_; }
    int head_dim() const { return head_dim_; }

    // For visualization/debugging
    tensor<TT>* get_last_attention() { return &last_attn_; }
    tensor<TT>* get_q_weights() { return qkv_proj_.weights(); }
    tensor<TT>* get_last_q() { return &last_q_; }

    void initialize_weights();

    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

private:

    int embed_dim_;
    int num_heads_;
    int head_dim_;
    bool use_mask_;

    // QKV projection layer (single FC for all three)
    FullyConnectedLayer<TT> qkv_proj_;
    FullyConnectedLayer<TT> out_proj_;

    // Softmax for attention weights
    SoftmaxLayer<TT> softmax_;

    tensor<TT> last_input_; // Cache for backward
    tensor<TT> last_q_, last_k_, last_v_, last_attn_;
    tensor<TT> last_logits_; // Cache for softmax backward
    tensor<TT> last_out_proj_input_; // Cache for output projection backward

    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<TT>> input_cache_;
};

}  // namespace dnn

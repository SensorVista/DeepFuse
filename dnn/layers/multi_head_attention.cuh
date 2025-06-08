#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"
#include "dnn/layers/fully_connected_layer.cuh"
#include "dnn/layers/softmax_layer.cuh"

namespace dnn {

template<typename T>
class MultiHeadAttentionLayer : public Layer<T> {
public:
    MultiHeadAttentionLayer(int embed_dim, int num_heads, bool use_mask = true);
    ~MultiHeadAttentionLayer();

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    void set_mask(tensor<T>* mask); // Optional attention mask [B, 1, T, T]

    std::string name() const override { return "MultiHeadAttention"; }

private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;

    // Q, K, V projection layers
    FullyConnectedLayer<T> q_proj_;
    FullyConnectedLayer<T> k_proj_;
    FullyConnectedLayer<T> v_proj_;
    FullyConnectedLayer<T> out_proj_;

    // Softmax for attention weights
    SoftmaxLayer<T> softmax_;

    tensor<T>* mask_; // External mask (non-owning)
    tensor<T> last_input_; // Cache for backward
    tensor<T> last_q_, last_k_, last_v_, last_attn_;
};

}  // namespace dnn

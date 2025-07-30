#include "transformer_block.cuh"
#include "../core/tensor.cuh"
#include "../utils/common.cuh"

#include <optional>

using namespace dnn::utils;

namespace dnn {

template<typename TT>
TransformerBlock<TT>::TransformerBlock(int embed_dim, int num_heads, int mlp_hidden_dim, bool training_enabled)
            : Layer<TT>(training_enabled)
            , embed_dim_(embed_dim)
            , num_heads_(num_heads)
            , mlp_hidden_dim_(mlp_hidden_dim)
            , norm1_(embed_dim, 1e-5f, true, training_enabled)
            , norm2_(embed_dim, 1e-5f, true, training_enabled)
            , attn_(embed_dim, num_heads, training_enabled)
            , mlp_fc1_(embed_dim, mlp_hidden_dim, training_enabled)
            , mlp_fc2_(mlp_hidden_dim, embed_dim, training_enabled)
            , residual1_({1, 1, embed_dim})
            , residual2_({1, 1, embed_dim})
            , post_attn_({1, 1, embed_dim})
            , post_mlp_({1, 1, embed_dim})
            , normed2_unshaped_({1, 1, embed_dim})
            , mlp_hidden_({1, 1, mlp_hidden_dim}) {
}

template<typename TT>
TransformerBlock<TT>::~TransformerBlock() {}

template<typename TT>
void TransformerBlock<TT>::set_mask(dnn::tensor<TT>* mask) {
    // Ensure mask has correct shape [B, 1, T, T]
    if (mask->shape().size() != 4 || mask->shape()[1] != 1) {
        throw std::invalid_argument("Mask must have shape [B, 1, T, T]");
    }
        
    // Set mask in attention layer
    attn_.set_mask(mask);
}

// Forward pass: always store intermediates in [B, T, E]
template<typename TT>
dnn::tensor<TT> TransformerBlock<TT>::forward(const dnn::tensor<TT>& input) {
    if (this->training_enabled_) {
        input_cache_ = input.clone();
    }
    // input: [B, T, E]
    const auto& input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::runtime_error("TransformerBlock expects input shape [B, T, E]");
    }
    int B = input_shape[0];
    int T = input_shape[1];
    int E = input_shape[2];

    // Store input for residual connection
    residual1_ = input.clone(); // [B, T, E]

    // LayerNorm: [B, T, E] -> [B, T, E]
    dnn::tensor<TT> normed1 = norm1_.forward(input); // [B, T, E]
    // MultiHeadAttention: [B, T, E] -> [B, T, E]
    dnn::tensor<TT> attn_out = attn_.forward(normed1); // [B, T, E]
    post_attn_ = attn_out.clone(); // [B, T, E]

    // Residual Add: [B, T, E] + [B, T, E] -> [B, T, E]
    dnn::tensor<TT> add1 = attn_out + residual1_; // [B, T, E]
    residual2_ = add1.clone(); // [B, T, E]

    // LayerNorm: [B, T, E] -> [B, T, E]
    dnn::tensor<TT> normed2 = norm2_.forward(add1); // [B, T, E]
    normed2_unshaped_ = normed2.clone(); // [B, T, E]

    // --- MLP/FC: [B, T, E] -> [B*T, E] -> [B*T, H] -> [B*T, E] -> [B, T, E] ---
    dnn::tensor<TT> mlp_in = normed2.clone();
    mlp_in.reshape({B * T, E}); // [B*T, E]
    dnn::tensor<TT> hidden = mlp_fc1_.forward(mlp_in); // [B*T, H]
    mlp_hidden_ = hidden.clone(); // Cache for backward
    dnn::tensor<TT> output = mlp_fc2_.forward(hidden); // [B*T, E]
    output.reshape({B, T, E}); // [B, T, E]
    post_mlp_ = output.clone(); // [B, T, E]

    // Residual Add: [B, T, E] + [B, T, E] -> [B, T, E]
    dnn::tensor<TT> final_out = output + residual2_; // [B, T, E]

    // Defensive: check for null device pointers before returning
    if (!final_out.data()) {
        throw std::runtime_error("Null device pointer in final_out in TransformerBlock::forward");
    }
    return final_out;
}

// Backward pass: always expect intermediates in [B, T, E]
template<typename TT>
dnn::tensor<TT> TransformerBlock<TT>::backward(const dnn::tensor<TT>& grad_output) {
    if (this->training_enabled_) {
        if (!input_cache_.has_value()) {
            throw std::runtime_error("TransformerBlock: input_cache_ is empty in backward().");
        }
    }
    const dnn::tensor<TT>& input = this->training_enabled_ ? input_cache_.value() : grad_output; // fallback for stateless
    const auto& input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::runtime_error("TransformerBlock expects input shape [B, T, E] in backward");
    }
    int B = input_shape[0];
    int T = input_shape[1];
    int E = input_shape[2];

    // Allocation checks for member tensors
    #define CHECK_TENSOR_ALLOC(tensor, name) \
        if (!(tensor).data()) throw std::runtime_error("Tensor not allocated: " name);
    CHECK_TENSOR_ALLOC(post_mlp_, "post_mlp_");
    CHECK_TENSOR_ALLOC(normed2_unshaped_, "normed2_unshaped_");
    CHECK_TENSOR_ALLOC(residual2_, "residual2_");
    CHECK_TENSOR_ALLOC(residual1_, "residual1_");
    CHECK_TENSOR_ALLOC(post_attn_, "post_attn_");

    // Residual gradient from final add
    dnn::tensor<TT> grad_mlp = grad_output.clone(); // [B, T, E]

    // --- MLP/FC backward: [B, T, E] -> [B*T, E] ---
    dnn::tensor<TT> grad_mlp_flat = grad_mlp.clone();
    grad_mlp_flat.reshape({B * T, E}); // [B*T, E]
    dnn::tensor<TT> mlp_hidden_flat = mlp_hidden_.clone();
    mlp_hidden_flat.reshape({B * T, mlp_hidden_dim_}); // [B*T, H]
    dnn::tensor<TT> grad_fc2 = mlp_fc2_.backward(grad_mlp_flat); // [B*T, H]
    grad_fc2.reshape({B, T, mlp_hidden_dim_}); // [B, T, H]
    dnn::tensor<TT> grad_fc1 = grad_fc2.clone();
    grad_fc1.reshape({B * T, mlp_hidden_dim_}); // [B*T, H]
    dnn::tensor<TT> normed2_unshaped_flat = normed2_unshaped_.clone();
    normed2_unshaped_flat.reshape({B * T, E}); // [B*T, E]
    grad_fc1 = mlp_fc1_.backward(grad_fc1); // [B*T, E]
    grad_fc1.reshape({B, T, E}); // [B, T, E]

    // LayerNorm backward: [B, T, E]
    dnn::tensor<TT> grad_norm2 = norm2_.backward(grad_fc1); // [B, T, E]
    grad_norm2.reshape({B, T, E}); // [B, T, E]
    grad_mlp.reshape({B, T, E}); // [B, T, E]
    grad_norm2 += grad_mlp; // [B, T, E]

    // Attention backward: [B, T, E]
    dnn::tensor<TT> grad_attn_out = attn_.backward(grad_norm2); // [B, T, E]
    dnn::tensor<TT> grad_norm1 = norm1_.backward(grad_attn_out); // [B, T, E]
    grad_norm1 += grad_attn_out; // [B, T, E]

    // Defensive: check for null device pointers before returning
    if (!grad_norm1.data()) {
        throw std::runtime_error("Null device pointer in grad_norm1 in TransformerBlock::backward");
    }
    return grad_norm1;
}

// Instantiate
template class TransformerBlock<float>;
template class TransformerBlock<__half>;

}  // namespace dnn

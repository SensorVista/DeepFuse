#include "dnn/layers/transformer_block.cuh"
#include "dnn/utils/common.cuh"

namespace dnn {

using namespace dnn::utils;

template<typename T>
TransformerBlock<T>::TransformerBlock(int embed_dim, int num_heads, int mlp_hidden_dim, bool use_mask)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      mlp_hidden_dim_(mlp_hidden_dim),
      norm1_(embed_dim),
      norm2_(embed_dim),
      attn_(embed_dim, num_heads, use_mask),
      mlp_fc1_(embed_dim, mlp_hidden_dim),
      mlp_fc2_(mlp_hidden_dim, embed_dim),
      // Initialize member tensors with minimal shapes
      residual1_(std::vector<int>{1, 1, 1}),
      residual2_(std::vector<int>{1, 1, 1}),
      post_attn_(std::vector<int>{1, 1, 1}),
      post_mlp_(std::vector<int>{1, 1, 1}) {}

template<typename T>
TransformerBlock<T>::~TransformerBlock() {}

template<typename T>
void TransformerBlock<T>::set_mask(tensor<T>* mask) {
    attn_.set_mask(mask);
}

template<typename T>
tensor<T> TransformerBlock<T>::forward(const tensor<T>& input) {
    // Resize and copy residual connection
    residual1_.reshape(input.shape());
    residual1_.copy_from(input);

    // Pre-Norm + Attention
    tensor<T> normed1 = norm1_.forward(input);
    tensor<T> attn_out = attn_.forward(normed1);
    
    // Resize and copy attention output
    post_attn_.reshape(attn_out.shape());
    post_attn_.copy_from(attn_out);

    // Residual Add
    tensor<T> add1(attn_out.shape());
    add1.copy_from(attn_out);
    for (int i = 0; i < add1.size(); ++i) {
        add1.data()[i] += residual1_.data()[i];
    }

    // Resize and copy second residual
    residual2_.reshape(add1.shape());
    residual2_.copy_from(add1);

    // Pre-Norm + MLP
    tensor<T> normed2 = norm2_.forward(add1);
    tensor<T> hidden = mlp_fc1_.forward(normed2);
    tensor<T> output = mlp_fc2_.forward(hidden);
    
    // Resize and copy MLP output
    post_mlp_.reshape(output.shape());
    post_mlp_.copy_from(output);

    // Residual Add
    tensor<T> final_out(output.shape());
    final_out.copy_from(output);
    for (int i = 0; i < final_out.size(); ++i) {
        final_out.data()[i] += residual2_.data()[i];
    }

    return final_out;
}

template<typename T>
tensor<T> TransformerBlock<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    // Residual gradient from final add
    tensor<T> grad_mlp(post_mlp_.shape());
    grad_mlp.copy_from(grad_output);

    // Backprop MLP
    tensor<T> grad_fc2 = mlp_fc2_.backward(grad_mlp, mlp_fc1_.forward(norm2_.forward(residual2_)));
    tensor<T> grad_fc1 = mlp_fc1_.backward(grad_fc2, norm2_.forward(residual2_));
    tensor<T> grad_norm2 = norm2_.backward(grad_fc1, residual2_);

    // Add to residual
    for (int i = 0; i < grad_norm2.size(); ++i) {
        grad_norm2.data()[i] += grad_mlp.data()[i];
    }

    // Backprop Attention
    tensor<T> grad_attn_out = attn_.backward(grad_norm2, norm1_.forward(residual1_));
    tensor<T> grad_norm1 = norm1_.backward(grad_attn_out, residual1_);

    // Add to residual
    for (int i = 0; i < grad_norm1.size(); ++i) {
        grad_norm1.data()[i] += grad_attn_out.data()[i];
    }

    return grad_norm1;
}

// Instantiate
template class TransformerBlock<float>;
template class TransformerBlock<__half>;

}  // namespace dnn

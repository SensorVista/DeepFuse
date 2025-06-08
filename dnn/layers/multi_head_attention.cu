#include "dnn/layers/multi_head_attention.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>
#include <cassert>
#include <vector>

namespace dnn {

using namespace dnn::utils;

// Matrix multiplication helper
template<typename T>
void matmul(const tensor<T>& a, const tensor<T>& b, tensor<T>& c, bool transpose_a, bool transpose_b, float alpha = 1.0f) {
    // TODO: Implement proper matrix multiplication using cuBLAS
    // For now, just a placeholder that copies data
    c.copy_from(a);
}

template<typename T>
MultiHeadAttentionLayer<T>::MultiHeadAttentionLayer(int embed_dim, int num_heads, bool use_mask)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      q_proj_(embed_dim, embed_dim),
      k_proj_(embed_dim, embed_dim),
      v_proj_(embed_dim, embed_dim),
      out_proj_(embed_dim, embed_dim),
      softmax_(use_mask),
      mask_(nullptr),
      last_input_({1, 1}),  // Initialize with minimal shape
      last_q_({1, 1}),
      last_k_({1, 1}),
      last_v_({1, 1}),
      last_attn_({1, 1})
{
    assert(embed_dim % num_heads == 0 && "embed_dim must be divisible by num_heads");
}

template<typename T>
MultiHeadAttentionLayer<T>::~MultiHeadAttentionLayer() {}

template<typename T>
void MultiHeadAttentionLayer<T>::set_mask(tensor<T>* mask) {
    mask_ = mask;
    softmax_.set_mask(mask);
}

template<typename T>
tensor<T> MultiHeadAttentionLayer<T>::forward(const tensor<T>& input) {
    const auto& shape = input.shape();  // [B, T, E]
    int batch_size = shape[0];
    int seq_len = shape[1];
    int E = embed_dim_;
    int H = num_heads_;
    int D = head_dim_;

    // Project to Q, K, V: [B, T, E]
    last_q_ = q_proj_.forward(input);
    last_k_ = k_proj_.forward(input);
    last_v_ = v_proj_.forward(input);

    // Reshape to [B, T, H, D]
    last_q_.reshape({batch_size, seq_len, H, D});
    last_k_.reshape({batch_size, seq_len, H, D});
    last_v_.reshape({batch_size, seq_len, H, D});

    // Create tensors for transposed views
    tensor<T> Q({batch_size, H, seq_len, D});
    tensor<T> K({batch_size, H, seq_len, D});
    tensor<T> V({batch_size, H, seq_len, D});

    // Copy and transpose
    Q.copy_from(last_q_);
    K.copy_from(last_k_);
    V.copy_from(last_v_);
    transpose(Q);  // In-place transpose
    transpose(K);  // In-place transpose
    transpose(V);  // In-place transpose

    // Compute attention scores: [B, H, T, T]
    tensor<T> attn_scores({batch_size, H, seq_len, seq_len});
    matmul(Q, K, attn_scores, true, false, 1.0f / sqrtf(D));  // Q·Kᵀ / sqrt(D)

    // Softmax over attention scores
    last_attn_ = softmax_.forward(attn_scores);  // [B, H, T, T]

    // Attention * V → [B, H, T, D]
    tensor<T> attn_out({batch_size, H, seq_len, D});
    matmul(last_attn_, V, attn_out, false, false);  // A·V

    // Create tensor for transposed view
    tensor<T> attn_out_transposed({batch_size, seq_len, H, D});
    attn_out_transposed.copy_from(attn_out);
    transpose(attn_out_transposed);  // In-place transpose
    attn_out_transposed.reshape({batch_size, seq_len, E});

    // Final projection: [B, T, E]
    tensor<T> output = out_proj_.forward(attn_out_transposed);
    return output;
}

template<typename T>
tensor<T> MultiHeadAttentionLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    int batch_size = input.shape()[0];
    int seq_len = input.shape()[1];
    int E = embed_dim_;
    int H = num_heads_;
    int D = head_dim_;

    // dOut → out_proj backward
    tensor<T> d_attn_out_flat = out_proj_.backward(grad_output, last_attn_);

    // Reshape: [B, T, E] → [B, T, H, D] → [B, H, T, D]
    d_attn_out_flat.reshape({batch_size, seq_len, H, D});
    tensor<T> d_attn_out({batch_size, H, seq_len, D});
    d_attn_out.copy_from(d_attn_out_flat);
    transpose(d_attn_out);  // In-place transpose

    // V backward from A·V
    tensor<T> d_attn({batch_size, H, seq_len, seq_len});
    tensor<T> d_v({batch_size, H, seq_len, D});
    matmul(last_attn_, d_attn_out, d_v, true, false);           // dV = Aᵀ·dY
    matmul(d_attn_out, last_v_, d_attn, false, true);           // dA = dY·Vᵀ

    // Softmax backward
    tensor<T> d_scores = softmax_.backward(d_attn, last_attn_);

    // Q backward from Q·Kᵀ
    tensor<T> d_q({batch_size, H, seq_len, D});
    tensor<T> d_k({batch_size, H, seq_len, D});
    matmul(d_scores, last_k_, d_q, false, false);               // dQ = dS·K
    matmul(d_scores, last_q_, d_k, false, true);                // dK = dSᵀ·Q

    // Transpose & reshape back to [B, T, E]
    tensor<T> dq({batch_size, seq_len, H, D});
    tensor<T> dk({batch_size, seq_len, H, D});
    tensor<T> dv({batch_size, seq_len, H, D});
    dq.copy_from(d_q);
    dk.copy_from(d_k);
    dv.copy_from(d_v);
    transpose(dq);  // In-place transpose
    transpose(dk);  // In-place transpose
    transpose(dv);  // In-place transpose

    dq.reshape({batch_size, seq_len, E});
    dk.reshape({batch_size, seq_len, E});
    dv.reshape({batch_size, seq_len, E});

    // Final input gradient: sum of q_proj, k_proj, v_proj backprops
    tensor<T> dx_q = q_proj_.backward(dq, input);
    tensor<T> dx_k = k_proj_.backward(dk, input);
    tensor<T> dx_v = v_proj_.backward(dv, input);

    tensor<T> dx(input.shape());
    dx.zero();
    dx += dx_q;
    dx += dx_k;
    dx += dx_v;
    return dx;
}

// Instantiate for float and half
template class MultiHeadAttentionLayer<float>;
template class MultiHeadAttentionLayer<__half>;

}  // namespace dnn

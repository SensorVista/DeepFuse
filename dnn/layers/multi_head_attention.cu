#include "dnn/layers/multi_head_attention.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>
#include <cassert>
#include <vector>

namespace dnn {

// Multi-head attention from "Attention Is All You Need" (Vaswani et al., 2017)
// Key insight: Parallel attention heads allow model to focus on different positions simultaneously

// Matrix multiplication kernel for attention computation
template<typename T>
__global__ void matmul_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int m,
    int k,
    int n,
    bool transpose_a,
    bool transpose_b,
    float alpha
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            int a_idx = transpose_a ? (i * m + row) : (row * k + i);
            int b_idx = transpose_b ? (col * k + i) : (i * n + col);
            sum += static_cast<float>(a[a_idx]) * static_cast<float>(b[b_idx]);
        }
        c[row * n + col] = static_cast<T>(sum * alpha);
    }
}

// Reshape kernel: [B, T, H, D] -> [B*H, T, D] for parallel head processing
template<typename T>
__global__ void reshape_bthd_to_bhtd_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * seq_len * num_heads * head_dim;
    
    if (idx < total_size) {
        // Calculate original indices
        int d = idx % head_dim;
        int h = (idx / head_dim) % num_heads;
        int t = (idx / (head_dim * num_heads)) % seq_len;
        int b = idx / (head_dim * num_heads * seq_len);
        
        // Calculate new index
        int new_idx = ((b * num_heads + h) * seq_len + t) * head_dim + d;
        output[new_idx] = input[idx];
    }
}

// Reshape kernel: [B*H, T, D] -> [B, T, H, D] for output projection
template<typename T>
__global__ void reshape_bhtd_to_bthd_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * seq_len * num_heads * head_dim;
    
    if (idx < total_size) {
        // Calculate original indices
        int d = idx % head_dim;
        int t = (idx / head_dim) % seq_len;
        int h = (idx / (head_dim * seq_len)) % num_heads;
        int b = idx / (head_dim * seq_len * num_heads);
        
        // Calculate new index
        int new_idx = ((b * seq_len + t) * num_heads + h) * head_dim + d;
        output[new_idx] = input[idx];
    }
}

// Add mask application kernel declaration at the top with other kernels
template<typename T>
__global__ void apply_mask_kernel(
    T* __restrict__ scores,
    const T* __restrict__ mask,
    int batch_size,
    int num_heads,
    int seq_len,
    float mask_value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * num_heads * seq_len * seq_len;
    
    if (idx < total_size) {
        // Calculate indices
        int s = idx % seq_len;  // source position
        int t = (idx / seq_len) % seq_len;  // target position
        int h = (idx / (seq_len * seq_len)) % num_heads;
        int b = idx / (seq_len * seq_len * num_heads);
        
        // Get mask value for this position
        // Mask shape is [B, 1, T, T], so we need to adjust the index
        // For each batch, we have 1 mask that's shared across all heads
        int mask_idx = b * seq_len * seq_len + t * seq_len + s;  // Skip the 1 dimension since it's shared
        T mask_val = mask[mask_idx];
        
        // Apply mask: add large negative value to masked positions
        // For causal masking, we want to mask future positions (s > t)
        // For explicit masking, we check if mask_val is negative (masked)
        if (s > t || static_cast<float>(mask_val) < 0.0f) {
            scores[idx] = static_cast<T>(mask_value);  // e.g., -1e9
        }
    }
}

// Add mask application kernel declaration at the top with other kernels
template<typename T>
__global__ void apply_attention_mask_kernel(
    T* __restrict__ scores,
    const T* __restrict__ mask,
    int batch_size,
    int num_heads,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * num_heads * seq_len * seq_len;
    
    if (idx < total_size) {
        // Calculate indices
        int s = idx % seq_len;  // source position
        int t = (idx / seq_len) % seq_len;  // target position
        int h = (idx / (seq_len * seq_len)) % num_heads;
        int b = idx / (seq_len * seq_len * num_heads);
        
        // Get mask value for this position
        // Mask shape is [B, 1, T, T], so we need to adjust the index
        int mask_idx = b * seq_len * seq_len + t * seq_len + s;  // Skip the 1 dimension since it's shared
        T mask_val = mask[mask_idx];
        
        // Apply mask: add large negative value to masked positions
        if (static_cast<float>(mask_val) < 0.0f) {
            if constexpr (std::is_same<T, __half>::value)
                scores[idx] = __float2half(-1e4f);  // -1e9 is too large for half
            else
                scores[idx] = static_cast<T>(-1e9f);
        }
    }
}

template<typename TT>
MultiHeadAttentionLayer<TT>::MultiHeadAttentionLayer(int embed_dim, int num_heads, bool training_enabled)
    : Layer<TT>(training_enabled),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      qkv_proj_(embed_dim, 3 * embed_dim, training_enabled),
      out_proj_(embed_dim, embed_dim, training_enabled),
      softmax_(training_enabled),
      last_input_({1, 1}),
      last_q_({1, 1}),
      last_k_({1, 1}),
      last_v_({1, 1}),
      last_attn_({1, 1}),
      last_logits_({1, 1}),
      last_out_proj_input_({1, 1})
{
    if (embed_dim <= 0 || num_heads <= 0) {
        throw std::runtime_error("embed_dim and num_heads must be positive");
    }
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("embed_dim must be divisible by num_heads");
    }
    initialize_weights();
}

template<typename TT>
MultiHeadAttentionLayer<TT>::~MultiHeadAttentionLayer() {}

template<typename TT>
void MultiHeadAttentionLayer<TT>::set_mask(tensor<TT>* mask) {
    if (mask && (mask->shape().size() != 4 || mask->shape()[1] != 1)) {
        throw std::runtime_error("Mask must be 4D tensor with shape [B, 1, T, T]");
    }
    softmax_.set_mask(mask);
}

template<typename TT>
void MultiHeadAttentionLayer<TT>::initialize_weights() {
    qkv_proj_.initialize_weights();
    out_proj_.initialize_weights();
}

template<typename TT>
tensor<TT> MultiHeadAttentionLayer<TT>::forward(const tensor<TT>& input) {
    if (this->training_enabled_) {
        input_cache_ = input.clone();
    }
    const auto& shape = input.shape();  // [B, T, E]
    if (shape.size() != 3) {
        throw std::runtime_error("Input tensor must be 3D [B, T, E]");
    }

    int batch_size = shape[0];
    int seq_len = shape[1];
    int E = embed_dim_;
    int H = num_heads_;
    int D = head_dim_;

    // Flatten input: [B, T, E] -> [B*T, E]
    tensor<TT> input_flat = input.clone();
    input_flat.reshape({batch_size * seq_len, E});

    // Project: [B*T, E] -> [B*T, 3*E]
    auto qkv_proj_output = qkv_proj_.forward(input_flat);

    // Reshape: [B*T, 3*E] -> [B, T, 3, E]
    qkv_proj_output.reshape({batch_size, seq_len, 3, E});

    // Split Q, K, V using clone, then slice and reshape
    tensor<TT> q_proj_output = qkv_proj_output.clone();
    q_proj_output.slice(2, 0, 1).reshape({batch_size, seq_len, E});
    tensor<TT> k_proj_output = qkv_proj_output.clone();
    k_proj_output.slice(2, 1, 2).reshape({batch_size, seq_len, E});
    tensor<TT> v_proj_output = qkv_proj_output.clone();
    v_proj_output.slice(2, 2, 3).reshape({batch_size, seq_len, E});

    // Continue as before
    last_q_ = q_proj_output.clone();
    last_q_.reshape({ batch_size, seq_len, H, D });

    last_k_ = k_proj_output.clone();
    last_k_.reshape({ batch_size, seq_len, H, D });

    last_v_ = v_proj_output.clone();
    last_v_.reshape({ batch_size, seq_len, H, D });

    tensor<TT> Q({ batch_size * H, seq_len, D });
    tensor<TT> K({ batch_size * H, seq_len, D });
    tensor<TT> V({ batch_size * H, seq_len, D });

    dim3 blockDim(256);
    dim3 gridDim((batch_size * seq_len * H * D + blockDim.x - 1) / blockDim.x);

    reshape_bthd_to_bhtd_kernel<TT> << <gridDim, blockDim >> > (last_q_.data(), Q.data(), batch_size, seq_len, H, D);
    utils::CHECK_CUDA_EX(cudaGetLastError()); utils::CHECK_CUDA_EX(cudaDeviceSynchronize());
    reshape_bthd_to_bhtd_kernel<TT> << <gridDim, blockDim >> > (last_k_.data(), K.data(), batch_size, seq_len, H, D);
    utils::CHECK_CUDA_EX(cudaGetLastError()); utils::CHECK_CUDA_EX(cudaDeviceSynchronize());
    reshape_bthd_to_bhtd_kernel<TT> << <gridDim, blockDim >> > (last_v_.data(), V.data(), batch_size, seq_len, H, D);
    utils::CHECK_CUDA_EX(cudaGetLastError()); utils::CHECK_CUDA_EX(cudaDeviceSynchronize());

    tensor<TT> attn_scores({ batch_size * H, seq_len, seq_len });
    dim3 attn_blockDim(16, 16);
    dim3 attn_gridDim((seq_len + 15) / 16, (seq_len + 15) / 16);

    matmul_kernel<TT> << <attn_gridDim, attn_blockDim >> > (
        Q.data(), K.data(), attn_scores.data(),
        seq_len, D, seq_len,
        true, false, 1.0f / sqrtf(D)
        );
    utils::CHECK_CUDA_EX(cudaGetLastError()); utils::CHECK_CUDA_EX(cudaDeviceSynchronize());

    attn_scores.reshape({ batch_size, H, seq_len, seq_len });

    // Ensure last_logits_ is correct shape
    std::vector<int> attn_shape = { batch_size, H, seq_len, seq_len };
    last_logits_ = attn_scores.clone();

    last_attn_ = softmax_.forward(attn_scores);
    last_attn_.reshape({ batch_size * H, seq_len, seq_len });

    tensor<TT> attn_out({ batch_size * H, seq_len, D });
    matmul_kernel<TT> << <attn_gridDim, attn_blockDim >> > (
        last_attn_.data(), V.data(), attn_out.data(),
        seq_len, seq_len, D,
        false, false, 1.0f
        );
    utils::CHECK_CUDA_EX(cudaGetLastError()); 
    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());

    tensor<TT> output({ batch_size, seq_len, H, D });
    reshape_bhtd_to_bthd_kernel<TT> << <gridDim, blockDim >> > (
        attn_out.data(), output.data(), batch_size, seq_len, H, D
        );
    utils::CHECK_CUDA_EX(cudaGetLastError()); 
    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());

    // Reshape for output projection: [B, T, H, D] -> [B, T, E] -> [B*T, E]
    output.reshape({batch_size, seq_len, E});
    output.reshape({batch_size * seq_len, E});
    tensor<TT> final_output = out_proj_.forward(output);
    final_output.reshape({batch_size, seq_len, E});

    // Ensure last_out_proj_input_ is correct shape
    last_out_proj_input_ = output.clone();

    return final_output;
}

template<typename TT>
tensor<TT> MultiHeadAttentionLayer<TT>::backward(const tensor<TT>& grad_output) {
    if (this->training_enabled_) {
        if (!input_cache_.has_value()) {
            throw std::runtime_error("MultiHeadAttentionLayer: input_cache_ is empty in backward().");
        }
    }
    // For now, just clone grad_output (real implementation would use input_cache_)
    return grad_output.clone();
}

template<typename TT>
void MultiHeadAttentionLayer<TT>::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&embed_dim_), sizeof(embed_dim_));
    out.write(reinterpret_cast<const char*>(&num_heads_), sizeof(num_heads_));
    qkv_proj_.save(out);
    out_proj_.save(out);
    softmax_.save(out);
}

template<typename TT>
void MultiHeadAttentionLayer<TT>::load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&embed_dim_), sizeof(embed_dim_));
    in.read(reinterpret_cast<char*>(&num_heads_), sizeof(num_heads_));
    qkv_proj_.load(in);
    out_proj_.load(in);
    softmax_.load(in);
}

// Instantiate for float and half
template class MultiHeadAttentionLayer<float>;
template class MultiHeadAttentionLayer<__half>;

}  // namespace dnn

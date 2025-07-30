#include "dnn/tokens/positional_encoding.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

namespace dnn {

// CUDA kernel to initialize positional encodings
template<typename T>
__global__ void init_positional_encodings_kernel(
    T* pos_encoding,
    int max_seq_len,
    int embed_dim
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos < max_seq_len && i < embed_dim) {
        float div_term = expf(-2.0f * (i / 2) * logf(10000.0f) / embed_dim);
        if (i % 2 == 0) {
            pos_encoding[pos * embed_dim + i] = sinf(pos * div_term);
        } else {
            pos_encoding[pos * embed_dim + i] = cosf(pos * div_term);
        }
    }
}

// CUDA kernel to add positional encodings to input
template<typename T>
__global__ void add_positional_encodings_kernel(
    T* output,
    const T* input,
    const T* pos_encoding,
    int batch_size,
    int seq_len,
    int embed_dim
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;

    if (b < batch_size && t < seq_len && d < embed_dim) {
        int idx = (b * seq_len + t) * embed_dim + d;
        output[idx] = input[idx] + pos_encoding[t * embed_dim + d];
    }
}

template<typename T>
PositionalEncoding<T>::PositionalEncoding(int embed_dim, int max_seq_len, bool training_enabled)
    : Layer<T>(training_enabled),
      embed_dim_(embed_dim),
      max_seq_len_(max_seq_len),
      pos_encoding_({max_seq_len, embed_dim}) {
    // Validate parameters before any CUDA operations
    if (embed_dim <= 0) {
        throw std::invalid_argument("Embedding dimension must be positive");
    }
    if (max_seq_len <= 0) {
        throw std::invalid_argument("Maximum sequence length must be positive");
    }
    initialize_positional_encodings();
}

template<typename T>
void PositionalEncoding<T>::initialize_positional_encodings() {
    dim3 block(16, 16);
    dim3 grid(
        (max_seq_len_ + block.x - 1) / block.x,
        (embed_dim_ + block.y - 1) / block.y
    );

    init_positional_encodings_kernel<<<grid, block>>>(
        pos_encoding_.data(),
        max_seq_len_,
        embed_dim_
    );

    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());
}

template<typename T>
tensor<T> PositionalEncoding<T>::forward(const tensor<T>& input) {
    if (this->training_enabled_) {
        input_cache_ = input.clone();
    }
    // input shape: [batch_size, seq_len, embed_dim]
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    
    tensor<T> output = input.clone();  // Copy input to output first

    dim3 block(embed_dim_);
    dim3 grid(batch_size, seq_len);

    add_positional_encodings_kernel<<<grid, block>>>(
        output.data(),
        input.data(),
        pos_encoding_.data(),
        batch_size,
        seq_len,
        embed_dim_
    );

    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());
    return output;
}

template<typename T>
tensor<T> PositionalEncoding<T>::backward(const tensor<T>& grad_output) {
    if (this->training_enabled_) {
        if (!input_cache_) throw std::runtime_error("PositionalEncoding: input_cache_ not set for backward");
    }
    // For positional encoding, backward is identity
    return grad_output.clone();
}

// Explicit template instantiations
template class PositionalEncoding<float>;
template class PositionalEncoding<__half>;

}  // namespace dnn 
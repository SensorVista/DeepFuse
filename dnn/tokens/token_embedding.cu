#include "token_embedding.cuh"
#include "../utils/common.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>

namespace dnn {

// Kernel: embedding initialization
template<typename T>
__global__ void embedding_init_kernel(T* weights, T* bias, int vocab_size, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size * embedding_dim) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        
        // Proper embedding initialization
        float scale = 0.02f;  // Standard scale for transformer embeddings
        float val = curand_normal(&state) * scale;
        weights[idx] = static_cast<T>(val);
    }
    
    // Initialize bias
    if (idx < embedding_dim) {
        bias[idx] = static_cast<T>(0.0f);  // Zero initialize bias
    }
}

// Kernel: forward pass
template<typename T>
__global__ void embedding_forward_kernel(const int* token_ids, T* output, const T* embeddings, const T* bias,
                                       int vocab_size, int embedding_dim, int sequence_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sequence_length * embedding_dim;

    if (idx < total) {
        int t = idx / embedding_dim;
        int d = idx % embedding_dim;
        int token = token_ids[t];
        if (token >= 0 && token < vocab_size) {  // Validate token ID
            output[idx] = embeddings[token * embedding_dim + d] + bias[d];  // Add bias
        }
    }
}

// Kernel: backward pass
template<typename T>
__global__ void embedding_backward_kernel(const int* token_ids, const T* grad_output, 
                                        T* grad_embeddings, T* grad_bias,
                                        int vocab_size, int embedding_dim, int sequence_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sequence_length * embedding_dim;

    if (idx < total) {
        int t = idx / embedding_dim;
        int d = idx % embedding_dim;
        int token = token_ids[t];
        if (token >= 0 && token < vocab_size) {  // Validate token ID
            T grad = grad_output[idx];
            
            // Update embedding gradients
            atomicAdd(&grad_embeddings[token * embedding_dim + d], grad);
            
            // Update bias gradients
            atomicAdd(&grad_bias[d], grad);
        }
    }
}

template<typename T>
TokenEmbedding<T>::TokenEmbedding(int vocab_size, int embedding_dim, int max_seq_len, bool training_enabled)
    : LayerWeightBiasAsymmetric<T, int>(
        tensor<T>({vocab_size, embedding_dim}),  // weights
        tensor<T>({embedding_dim}),              // bias
        tensor<T>({vocab_size, embedding_dim}),  // grad_weights
        tensor<T>({embedding_dim}),              // grad_bias
        training_enabled),
      vocab_size_(vocab_size),
      embedding_dim_(embedding_dim),
      max_seq_len_(max_seq_len)
{
    // Validate parameters before any CUDA operations
    if (vocab_size <= 0) {
        throw std::invalid_argument("Vocabulary size must be positive");
    }
    if (embedding_dim <= 0) {
        throw std::invalid_argument("Embedding dimension must be positive");
    }
    if (max_seq_len <= 0) {
        throw std::invalid_argument("Maximum sequence length must be positive");
    }
    initialize_weights();
}

template<typename T>
void TokenEmbedding<T>::initialize_weights() {
    const int block = 256;
    int total = vocab_size_ * embedding_dim_;
    int grid = (total + block - 1) / block;

    embedding_init_kernel<<<grid, block>>>(
        this->weights_.data(),
        this->bias_.data(),
        vocab_size_,
        embedding_dim_
    );
    utils::THROW_CUDA_EX();
    
    // Zero gradients
    this->grad_weights_.zero();
    this->grad_bias_.zero();
}

template<typename T>
tensor<T> TokenEmbedding<T>::forward(const tensor<int>& input) {
    if (this->training_enabled_) {
        input_cache_ = input.clone();
    }
    const auto& in_shape = input.shape();  // [B, T]
    if (in_shape.size() != 2)
        throw std::runtime_error("TokenEmbedding input must be 2D [B, T]");

    int batch_size = in_shape[0];
    int seq_len = in_shape[1];
    
    // Validate sequence length
    if (seq_len > max_seq_len_)
        throw std::runtime_error("Input sequence length exceeds maximum allowed length");

    // Validate token IDs
    std::vector<int> host_data(input.size());
    input.download(host_data.data());
    for (int token : host_data) {
        if (token < 0 || token >= vocab_size_) {
            throw std::runtime_error("Invalid token ID: " + std::to_string(token));
        }
    }

    tensor<T> output({batch_size, seq_len, embedding_dim_});

    const int block = 256;
    int total = batch_size * seq_len * embedding_dim_;
    int grid = (total + block - 1) / block;

    embedding_forward_kernel<<<grid, block>>>(
        input.data(),
        output.data(),
        this->weights_.data(),
        this->bias_.data(),
        vocab_size_,
        embedding_dim_,
        batch_size * seq_len
    );
    utils::THROW_CUDA_EX();
    return output;
}

template<typename T>
tensor<T> TokenEmbedding<T>::backward(const tensor<T>& grad_output) {
    if (!this->training_enabled_ || !input_cache_) {
        throw std::runtime_error("TokenEmbedding::backward requires cached input in training mode.");
    }
    const tensor<int>& input = *input_cache_;
    const auto& in_shape = input.shape();  // [B, T]
    int batch_size = in_shape[0];
    int seq_len = in_shape[1];

    grad_weights_.zero();
    grad_bias_.zero();

    const int block = 256;
    int total_elements = batch_size * seq_len * embedding_dim_;
    int grid = (total_elements + block - 1) / block;

    embedding_backward_kernel<<<grid, block>>>(
        input.data(),
        grad_output.data(),
        grad_weights_.data(),
        grad_bias_.data(),
        vocab_size_,
        embedding_dim_,
        batch_size * seq_len
    );
    utils::THROW_CUDA_EX();

    // Return gradient with same shape as input
    return tensor<T>(in_shape);
}

template<typename T>
void TokenEmbedding<T>::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&vocab_size_), sizeof(vocab_size_));
    out.write(reinterpret_cast<const char*>(&embedding_dim_), sizeof(embedding_dim_));
    out.write(reinterpret_cast<const char*>(&max_seq_len_), sizeof(max_seq_len_));
    this->weights_.save(out);
    this->bias_.save(out);
}

template<typename T>
void TokenEmbedding<T>::load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&vocab_size_), sizeof(vocab_size_));
    in.read(reinterpret_cast<char*>(&embedding_dim_), sizeof(embedding_dim_));
    in.read(reinterpret_cast<char*>(&max_seq_len_), sizeof(max_seq_len_));
    this->weights_.load(in);
    this->bias_.load(in);
}

// Explicit instantiation
template class TokenEmbedding<float>;
//template class TokenEmbedding<__half>;

} // namespace dnn

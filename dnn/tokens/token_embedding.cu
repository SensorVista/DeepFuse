#include "token_embedding.cuh"

#include "../utils/common.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>

namespace dnn {

// Kernel: kaiming uniform initialization
template<typename T>
__global__ void kaiming_uniform_kernel(T* data, int size, int fan_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        
        // Kaiming uniform initialization
        float bound = sqrt(6.0f / fan_in);
        float val = curand_uniform(&state) * 2.0f * bound - bound;
        data[idx] = static_cast<T>(val);
    }
}

// Kernel: flatten and convert to int
template<typename T>
__global__ void flatten_int_kernel(const T* input, int* output, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = static_cast<int>(input[idx]);
    }
}

// Kernel: forward pass
template<typename T>
__global__ void embedding_forward_kernel(const int* token_ids, T* output, const T* embeddings,
                                         int vocab_size, int embedding_dim, int sequence_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sequence_length * embedding_dim;

    if (idx < total) {
        int t = idx / embedding_dim;
        int d = idx % embedding_dim;
        int token = token_ids[t];
        if (token < vocab_size) {
            output[idx] = embeddings[token * embedding_dim + d];
        }
    }
}

// Kernel: backward pass
template<typename T>
__global__ void embedding_backward_kernel(const int* token_ids, const T* grad_output, T* grad_embeddings,
                                          int embedding_dim, int sequence_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sequence_length * embedding_dim;

    if (idx < total) {
        int t = idx / embedding_dim;
        int d = idx % embedding_dim;
        int token = token_ids[t];
        atomicAdd(&grad_embeddings[token * embedding_dim + d], grad_output[idx]);
    }
}

// Init
template<typename T>
void TokenEmbedding<T>::initialize_embeddings() {
    const int block = 256;
    int total = vocab_size_ * embedding_dim_;
    int grid = (total + block - 1) / block;

    kaiming_uniform_kernel<<<grid, block>>>(embeddings_.data(), total, embedding_dim_);
    utils::THROW_CUDA_EX();
    grad_embeddings_.zero();
}

// Forward
template<typename T>
tensor<T> TokenEmbedding<T>::forward(const tensor<T>& input) {
    const auto& in_shape = input.shape();  // [B, T]
    if (in_shape.size() != 2)
        throw std::runtime_error("TokenEmbedding input must be 2D [B, T]");

    int batch_size = in_shape[0];
    int seq_len = in_shape[1];
    tensor<T> output({batch_size, seq_len, embedding_dim_});

    // Create flattened int tensor
    tensor<int> flat_ids({batch_size * seq_len});
    const int block = 256;
    int total = batch_size * seq_len;
    int grid = (total + block - 1) / block;

    // Convert input to int
    flatten_int_kernel<<<grid, block>>>(input.data(), flat_ids.data(), total);
    utils::THROW_CUDA_EX();

    // Forward pass
    total = batch_size * seq_len * embedding_dim_;
    grid = (total + block - 1) / block;

    embedding_forward_kernel<<<grid, block>>>(
        flat_ids.data(),
        output.data(),
        embeddings_.data(),
        vocab_size_,
        embedding_dim_,
        batch_size * seq_len
    );
    utils::THROW_CUDA_EX();
    return output;
}

// Backward
template<typename T>
tensor<T> TokenEmbedding<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    const auto& in_shape = input.shape();  // [B, T]
    if (in_shape.size() != 2)
        throw std::runtime_error("TokenEmbedding backward input must be 2D [B, T]");

    int batch_size = in_shape[0];
    int seq_len = in_shape[1];

    // Create flattened int tensor
    tensor<int> flat_ids({batch_size * seq_len});
    const int block = 256;
    int total = batch_size * seq_len;
    int grid = (total + block - 1) / block;

    // Convert input to int
    flatten_int_kernel<<<grid, block>>>(input.data(), flat_ids.data(), total);
    utils::THROW_CUDA_EX();

    // Create a new tensor with the desired shape and copy data
    tensor<T> flat_grad({batch_size * seq_len, embedding_dim_});
    flat_grad.copy_from(grad_output);  
    
    grad_embeddings_.zero();

    total = batch_size * seq_len * embedding_dim_;
    grid = (total + block - 1) / block;

    embedding_backward_kernel<<<grid, block>>>(
        flat_ids.data(),
        flat_grad.data(),
        grad_embeddings_.data(),
        embedding_dim_,
        batch_size * seq_len
    );
    utils::THROW_CUDA_EX();
    return tensor<T>({});  // no input grad
}

// Explicit instantiation
template class TokenEmbedding<float>;

} // namespace dnn

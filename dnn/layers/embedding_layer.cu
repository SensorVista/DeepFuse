#include "dnn/layers/embedding_layer.cuh"
#include "dnn/utils/common.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

namespace dnn {

using namespace dnn::utils;

namespace {
    template<typename T>
    __global__ void lookup_kernel(T* output, const T* embeddings, const int* indices,
                                int batch_size, int seq_len, int embedding_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_size * seq_len * embedding_dim) {
            int b = idx / (seq_len * embedding_dim);
            int s = (idx / embedding_dim) % seq_len;
            int d = idx % embedding_dim;
            int word_idx = indices[b * seq_len + s];
            output[idx] = embeddings[word_idx * embedding_dim + d];
        }
    }

    template<typename T>
    __global__ void scatter_grad_kernel(T* grad_embeddings, const T* grad_output,
                                      const int* indices, int batch_size, int seq_len,
                                      int embedding_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_size * seq_len * embedding_dim) {
            int b = idx / (seq_len * embedding_dim);
            int s = (idx / embedding_dim) % seq_len;
            int d = idx % embedding_dim;
            int word_idx = indices[b * seq_len + s];
            atomicAdd(&grad_embeddings[word_idx * embedding_dim + d],
                     grad_output[idx]);
        }
    }
}

template<typename T>
EmbeddingLayer<T>::EmbeddingLayer(int num_tokens, int embedding_dim)
    : LayerWeightBiasAsymmetric<T, int>(
        tensor<T>(std::vector<int>{num_tokens, embedding_dim}),  // weights
        tensor<T>(std::vector<int>{embedding_dim}),             // bias
        tensor<T>(std::vector<int>{num_tokens, embedding_dim}), // grad_weights
        tensor<T>(std::vector<int>{embedding_dim})              // grad_bias
      ),
      num_tokens_(num_tokens),
      embedding_dim_(embedding_dim) {
    initialize_weights();
}

template<typename T>
void EmbeddingLayer<T>::initialize_weights() {
    this->weights_.fill(static_cast<T>(0.01f));
    this->bias_.zero();
}

template<typename T>
tensor<T> EmbeddingLayer<T>::forward(const tensor<int>& input_indices) {
    const auto& shape = input_indices.shape();
    int batch_size = shape[0];
    int seq_len = shape[1];

    // Create output tensor
    tensor<T> output(std::vector<int>{batch_size, seq_len, embedding_dim_});

    // Launch kernel for embedding lookup
    dim3 block(256);
    dim3 grid((batch_size * seq_len * embedding_dim_ + block.x - 1) / block.x);
    lookup_kernel<<<grid, block>>>(output.data(), this->weights_.data(),
                                 input_indices.data(),
                                 batch_size, seq_len, embedding_dim_);
    CHECK_CUDA_EX(cudaDeviceSynchronize());

    // Add bias
    for (int i = 0; i < output.size(); ++i) {
        int d = i % embedding_dim_;
        if constexpr (std::is_same<T, __half>::value) {
            float result = __half2float(output.data()[i]) + __half2float(this->bias_.data()[d]);
            output.data()[i] = __float2half(result);
        } else {
            output.data()[i] += this->bias_.data()[d];
        }
    }

    return output;
}

template<typename T>
tensor<T> EmbeddingLayer<T>::backward(const tensor<T>& grad_output, const tensor<int>& input_indices) {
    const auto& shape = input_indices.shape();
    int batch_size = shape[0];
    int seq_len = shape[1];

    // Initialize gradient tensors
    this->grad_weights_.zero();
    this->grad_bias_.zero();

    // Launch kernel for gradient scattering
    dim3 block(256);
    dim3 grid((batch_size * seq_len * embedding_dim_ + block.x - 1) / block.x);
    scatter_grad_kernel<<<grid, block>>>(this->grad_weights_.data(), grad_output.data(),
                                       input_indices.data(),
                                       batch_size, seq_len, embedding_dim_);
    CHECK_CUDA_EX(cudaDeviceSynchronize());

    // Compute bias gradient
    for (int i = 0; i < grad_output.size(); ++i) {
        int d = i % embedding_dim_;
        if constexpr (std::is_same<T, __half>::value) {
            float result = __half2float(this->grad_bias_.data()[d]) + __half2float(grad_output.data()[i]);
            this->grad_bias_.data()[d] = __float2half(result);
        } else {
            this->grad_bias_.data()[d] += grad_output.data()[i];
        }
    }

    // Return gradient for input indices (not used in practice)
    return tensor<T>(grad_output.shape());
}

// Explicit template instantiations
template class EmbeddingLayer<float>;
template class EmbeddingLayer<__half>;

} // namespace dnn 
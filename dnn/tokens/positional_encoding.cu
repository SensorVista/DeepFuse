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
PositionalEncoding<T>::PositionalEncoding(int embed_dim, int max_seq_len)
    : embed_dim_(embed_dim)
    , max_seq_len_(max_seq_len)
    , pos_encoding_({max_seq_len, embed_dim}) {
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
    // input shape: [batch_size, seq_len, embed_dim]
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    
    tensor<T> output(input.shape());
    output.copy_from(input);  // Copy input to output first

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
tensor<T> PositionalEncoding<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    // Gradient flows through unchanged since positional encodings are constants
    tensor<T> grad_input(grad_output.shape());
    grad_input.copy_from(grad_output);  // Use copy_from instead of copy constructor
    return grad_input;
}

// Explicit template instantiations
template class PositionalEncoding<float>;
template class PositionalEncoding<__half>;

}  // namespace dnn 
#include "dnn/layers/layer_norm_layer.cuh"
#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>

namespace dnn {

template<typename T>
LayerNormLayer<T>::LayerNormLayer(int norm_size, float epsilon, bool affine, bool training_enabled)
    : Layer<T>(training_enabled),
      norm_size_(norm_size),
      epsilon_(epsilon),
      affine_(affine),
      gamma_({norm_size}),
      beta_({norm_size}),
      grad_gamma_({norm_size}),
      grad_beta_({norm_size})
{
    if (affine_) {
        gamma_.fill(static_cast<T>(1));
        beta_.fill(static_cast<T>(0));
    }
}

template<typename T>
__global__ void layer_norm_forward_kernel(
    const T* input, T* output,
    const T* gamma, const T* beta,
    int B, int C, float epsilon, bool affine)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;

    if (b >= B || tid >= C) return;

    // Compute mean
    __shared__ float mean;
    __shared__ float var;

    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < C; ++i) {
            sum += static_cast<float>(input[b * C + i]);
        }
        mean = sum / C;

        float sq_sum = 0.0f;
        for (int i = 0; i < C; ++i) {
            float val = static_cast<float>(input[b * C + i]) - mean;
            sq_sum += val * val;
        }
        var = sq_sum / C;
    }

    __syncthreads();

    float rstd = rsqrtf(var + epsilon);
    float x = static_cast<float>(input[b * C + tid]);
    float norm = (x - mean) * rstd;

    float scaled = affine ? norm * static_cast<float>(gamma[tid]) + static_cast<float>(beta[tid]) : norm;
    output[b * C + tid] = static_cast<T>(scaled);
}

template<typename T>
tensor<T> LayerNormLayer<T>::forward(const tensor<T>& input) {
    if (this->training_enabled_) {
        input_cache_ = input.clone();
    }
    const auto& shape = input.shape();
    int B = shape[0];
    int C = shape[1];
    tensor<T> output(shape);
    const int threads = norm_size_;
    layer_norm_forward_kernel<<<B, threads>>>(
        input.data(), output.data(),
        affine_ ? gamma_.data() : nullptr,
        affine_ ? beta_.data() : nullptr,
        B, C, epsilon_, affine_);
    return output;
}

template<typename T>
tensor<T> LayerNormLayer<T>::backward(const tensor<T>& grad_output) {
    if (this->training_enabled_) {
        if (!input_cache_.has_value()) {
            throw std::runtime_error("LayerNormLayer: input_cache_ is empty in backward().");
        }
    }
    return grad_output.clone();
}

template<typename T>
std::vector<tensor<T>*> LayerNormLayer<T>::parameters() {
    return affine_ ? std::vector<tensor<T>*> { &gamma_, &beta_ } : std::vector<tensor<T>*>{};
}

template<typename T>
std::vector<tensor<T>*> LayerNormLayer<T>::gradients() {
    return affine_ ? std::vector<tensor<T>*> { &grad_gamma_, &grad_beta_ } : std::vector<tensor<T>*>{};
}

template<typename T>
void LayerNormLayer<T>::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&norm_size_), sizeof(norm_size_));
    gamma_.save(out);
    beta_.save(out);
    grad_gamma_.save(out);
    grad_beta_.save(out);
}

template<typename T>
void LayerNormLayer<T>::load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&norm_size_), sizeof(norm_size_));
    gamma_.load(in);
    beta_.load(in);
    grad_gamma_.load(in);
    grad_beta_.load(in);
}

// Explicit instantiation
template class LayerNormLayer<float>;
template class LayerNormLayer<__half>;

}  // namespace dnn

#include "dnn/layers/batch_norm_layer.cuh"
#include "dnn/utils/common.cuh"

#ifndef ENABLE_CUDNN
// Corrected CUDA kernel for BatchNorm2D forward (NCHW)
template<typename T>
__global__ void batchnorm_forward_kernel(
    const T* __restrict__ input, // [N, C, H, W]
    T* __restrict__ output,      // [N, C, H, W]
    T* __restrict__ mean,        // [C]
    T* __restrict__ var,         // [C]
    const T* __restrict__ gamma, // [C] (optional)
    const T* __restrict__ beta,  // [C] (optional)
    int N, int C, int H, int W,
    float epsilon, bool affine)
{
    int c = blockIdx.x;
    if (c >= C) return;
    int num = N * H * W;
    // Compute mean and variance for channel c
    float sum = 0.0f, sq_sum = 0.0f;
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                int idx = ((n * C + c) * H + h) * W + w;
                float val = static_cast<float>(input[idx]);
                sum += val;
                sq_sum += val * val;
            }
    float mean_val = sum / num;
    float var_val = sq_sum / num - mean_val * mean_val;
    mean[c] = static_cast<T>(mean_val);
    var[c] = static_cast<T>(var_val);
    float rstd = rsqrtf(var_val + epsilon);
    // Normalize and apply affine
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                int idx = ((n * C + c) * H + h) * W + w;
                float xhat = (static_cast<float>(input[idx]) - mean_val) * rstd;
                float y = xhat;
                if (affine) {
                    y = xhat * static_cast<float>(gamma[c]) + static_cast<float>(beta[c]);
                }
                output[idx] = static_cast<T>(y);
            }
}

// Corrected CUDA kernel for BatchNorm2D backward (NCHW)
template<typename T>
__global__ void batchnorm_backward_kernel(
    const T* __restrict__ grad_output, // [N, C, H, W]
    const T* __restrict__ input,       // [N, C, H, W]
    const T* __restrict__ mean,        // [C]
    const T* __restrict__ var,         // [C]
    const T* __restrict__ gamma,       // [C] (optional)
    T* __restrict__ grad_input,        // [N, C, H, W]
    T* __restrict__ grad_gamma,        // [C]
    T* __restrict__ grad_beta,         // [C]
    int N, int C, int H, int W,
    float epsilon, bool affine)
{
    int c = blockIdx.x;
    if (c >= C) return;
    int num = N * H * W;
    float mean_val = static_cast<float>(mean[c]);
    float var_val = static_cast<float>(var[c]);
    float rstd = rsqrtf(var_val + epsilon);
    // Compute grad_gamma and grad_beta
    float dgamma = 0.0f, dbeta = 0.0f;
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                int idx = ((n * C + c) * H + h) * W + w;
                float xhat = (static_cast<float>(input[idx]) - mean_val) * rstd;
                float go = static_cast<float>(grad_output[idx]);
                dgamma += go * xhat;
                dbeta  += go;
            }
    if (affine) {
        grad_gamma[c] = static_cast<T>(dgamma);
        grad_beta[c]  = static_cast<T>(dbeta);
    }
    // Compute grad_input
    float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                int idx = ((n * C + c) * H + h) * W + w;
                float xhat = (static_cast<float>(input[idx]) - mean_val) * rstd;
                float go = static_cast<float>(grad_output[idx]);
                float g = affine ? go * static_cast<float>(gamma[c]) : go;
                sum_dy += g;
                sum_dy_xhat += g * xhat;
            }
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                int idx = ((n * C + c) * H + h) * W + w;
                float xhat = (static_cast<float>(input[idx]) - mean_val) * rstd;
                float go = static_cast<float>(grad_output[idx]);
                float g = affine ? go * static_cast<float>(gamma[c]) : go;
                float dx = (g * num - sum_dy - xhat * sum_dy_xhat) * rstd / num;
                grad_input[idx] = static_cast<T>(dx);
            }
}
#endif

namespace dnn {

template<typename T>
BatchNormLayer<T>::BatchNormLayer(int num_channels, float epsilon, float momentum, bool affine, bool training_enabled)
    : Layer<T>(training_enabled),
      channels_(num_channels),
      epsilon_(epsilon),
      momentum_(momentum),
      affine_(affine),
      gamma_({num_channels}),
      beta_({num_channels}),
      grad_gamma_({num_channels}),
      grad_beta_({num_channels}),
      running_mean_({num_channels}),
      running_var_({num_channels}),
      save_mean_({num_channels}),
      save_var_({num_channels})
      // normed_input_ is constructed in forward() when shape is known
{
    // gamma: [C], beta: [C]
    // running_mean: [C], running_var: [C]
    // grad_gamma: [C], grad_beta: [C]
    // save_mean: [C], save_var: [C]
    if (affine_) {
        gamma_.fill(static_cast<T>(1));
        beta_.fill(static_cast<T>(0));
    }
    running_mean_.fill(static_cast<T>(0));
    running_var_.fill(static_cast<T>(1));
}

template<typename T>
BatchNormLayer<T>::~BatchNormLayer() {
    // All tensors auto-release
}

template<typename T>
tensor<T> BatchNormLayer<T>::forward(const tensor<T>& input) {
    if (this->training_enabled_) {
        input_cache_ = input.clone();
    }
#ifndef ENABLE_CUDNN
    // input: [N, C, H, W]
    const auto& shape = input.shape();
    int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    tensor<T> output(shape);
    save_mean_ = tensor<T>({C});
    save_var_ = tensor<T>({C});

    // Launch one block per channel
    int blocks = C;
    int threads = 1;
    // batchnorm_forward_kernel expects:
    // (input, output, mean, var, gamma, beta, N, C, H, W, epsilon, affine)
    batchnorm_forward_kernel<T><<<blocks, threads>>>(
        input.data(), output.data(),
        save_mean_.data(), save_var_.data(),
        affine_ ? gamma_.data() : nullptr,
        affine_ ? beta_.data() : nullptr,
        N, C, H, W, epsilon_, affine_
    );
    dnn::utils::THROW_CUDA_EX();

    // Update running stats (training mode)
    std::vector<T> host_mean(C), host_var(C), host_running_mean(C), host_running_var(C);
    save_mean_.download(host_mean.data());
    save_var_.download(host_var.data());
    running_mean_.download(host_running_mean.data());
    running_var_.download(host_running_var.data());
    for (int c = 0; c < C; ++c) {
        float rm = static_cast<float>(host_running_mean[c]);
        float rv = static_cast<float>(host_running_var[c]);
        float bm = static_cast<float>(host_mean[c]);
        float bv = static_cast<float>(host_var[c]);
        rm = momentum_ * rm + (1.0f - momentum_) * bm;
        rv = momentum_ * rv + (1.0f - momentum_) * bv;
        host_running_mean[c] = static_cast<T>(rm);
        host_running_var[c]  = static_cast<T>(rv);
    }
    running_mean_.upload(host_running_mean.data());
    running_var_.upload(host_running_var.data());

    return output;
#else
    // TODO: cuDNN path
    return input.clone();
#endif
}

template<typename T>
tensor<T> BatchNormLayer<T>::backward(const tensor<T>& grad_output) {
#ifndef ENABLE_CUDNN
    const tensor<T>* used_input = nullptr;
    if (this->training_enabled_) {
        if (!input_cache_) throw std::runtime_error("BatchNormLayer: input_cache_ not set for backward");
        used_input = &(*input_cache_);
    } else {
        throw std::runtime_error("BatchNormLayer: backward called without input argument in stateless mode");
    }
    // grad_output: [N, C, H, W], input: [N, C, H, W]
    const auto& shape = grad_output.shape();
    int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    tensor<T> grad_input(shape);
    grad_gamma_.zero();
    grad_beta_.zero();

    // Launch one block per channel
    int blocks = C;
    int threads = 1;
    // batchnorm_backward_kernel expects:
    // (grad_output, input, mean, var, gamma, grad_input, grad_gamma, grad_beta, N, C, H, W, epsilon, affine)
    batchnorm_backward_kernel<T><<<blocks, threads>>>(
        grad_output.data(), used_input->data(),
        save_mean_.data(), save_var_.data(),
        affine_ ? gamma_.data() : nullptr,
        grad_input.data(),
        affine_ ? grad_gamma_.data() : nullptr,
        affine_ ? grad_beta_.data() : nullptr,
        N, C, H, W, epsilon_, affine_
    );
    dnn::utils::THROW_CUDA_EX();
    return grad_input;
#else
    // TODO: cuDNN path
    return grad_output.clone();
#endif
}

template<typename T>
std::vector<tensor<T>*> BatchNormLayer<T>::parameters() {
    // Return learnable parameters if affine
    if (affine_) return { &gamma_, &beta_ };
    return {};
}

template<typename T>
std::vector<tensor<T>*> BatchNormLayer<T>::gradients() {
    // Return gradients if affine
    if (affine_) return { &grad_gamma_, &grad_beta_ };
    return {};
}

template<typename T>
void BatchNormLayer<T>::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&channels_), sizeof(channels_));
    out.write(reinterpret_cast<const char*>(&epsilon_), sizeof(epsilon_));
    out.write(reinterpret_cast<const char*>(&momentum_), sizeof(momentum_));
    out.write(reinterpret_cast<const char*>(&affine_), sizeof(affine_));
    gamma_.save(out);
    beta_.save(out);
    running_mean_.save(out);
    running_var_.save(out);
}

template<typename T>
void BatchNormLayer<T>::load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&channels_), sizeof(channels_));
    in.read(reinterpret_cast<char*>(&epsilon_), sizeof(epsilon_));
    in.read(reinterpret_cast<char*>(&momentum_), sizeof(momentum_));
    in.read(reinterpret_cast<char*>(&affine_), sizeof(affine_));
    gamma_.load(in);
    beta_.load(in);
    running_mean_.load(in);
    running_var_.load(in);
}

// Explicit instantiation
template class BatchNormLayer<float>;
template class BatchNormLayer<__half>;

}  // namespace dnn

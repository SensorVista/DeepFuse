#include "dnn/layers/batch_norm_layer.cuh"
#include "dnn/utils/common.cuh"

namespace dnn {

template<typename T>
BatchNormLayer<T>::BatchNormLayer(int num_channels, float epsilon, float momentum, bool affine)
    : channels_(num_channels),
      epsilon_(epsilon),
      momentum_(momentum),
      affine_(affine),
      running_mean_({num_channels}),
      running_var_({num_channels}),
      gamma_({num_channels}),
      beta_({num_channels}),
      grad_gamma_({num_channels}),
      grad_beta_({num_channels}),
      is_training_(true)
#ifdef ENABLE_CUDNN
    , input_desc_(nullptr)
    , bn_desc_(nullptr)
#endif
{
    if (affine_) {
        gamma_.fill(static_cast<T>(1));  // identity scale
        beta_.fill(static_cast<T>(0));   // zero shift
    }

#ifdef ENABLE_CUDNN
    utils::CHECK_CUDNN_EX(cudnnCreateTensorDescriptor(&input_desc_));
    utils::CHECK_CUDNN_EX(cudnnCreateTensorDescriptor(&bn_desc_));
#endif
}

template<typename T>
BatchNormLayer<T>::~BatchNormLayer() {
#ifdef ENABLE_CUDNN
    if (input_desc_) cudnnDestroyTensorDescriptor(input_desc_);
    if (bn_desc_) cudnnDestroyTensorDescriptor(bn_desc_);
#endif
}

template<typename T>
tensor<T> BatchNormLayer<T>::forward(const tensor<T>& input) {
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    tensor<T> output(input.shape());

#ifdef ENABLE_CUDNN
    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(
        input_desc_, CUDNN_TENSOR_NCHW, cudnn_dtype<T>(), B, C, H, W));
    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(
        bn_desc_, CUDNN_TENSOR_NCHW, cudnn_dtype<T>(), 1, C, 1, 1));

    const float alpha = 1.0f, beta = 0.0f;

    if (is_training_) {
        utils::CHECK_CUDNN_EX(cudnnBatchNormalizationForwardTraining(
            utils::cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            input_desc_,
            input.data(),
            input_desc_,
            output.data(),
            bn_desc_,
            gamma_.data(),
            beta_.data(),
            momentum_,
            running_mean_.data(),
            running_var_.data(),
            epsilon_,
            nullptr, nullptr));
    } else {
        utils::CHECK_CUDNN_EX(cudnnBatchNormalizationForwardInference(
            utils::cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            input_desc_,
            input.data(),
            input_desc_,
            output.data(),
            bn_desc_,
            gamma_.data(),
            beta_.data(),
            running_mean_.data(),
            running_var_.data(),
            epsilon_));
    }
#else
    throw std::runtime_error("BatchNormLayer requires cuDNN in DeepFuse v1.");
#endif

    return output;
}

template<typename T>
tensor<T> BatchNormLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    tensor<T> grad_input(input.shape());

#ifdef ENABLE_CUDNN
    const float alpha_data = 1.0f, beta_data = 0.0f;
    const float alpha_param = 1.0f, beta_param = 0.0f;

    utils::CHECK_CUDNN_EX(cudnnBatchNormalizationBackward(
        utils::cudnn_handle(),
        CUDNN_BATCHNORM_SPATIAL,
        &alpha_data, &beta_data,
        &alpha_param, &beta_param,
        input_desc_,
        input.data(),
        input_desc_,
        grad_output.data(),
        input_desc_,
        grad_input.data(),
        bn_desc_,
        gamma_.data(),
        grad_gamma_.data(),
        grad_beta_.data(),
        epsilon_,
        nullptr, nullptr));
#else
    throw std::runtime_error("BatchNormLayer::backward requires cuDNN in DeepFuse v1.");
#endif

    return grad_input;
}

template<typename T>
std::vector<tensor<T>*> BatchNormLayer<T>::parameters() {
    if (!affine_) return {};
    return { &gamma_, &beta_ };
}

template<typename T>
std::vector<tensor<T>*> BatchNormLayer<T>::gradients() {
    if (!affine_) return {};
    return { &grad_gamma_, &grad_beta_ };
}

// -------------------------
// Explicit instantiation
// -------------------------

template class BatchNormLayer<float>;
template class BatchNormLayer<__half>;

}  // namespace dnn

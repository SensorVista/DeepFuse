#include "dnn/layers/residual_add_layer.cuh"
#include "dnn/utils/common.cuh"

namespace dnn {

using namespace dnn::utils;

template<typename T>
ResidualAddLayer<T>::ResidualAddLayer()
    : residual_(nullptr)
#ifdef ENABLE_CUDNN
    , desc_(nullptr)
#endif
{
#ifdef ENABLE_CUDNN
    utils::CHECK_CUDNN_EX(cudnnCreateTensorDescriptor(&desc_));
#endif
}

template<typename T>
ResidualAddLayer<T>::~ResidualAddLayer() {
#ifdef ENABLE_CUDNN
    if (desc_) cudnnDestroyTensorDescriptor(desc_);
#endif
}

template<typename T>
void ResidualAddLayer<T>::set_residual(tensor<T>* residual) {
    residual_ = residual;
}

template<typename T>
tensor<T> ResidualAddLayer<T>::forward(const tensor<T>& input) {
    if (!residual_) {
        throw std::runtime_error("ResidualAddLayer: residual_ is not set before forward().");
    }

    if (residual_->shape() != input.shape()) {
        throw std::runtime_error("ResidualAddLayer: input and residual shapes must match.");
    }

    tensor<T> output(input.shape());

#ifdef ENABLE_CUDNN
    const int N = input.shape(0);
    const int C = input.shape(1);
    const int H = input.shape(2);
    const int W = input.shape(3);

    utils::CHECK_CUDNN_EX(cudnnSetTensor4dDescriptor(
        desc_, CUDNN_TENSOR_NCHW, cudnn_dtype<T>(), N, C, H, W));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Copy input to output
    utils::CHECK_CUDA_EX(cudaMemcpy(output.data(), input.data(), input.size() * sizeof(T), cudaMemcpyDeviceToDevice));

    // Add residual to output
    const float beta_add = 1.0f;
    utils::CHECK_CUDNN_EX(cudnnAddTensor(
        utils::cudnn_handle(), &alpha, desc_, residual_->data(), &beta_add, desc_, output.data()));
#else
    throw std::runtime_error("ResidualAddLayer requires ENABLE_CUDNN in DeepFuse.");
#endif

    return output;
}

template<typename T>
tensor<T> ResidualAddLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    // Create gradient tensor with proper shape
    tensor<T> grad(grad_output.shape());
    grad.copy_from(grad_output);
    return grad;
}

// Explicit template instantiations
template class ResidualAddLayer<float>;
template class ResidualAddLayer<__half>;

} // namespace dnn

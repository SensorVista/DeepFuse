#pragma once

#include "dnn/core/tensor.cuh"
#include "dnn/core/layer.cuh"

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

namespace dnn {

enum class ActivationType {
#ifdef ENABLE_CUDNN    
    ReLU = CUDNN_ACTIVATION_RELU,
    Sigmoid = CUDNN_ACTIVATION_SIGMOID,
    Tanh = CUDNN_ACTIVATION_TANH,
    ClippedReLU = CUDNN_ACTIVATION_CLIPPED_RELU,
    Elu = CUDNN_ACTIVATION_ELU
#else
    ReLU,
    Sigmoid,
    Tanh,
    ClippedReLU,
    Elu
#endif
};

template<typename T>
class ActivationLayer : public Layer<T> {
public:
    explicit ActivationLayer(ActivationType type);
    ~ActivationLayer();

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;
    std::string name() const override;

private:
    ActivationType type_;
#ifdef ENABLE_CUDNN
    cudnnActivationDescriptor_t act_desc_;
#endif
};

// Explicit template instantiations
template class ActivationLayer<float>;

} // namespace dnn 
#pragma once

#include "dnn/core/tensor.cuh"
#include "dnn/core/layer.cuh"
#include <optional>

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
    Elu = CUDNN_ACTIVATION_ELU,
#else
    ReLU,
    Sigmoid,
    Tanh,
    ClippedReLU,
    Elu,
#endif
    GELU,
};

template<typename T>
class ActivationLayer : public Layer<T> {
public:
    ActivationLayer(ActivationType type, bool training_enabled = false);
    ~ActivationLayer();

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output) override;
    std::string name() const override;

    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

private:
    ActivationType type_;
#ifdef ENABLE_CUDNN
    cudnnActivationDescriptor_t act_desc_;
#endif
    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<T>> input_cache_;
};

} // namespace dnn 
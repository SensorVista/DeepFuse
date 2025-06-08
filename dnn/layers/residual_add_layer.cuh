#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

namespace dnn {

template<typename T>
class ResidualAddLayer : public Layer<T> {
public:
    ResidualAddLayer();
    ~ResidualAddLayer();

    // Set external residual input (non-owning)
    void set_residual(tensor<T>* residual);

    // Standard forward/backward API
    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    std::string name() const override { return "ResidualAdd"; }

private:
    tensor<T>* residual_;

#ifdef ENABLE_CUDNN
    cudnnTensorDescriptor_t desc_;
#endif
};

}  // namespace dnn

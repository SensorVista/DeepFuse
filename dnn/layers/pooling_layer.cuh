#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

namespace dnn {

enum class PoolingType {
    Max,
    Average,
    MaxDeterministic
};

template<typename T>
class PoolingLayer : public Layer<T> {
public:
    PoolingLayer(PoolingType type, int kernel_size, int stride);
    ~PoolingLayer();

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    std::vector<tensor<T>*> parameters() override { return {}; }
    std::vector<tensor<T>*> gradients() override { return {}; }

    std::string name() const override { return "Pooling"; }

private:
    PoolingType type_;
    int kernel_size_;
    int stride_;

#ifdef ENABLE_CUDNN
    cudnnPoolingDescriptor_t pool_desc_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
#endif
};

} // namespace dnn 
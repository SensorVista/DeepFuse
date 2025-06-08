#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

namespace dnn {

template<typename T>
class BatchNormLayer : public Layer<T> {
public:
    BatchNormLayer(int num_channels, float epsilon = 1e-5f, float momentum = 0.9f, bool affine = true);
    ~BatchNormLayer();

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    std::vector<tensor<T>*> parameters() override;
    std::vector<tensor<T>*> gradients() override;

    std::string name() const override { return "BatchNorm"; }

private:
    int channels_;
    float epsilon_;
    float momentum_;
    bool affine_;  // Whether to apply learnable scale/shift (gamma/beta)

    tensor<T> running_mean_;
    tensor<T> running_var_;

    tensor<T> gamma_;   // scale (only if affine_)
    tensor<T> beta_;    // shift (only if affine_)

    tensor<T> grad_gamma_;
    tensor<T> grad_beta_;

#ifdef ENABLE_CUDNN
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t bn_desc_;
#endif

    bool is_training_;  // Can be toggled externally in future
};

}  // namespace dnn

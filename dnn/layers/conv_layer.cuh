#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

#include <iostream>
#include <random>

namespace dnn {

template<typename T>
class ConvLayer : public LayerWeightBias<T> {
public:
    ConvLayer(int in_channels, int out_channels, const std::vector<int>& kernel_size, int stride, int padding, const std::vector<std::vector<bool>>& connection_table = {});
    ~ConvLayer();

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    std::string name() const override { 
        return "Conv"; 
    }

    void initialize_weights() override;

private:
    int in_channels_;
    int out_channels_;
    std::vector<int> kernel_size_;
    int stride_;
    int padding_;
    tensor<uint8_t> connection_mask_dev_; // Flattened mask for device
    bool use_sparse_connectivity_;

#ifdef ENABLE_CUDNN
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
#endif

};

} // namespace dnn 
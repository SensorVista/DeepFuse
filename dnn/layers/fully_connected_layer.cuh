#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

#include <vector>
#include <random>

namespace dnn {

template<typename T>
class FullyConnectedLayer : public LayerWeightBias<T> {
public:
	FullyConnectedLayer(int in_features, int out_features);
	~FullyConnectedLayer();
	
	tensor<T> forward(const tensor<T>& input) override;
	tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

	std::string name() const override {
		return "FullyConnected";
	}

	void initialize_weights() override;

private:
	int in_features_;
	int out_features_;
#ifdef ENABLE_CUDNN
	cudnnFilterDescriptor_t filter_desc_;
	cudnnConvolutionDescriptor_t conv_desc_;
	cudnnTensorDescriptor_t input_desc_;
	cudnnTensorDescriptor_t output_desc_;
#endif

};

} // namespace dnn 

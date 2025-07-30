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
#include <optional>
#include <iostream>

namespace dnn {

template<typename TT>
class FullyConnectedLayer : public LayerWeightBias<TT> {
public:
	FullyConnectedLayer(int in_features, int out_features, bool training_enabled = false);
	~FullyConnectedLayer() override;
	
	tensor<TT> forward(const tensor<TT>& input) override;
	tensor<TT> backward(const tensor<TT>& grad_output) override;

	std::string name() const override {
		return "FullyConnected";
	}

	void initialize_weights() override;

    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

private:
	int in_features_;
	int out_features_;
#ifdef ENABLE_CUDNN
	cudnnFilterDescriptor_t filter_desc_;
	cudnnConvolutionDescriptor_t conv_desc_;
	cudnnTensorDescriptor_t input_desc_;
	cudnnTensorDescriptor_t output_desc_;
#endif

    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<TT>> input_cache_;

};

} // namespace dnn 

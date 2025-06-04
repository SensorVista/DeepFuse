#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

#include <vector>
#include <random>

namespace dnn {

template<typename T>
class FullyConnectedLayer : public Layer<T> {
private:
	tensor<T> weights_;
	tensor<T> bias_;
	tensor<T> grad_weights_;
	tensor<T> grad_bias_;
	int in_features_;
	int out_features_;

public:
	FullyConnectedLayer(int in_features, int out_features)
		: in_features_(in_features)
		, out_features_(out_features)
		, weights_({ out_features, in_features })
		, bias_({ out_features })
		, grad_weights_({ out_features, in_features })
		, grad_bias_({ out_features }) {
		initialize_weights();
	}

	tensor<T> forward(const tensor<T>& input) override;

	tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

	std::vector<tensor<T>*> parameters() override {
		return { &weights_, &bias_ };
	}

	std::vector<tensor<T>*> gradients() override {
		return { &grad_weights_, &grad_bias_ };
	}

	const char* name() const override {
		return "FullyConnected";
	}

private:
	void initialize_weights() {
		std::random_device rd;
        std::mt19937 gen(rd());
		// Xavier/Glorot initialization
		float limit = std::sqrt(6.0f / (in_features_ + out_features_)); // Adjusted for Glorot uniform
        std::uniform_real_distribution<T> dist(-limit, limit);

		std::vector<T> host_weights(weights_.size());
		std::vector<T> host_bias(bias_.size());
		for (int i = 0; i < host_weights.size(); ++i) {
			host_weights[i] = dist(gen);
		}
		std::fill(host_bias.begin(), host_bias.end(), static_cast<T>(0.0f));
		weights_.upload(host_weights.data());
		bias_.upload(host_bias.data());
	}
};

// // Concrete aliases for commonly used FullyConnectedLayer types
 //using FullyConnectedLayerf   = FullyConnectedLayer<float>;    // 32-bit float FullyConnectedLayer
 //using FullyConnectedLayerh   = FullyConnectedLayer<__half>;   // 16-bit half FullyConnectedLayer
 //using FullyConnectedLayeri8  = FullyConnectedLayer<int8_t>;   // 8-bit integer FullyConnectedLayer (for quantized models)

} // namespace dnn 

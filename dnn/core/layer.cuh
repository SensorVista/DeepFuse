#pragma once

#include "tensor.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>
#include <memory>

namespace lenet5 {

template<typename T>
class Layer {
public:
    virtual ~Layer() = default;

    // Forward pass
    virtual tensor<T> forward(const tensor<T>& input) = 0;

    // Backward pass
    virtual tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) = 0;

    // Get layer parameters
    virtual std::vector<tensor<T>*> parameters() { return {}; };

    // Get layer gradients
    virtual std::vector<tensor<T>*> gradients() { return {}; };

    // Get layer name
    virtual const char* name() const = 0;

protected:
    // Helper function to check input shape
    bool check_input_shape(const tensor<T>& input, const std::vector<size_t>& expected_shape) const {
        return input.shape() == expected_shape;
    }
};

// Forward declarations for supported template types
template class Layer<float>;  // FP32
// template class Layer<__half>; // FP16
// template class Layer<__nv_bfloat16>; // BF16

} // namespace lenet5 
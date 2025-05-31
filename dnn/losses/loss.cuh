#pragma once

#include "dnn/core/tensor.cuh"

namespace lenet5 {

template<typename T>
class Loss {
public:
    virtual ~Loss() = default;
    
    // Compute loss value
    virtual T compute(const tensor<T>& predictions, const tensor<T>& targets) = 0;
    
    // Compute loss gradient
    virtual tensor<T> compute_gradient(const tensor<T>& predictions, const tensor<T>& targets) = 0;
};

} // namespace lenet5 
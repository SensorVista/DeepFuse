#pragma once

#include "dnn/utils/common.cuh"
#include "dnn/losses/loss.cuh"

namespace dnn {

template<typename T>
class CrossEntropyLoss : public Loss<T> {
public:
    CrossEntropyLoss() = default;
    
    T compute(const tensor<T>& predictions, const tensor<T>& targets) override;
    tensor<T> compute_gradient(const tensor<T>& predictions, const tensor<T>& targets) override;

    // Overloads for class index targets
    T compute(const tensor<T>& predictions, const tensor<int>& target_indices);
    tensor<T> compute_gradient(const tensor<T>& predictions, const tensor<int>& target_indices);
};

} // namespace dnn 
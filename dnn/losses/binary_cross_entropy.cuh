#pragma once

#include "dnn/losses/loss.cuh"

namespace dnn {

template<typename T>
class BinaryCrossEntropyLoss : public Loss<T> {
public:
    BinaryCrossEntropyLoss() = default;
    
    T compute(const tensor<T>& predictions, const tensor<T>& targets) override;
    tensor<T> compute_gradient(const tensor<T>& predictions, const tensor<T>& targets) override;
};

} // namespace dnn 
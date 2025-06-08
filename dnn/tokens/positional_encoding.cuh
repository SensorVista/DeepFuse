#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#include <cmath>
#include <cuda_runtime.h>

namespace dnn {

template <typename T>
class PositionalEncoding : public Layer<T> {
public:
    PositionalEncoding(int embed_dim, int max_seq_len);
    ~PositionalEncoding() override = default;

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output, const tensor<T>& input) override;

    std::string name() const override { return "PositionalEncoding"; }

private:
    void initialize_positional_encodings();

    int embed_dim_;
    int max_seq_len_;
    tensor<T> pos_encoding_;
};

}  // namespace dnn 
#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <optional>

namespace dnn {

template <typename T>
class PositionalEncoding : public Layer<T> {
public:
    PositionalEncoding(int embed_dim, int max_seq_len, bool training_enabled = false);
    ~PositionalEncoding() override = default;

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output) override;

    std::string name() const override { return "PositionalEncoding"; }

    void save(std::ostream& out) const override {}
    void load(std::istream& in) override {}

private:
    void initialize_positional_encodings();

    int embed_dim_;
    int max_seq_len_;
    tensor<T> pos_encoding_;
    
    std::optional<tensor<T>> input_cache_;
};

}  // namespace dnn 
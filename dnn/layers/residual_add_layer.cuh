#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"
#include <optional>

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

namespace dnn {

template<typename T>
class ResidualAddLayer : public Layer<T> {
public:
    ResidualAddLayer(bool training_enabled = false);
    ~ResidualAddLayer();

    // Set external residual input (non-owning)
    void set_residual(tensor<T>* residual);

    // Standard forward/backward API
    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output) override;

    std::string name() const override { return "ResidualAdd"; }

    void save(std::ostream& out) const override {}
    void load(std::istream& in) override {}

private:
    tensor<T>* residual_;

    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<T>> input_cache_;

#ifdef ENABLE_CUDNN
    cudnnTensorDescriptor_t desc_;
#endif
};

}  // namespace dnn

#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"
#include <optional>

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

namespace dnn {

template<typename T>
class BatchNormLayer : public Layer<T> {
public:
    BatchNormLayer(int num_channels, float epsilon = 1e-5f, float momentum = 0.9f, bool affine = true, bool training_enabled = false);
    ~BatchNormLayer();

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output) override;

    std::vector<tensor<T>*> parameters() override;
    std::vector<tensor<T>*> gradients() override;

    std::string name() const override { return "BatchNorm"; }

    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

private:
    int channels_;
    float epsilon_;
    float momentum_;
    bool affine_;  // Whether to apply learnable scale/shift (gamma/beta)

    // Learnable parameters (if affine)
    tensor<T> gamma_;
    tensor<T> beta_;
    tensor<T> grad_gamma_;
    tensor<T> grad_beta_;

    // Running statistics (for inference)
    tensor<T> running_mean_;
    tensor<T> running_var_;

    // Saved for backward
    tensor<T> save_mean_;
    tensor<T> save_var_;

    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<T>> input_cache_;
};

}  // namespace dnn

#pragma once

#include "dnn/core/layer.cuh"
#include "dnn/core/tensor.cuh"
#include <optional>

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

namespace dnn {

enum class PoolingType {
    Max,
    Average,
    MaxDeterministic
};

template<typename T>
class PoolingLayer : public Layer<T> {
public:
    PoolingLayer(PoolingType type, int kernel_size, int stride, bool training_enabled = false);
    ~PoolingLayer();

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output) override;

    std::vector<tensor<T>*> parameters() override { return {}; }
    std::vector<tensor<T>*> gradients() override { return {}; }

    std::string name() const override { return "Pooling"; }

    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

private:
    PoolingType type_;
    int kernel_size_;
    int stride_;

    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<T>> input_cache_;

#ifdef ENABLE_CUDNN
    cudnnPoolingDescriptor_t pool_desc_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
#endif
};

template<typename T>
class GlobalPoolingLayer : public Layer<T> {
public:
    explicit GlobalPoolingLayer(PoolingType type, bool training_enabled = false);
    ~GlobalPoolingLayer();

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> backward(const tensor<T>& grad_output) override;

    std::vector<tensor<T>*> parameters() override { return {}; }
    std::vector<tensor<T>*> gradients() override { return {}; }

    std::string name() const override { return "GlobalPooling"; }

    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

private:
    PoolingType type_;

    // Caches input tensor during forward if training_enabled_ is true
    std::optional<tensor<T>> input_cache_;

#ifdef ENABLE_CUDNN
    cudnnPoolingDescriptor_t pool_desc_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
#endif
};

} // namespace dnn 
#include "dnn/layers/residual_block.cuh"

namespace dnn {

template<typename T>
ResidualBlock<T>::ResidualBlock(int in_channels, int out_channels, int stride, bool bottleneck)
    : bottleneck_(bottleneck), in_channels_(in_channels), out_channels_(out_channels), stride_(stride)
{
    use_projection_ = bottleneck || stride != 1 || in_channels != out_channels;

    if (bottleneck_) {
        int mid_channels = out_channels / 4;

        main_path_.emplace_back(std::make_unique<ConvLayer<T>>(in_channels, mid_channels, std::vector<int>{1, 1}, stride, 0));
        main_path_.emplace_back(std::make_unique<BatchNormLayer<T>>(mid_channels));
        main_path_.emplace_back(std::make_unique<ActivationLayer<T>>(ActivationType::ReLU));

        main_path_.emplace_back(std::make_unique<ConvLayer<T>>(mid_channels, mid_channels, std::vector<int>{3, 3}, 1, 1));
        main_path_.emplace_back(std::make_unique<BatchNormLayer<T>>(mid_channels));
        main_path_.emplace_back(std::make_unique<ActivationLayer<T>>(ActivationType::ReLU));

        main_path_.emplace_back(std::make_unique<ConvLayer<T>>(mid_channels, out_channels, std::vector<int>{1, 1}, 1, 0));
        main_path_.emplace_back(std::make_unique<BatchNormLayer<T>>(out_channels));
    } else {
        main_path_.emplace_back(std::make_unique<ConvLayer<T>>(in_channels, out_channels, std::vector<int>{3, 3}, stride, 1));
        main_path_.emplace_back(std::make_unique<BatchNormLayer<T>>(out_channels));
        main_path_.emplace_back(std::make_unique<ActivationLayer<T>>(ActivationType::ReLU));

        main_path_.emplace_back(std::make_unique<ConvLayer<T>>(out_channels, out_channels, std::vector<int>{3, 3}, 1, 1));
        main_path_.emplace_back(std::make_unique<BatchNormLayer<T>>(out_channels));
    }

    if (use_projection_) {
        projection_ = std::make_unique<ConvLayer<T>>(in_channels, out_channels, std::vector<int>{1, 1}, stride, 0);
    }
}

template<typename T>
tensor<T> ResidualBlock<T>::forward(const tensor<T>& input) {
    // Forward through main path
    tensor<T> x(input.shape());
    x.copy_from(input);
    
    for (auto& layer : main_path_) {
        x = layer->forward(x);
    }

    // Handle residual path
    tensor<T> residual(input.shape());  // Initialize with input shape
    if (use_projection_) {
        residual = projection_->forward(input);
    } else {
        residual.copy_from(input);
    }

    // Add residual
    add_.set_residual(&residual);
    return add_.forward(x);
}

template<typename T>
tensor<T> ResidualBlock<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    // Backward through residual add
    tensor<T> grad_main = add_.backward(grad_output, input);

    // Backward through main path
    tensor<T> grad(grad_main.shape());
    grad.copy_from(grad_main);

    for (int i = static_cast<int>(main_path_.size()) - 1; i >= 0; --i) {
        tensor<T> prev_output(input.shape());  // Initialize with input shape
        if (i == 0) {
            prev_output.copy_from(input);
        } else {
            prev_output = main_path_[i - 1]->forward(input);
        }
        grad = main_path_[i]->backward(grad, prev_output);
    }

    // Backward through skip path if projection was used
    if (use_projection_) {
        projection_->backward(grad_output, input);
    }

    return grad;
}

template<typename T>
std::vector<tensor<T>*> ResidualBlock<T>::parameters() {
    std::vector<tensor<T>*> params;
    for (auto& layer : main_path_) {
        auto p = layer->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    if (use_projection_ && projection_) {
        auto p = projection_->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

template<typename T>
std::vector<tensor<T>*> ResidualBlock<T>::gradients() {
    std::vector<tensor<T>*> grads;
    for (auto& layer : main_path_) {
        auto g = layer->gradients();
        grads.insert(grads.end(), g.begin(), g.end());
    }
    if (use_projection_ && projection_) {
        auto g = projection_->gradients();
        grads.insert(grads.end(), g.begin(), g.end());
    }
    return grads;
}

// Explicit instantiation
template class ResidualBlock<float>;
template class ResidualBlock<__half>;

} // namespace dnn

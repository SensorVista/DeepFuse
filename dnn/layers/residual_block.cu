#include "dnn/layers/residual_block.cuh"

namespace dnn {

template<typename T>
ResidualBlock<T>::ResidualBlock(int in_channels, int out_channels, int stride, bool bottleneck, bool training_enabled)
    : Layer<T>(training_enabled), bottleneck_(bottleneck), in_channels_(in_channels), out_channels_(out_channels), stride_(stride)
{
    use_projection_ = bottleneck || stride != 1 || in_channels != out_channels;

    if (bottleneck_) {
        int mid_channels = out_channels / 4;

        main_path_.emplace_back(std::make_unique<ConvLayer<T>>(in_channels, mid_channels, std::vector<int>{1, 1}, stride, 0, std::vector<std::vector<bool>>{}, training_enabled));
        main_path_.emplace_back(std::make_unique<BatchNormLayer<T>>(mid_channels, 1e-5f, 0.9f, true, training_enabled));
        main_path_.emplace_back(std::make_unique<ActivationLayer<T>>(ActivationType::ReLU, training_enabled));

        main_path_.emplace_back(std::make_unique<ConvLayer<T>>(mid_channels, mid_channels, std::vector<int>{3, 3}, 1, 1, std::vector<std::vector<bool>>{}, training_enabled));
        main_path_.emplace_back(std::make_unique<BatchNormLayer<T>>(mid_channels, 1e-5f, 0.9f, true, training_enabled));
        main_path_.emplace_back(std::make_unique<ActivationLayer<T>>(ActivationType::ReLU, training_enabled));

        main_path_.emplace_back(std::make_unique<ConvLayer<T>>(mid_channels, out_channels, std::vector<int>{1, 1}, 1, 0, std::vector<std::vector<bool>>{}, training_enabled));
        main_path_.emplace_back(std::make_unique<BatchNormLayer<T>>(out_channels, 1e-5f, 0.9f, true, training_enabled));
    } else {
        main_path_.emplace_back(std::make_unique<ConvLayer<T>>(in_channels, out_channels, std::vector<int>{3, 3}, stride, 1, std::vector<std::vector<bool>>{}, training_enabled));
        main_path_.emplace_back(std::make_unique<BatchNormLayer<T>>(out_channels, 1e-5f, 0.9f, true, training_enabled));
        main_path_.emplace_back(std::make_unique<ActivationLayer<T>>(ActivationType::ReLU, training_enabled));

        main_path_.emplace_back(std::make_unique<ConvLayer<T>>(out_channels, out_channels, std::vector<int>{3, 3}, 1, 1, std::vector<std::vector<bool>>{}, training_enabled));
        main_path_.emplace_back(std::make_unique<BatchNormLayer<T>>(out_channels, 1e-5f, 0.9f, true, training_enabled));
    }

    if (use_projection_) {
        projection_ = std::make_unique<ConvLayer<T>>(in_channels, out_channels, std::vector<int>{1, 1}, stride, 0, std::vector<std::vector<bool>>{}, training_enabled);
    }
}

template<typename T>
tensor<T> ResidualBlock<T>::forward(const tensor<T>& input) {
    if (this->training_enabled_) {
        input_cache_ = input.clone();
    }
    tensor<T> x = input.clone();
    
    for (auto& layer : main_path_) {
        Layer<T>* typed_layer = dynamic_cast<Layer<T>*>(layer.get());
        if (!typed_layer) {
            throw std::runtime_error("Invalid layer type in ResidualBlock");
        }
        x = typed_layer->forward(x);
    }

    // Handle residual path
    tensor<T> residual(input.shape());  // Initialize with input shape
    if (use_projection_) {
        residual = projection_->forward(input);
    } else {
        residual = input.clone();
    }

    // Add residual
    add_.set_residual(&residual);
    return add_.forward(x);
}

template<typename T>
tensor<T> ResidualBlock<T>::backward(const tensor<T>& grad_output) {
    if (this->training_enabled_) {
        if (!input_cache_.has_value()) {
            throw std::runtime_error("ResidualBlock: input_cache_ is empty in backward().");
        }
    }
    const tensor<T>& input = this->training_enabled_ ? input_cache_.value() : grad_output; // fallback for stateless
    tensor<T> grad_main = add_.backward(grad_output);

    // Backward through main path
    tensor<T> grad = grad_main.clone();

    for (int i = static_cast<int>(main_path_.size()) - 1; i >= 0; --i) {
        Layer<T>* typed_layer = dynamic_cast<Layer<T>*>(main_path_[i].get());
        if (!typed_layer) {
            throw std::runtime_error("Invalid layer type in ResidualBlock");
        }
        grad = typed_layer->backward(grad);
    }

    // Backward through skip path if projection was used
    if (use_projection_) {
        projection_->backward(grad_output);
    }

    return grad;
}

template<typename T>
std::vector<tensor<T>*> ResidualBlock<T>::parameters() {
    std::vector<tensor<T>*> params;
    for (auto& layer : main_path_) {
        Layer<T>* typed_layer = dynamic_cast<Layer<T>*>(layer.get());
        if (!typed_layer) {
            throw std::runtime_error("Invalid layer type in ResidualBlock");
        }
        auto p = typed_layer->parameters();
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
        Layer<T>* typed_layer = dynamic_cast<Layer<T>*>(layer.get());
        if (!typed_layer) {
            throw std::runtime_error("Invalid layer type in ResidualBlock");
        }
        auto g = typed_layer->gradients();
        grads.insert(grads.end(), g.begin(), g.end());
    }
    if (use_projection_ && projection_) {
        auto g = projection_->gradients();
        grads.insert(grads.end(), g.begin(), g.end());
    }
    return grads;
}

template<typename T>
void ResidualBlock<T>::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&bottleneck_), sizeof(bottleneck_));
    out.write(reinterpret_cast<const char*>(&use_projection_), sizeof(use_projection_));
    out.write(reinterpret_cast<const char*>(&in_channels_), sizeof(in_channels_));
    out.write(reinterpret_cast<const char*>(&out_channels_), sizeof(out_channels_));
    out.write(reinterpret_cast<const char*>(&stride_), sizeof(stride_));
    int num_layers = static_cast<int>(main_path_.size());
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    for (const auto& layer : main_path_) {
        layer->save(out);
    }
    bool has_proj = (projection_ != nullptr);
    out.write(reinterpret_cast<const char*>(&has_proj), sizeof(has_proj));
    if (has_proj) {
        projection_->save(out);
    }
}

template<typename T>
void ResidualBlock<T>::load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&bottleneck_), sizeof(bottleneck_));
    in.read(reinterpret_cast<char*>(&use_projection_), sizeof(use_projection_));
    in.read(reinterpret_cast<char*>(&in_channels_), sizeof(in_channels_));
    in.read(reinterpret_cast<char*>(&out_channels_), sizeof(out_channels_));
    in.read(reinterpret_cast<char*>(&stride_), sizeof(stride_));
    int num_layers;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    for (int i = 0; i < num_layers; ++i) {
        main_path_[i]->load(in);
    }
    bool has_proj;
    in.read(reinterpret_cast<char*>(&has_proj), sizeof(has_proj));
    if (has_proj && projection_) {
        projection_->load(in);
    }
}

// Explicit instantiation
template class ResidualBlock<float>;
template class ResidualBlock<__half>;

} // namespace dnn

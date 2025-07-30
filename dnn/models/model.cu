#include "model.cuh"

#include <fstream>
#include <ctime>

namespace dnn {

template<typename T>
tensor<T> Model<T>::forward(const tensor<T>& input) {
    tensor<T> current = input.clone();
    for (auto* layer : layers()) {
        // Cast BaseLayer to Layer<T> for forward pass
        auto typed_layer = dynamic_cast<Layer<T>*>(layer);
        if (!typed_layer) {
            throw std::runtime_error("Layer type mismatch in forward pass");
        }
        current = typed_layer->forward(current);
    }
    return current;
}

template<typename T>
std::vector<tensor<T>*> Model<T>::parameters() {
    std::vector<tensor<T>*> params;
    for (auto* layer : layers()) {
        // Cast BaseLayer to Layer<T> for parameter access
        auto typed_layer = dynamic_cast<Layer<T>*>(layer);
        if (!typed_layer) {
            throw std::runtime_error("Layer type mismatch in parameter access");
        }
        auto layer_params = typed_layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

// Explicit template instantiations
template class Model<float>;
template class Model<__half>;
// template class Model<__nv_bfloat16>;

} // namespace dnn

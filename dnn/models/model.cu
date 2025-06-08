#include "model.cuh"
#include "../utils/onnx_loader.cuh"

#include <fstream>
#include <ctime>

namespace dnn {

template<typename T>
tensor<T> Model<T>::forward(const tensor<T>& input) {
    tensor<T> current(input.shape());
    current.copy_from(input);
    for (const auto& layer : layers_) {
        current = layer->forward(current);  // move assignment
    }
    return current;
}

template<typename T>
std::vector<tensor<T>*> Model<T>::parameters() {
    std::vector<tensor<T>*> params;
    for (const auto& layer : layers_) {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

template<typename T>
void Model<T>::save(const std::string& path, StorageType type) const {
    if (type == StorageType::ONNX) {
        dnn::OnnxLoader::save_onyx_v2<T>(this, path);
    }
    throw std::runtime_error("Unsupported storage type");
}

template<typename T>
/* static */ Model<T>* Model<T>::load(const std::string& path, StorageType type) {
    if (type == StorageType::ONNX) {
        return OnnxLoader::load_onyx_v2<T>(path);
    }
    throw std::runtime_error("Unsupported storage type");
}

// Explicit template instantiations
template class Model<float>;
// template class Model<__half>;
// template class Model<__nv_bfloat16>;

} // namespace dnn

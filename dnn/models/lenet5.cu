#include "lenet5.cuh"
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

namespace dnn {

template<typename T>
void LeNet5<T>::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file for saving: " + path);
    // Save model metadata
    int num_layers = 12; // Number of explicit layers
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    // Save each layer explicitly
    conv1_.save(out);
    act1_.save(out);
    pool1_.save(out);
    conv2_.save(out);
    act2_.save(out);
    pool2_.save(out);
    conv3_.save(out);
    act3_.save(out);
    flatten_.save(out);
    fc1_.save(out);
    act4_.save(out);
    fc2_.save(out);
    // Save optimizer and loss if needed (not implemented here)
}

template<typename T>
std::unique_ptr<LeNet5<T>> LeNet5<T>::load(const std::string& path, bool training_enabled) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file for loading: " + path);
    // Reconstruct model architecture
    auto model = std::make_unique<LeNet5<T>>(0.01, 0.9, training_enabled); // Use default args, or read from file if you serialize them
    int num_layers = 0;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    if (num_layers != 12)
        throw std::runtime_error("Layer count mismatch in LeNet5::load");
    model->conv1_.load(in);
    model->act1_.load(in);
    model->pool1_.load(in);
    model->conv2_.load(in);
    model->act2_.load(in);
    model->pool2_.load(in);
    model->conv3_.load(in);
    model->act3_.load(in);
    model->flatten_.load(in);
    model->fc1_.load(in);
    model->act4_.load(in);
    model->fc2_.load(in);
    // Load optimizer and loss if needed (not implemented here)
    return model;
}

// Explicit instantiation
template class LeNet5<float>;
template class LeNet5<__half>;

} // namespace dnn 
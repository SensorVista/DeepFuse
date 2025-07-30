#include "perceptron.cuh"
#include <fstream>
#include <memory>
#include <string>
#include <iostream>

namespace dnn {

template<typename T>
Perceptron<T>::Perceptron(int input_size, int hidden_size, int output_size, T learning_rate, T momentum, bool training_enabled)
    : TrainingModel<T>(training_enabled),
      fc1_(input_size, hidden_size, training_enabled),
      act1_(ActivationType::Tanh, training_enabled),
      fc2_(hidden_size, output_size, training_enabled),
      act2_(ActivationType::Sigmoid, training_enabled)
{
    this->set_loss(std::make_unique<dnn::BinaryCrossEntropyLoss<T>>());
    this->set_optimizer(std::make_unique<dnn::SGDOptimizer<T>>(learning_rate, momentum));
}

template<typename T>
std::vector<BaseLayer*> Perceptron<T>::layers() {
    return { &fc1_, &act1_, &fc2_, &act2_ };
}

template<typename T>
void Perceptron<T>::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file for saving: " + path);
    // Save model metadata
    int num_layers = 4; // Number of explicit layers
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    // Save each layer explicitly
    fc1_.save(out);
    act1_.save(out);
    fc2_.save(out);
    act2_.save(out);
    // Save optimizer and loss if needed (not implemented here)
}

template<typename T>
std::unique_ptr<Perceptron<T>> Perceptron<T>::load(const std::string& path, bool training_enabled) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file for loading: " + path);
    // Reconstruct model architecture
    // Use default args, or read from file if you serialize them
    // (Here, we use placeholder values; adjust as needed)
    auto model = std::make_unique<Perceptron<T>>(1, 1, 1, 0.01, 0.9, training_enabled);
    int num_layers = 0;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    if (num_layers != 4)
        throw std::runtime_error("Layer count mismatch in Perceptron::load");
    model->fc1_.load(in);
    model->act1_.load(in);
    model->fc2_.load(in);
    model->act2_.load(in);
    // Load optimizer and loss if needed (not implemented here)
    return model;
}

// Explicit instantiation
template class Perceptron<float>;
template class Perceptron<__half>;

} // namespace dnn

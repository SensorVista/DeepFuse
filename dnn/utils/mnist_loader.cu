#include "mnist_loader.cuh"

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

namespace dnn {

/* static */ std::vector<std::vector<float>> MNISTLoader::load_images(const std::string& filepath, int num_images, int image_size) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filepath);
    }

    // Skip the header
    file.ignore(16);

    std::vector<std::vector<float>> images(static_cast<size_t>(num_images), std::vector<float>(static_cast<size_t>(image_size)));
    for (int i = 0; i < num_images; ++i) {
        std::vector<uint8_t> buffer(static_cast<size_t>(image_size));
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(image_size));
        std::transform(buffer.begin(), buffer.end(), images[static_cast<size_t>(i)].begin(), [](uint8_t pixel) {
            return static_cast<float>(pixel) / 255.0f;
        });
    }

    return images;
}


/* static */ std::vector<uint8_t> MNISTLoader::load_labels(const std::string& filepath, int num_labels) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filepath);
    }

    // Skip the header
    file.ignore(8);

    std::vector<uint8_t> labels(static_cast<size_t>(num_labels));
    file.read(reinterpret_cast<char*>(labels.data()), static_cast<std::streamsize>(num_labels));

    return labels;
}

} // namespace dnn 
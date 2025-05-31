#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

namespace lenet5 {

class MNISTLoader {
public:
    static std::vector<std::vector<float>> load_images(const std::string& filepath, int num_images, int image_size) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file " + filepath);
        }

        // Skip the header
        file.ignore(16);

        std::vector<std::vector<float>> images(num_images, std::vector<float>(image_size));
        for (int i = 0; i < num_images; ++i) {
            std::vector<uint8_t> buffer(image_size);
            file.read(reinterpret_cast<char*>(buffer.data()), image_size);
            std::transform(buffer.begin(), buffer.end(), images[i].begin(), [](uint8_t pixel) {
                return static_cast<float>(pixel) / 255.0f;
            });
        }

        return images;
    }

    static std::vector<uint8_t> load_labels(const std::string& filepath, int num_labels) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file " + filepath);
        }

        // Skip the header
        file.ignore(8);

        std::vector<uint8_t> labels(num_labels);
        file.read(reinterpret_cast<char*>(labels.data()), num_labels);

        return labels;
    }
};

} // namespace lenet5 
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dnn {

class MNISTLoader {
public:
    static std::vector<std::vector<float>> load_images(const std::string& filepath, int num_images, int image_size);

    static std::vector<uint8_t> load_labels(const std::string& filepath, int num_labels);
};

} // namespace dnn 
#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <array>
#include <filesystem>

#include "../core/tensor.cuh" // For tensor<T>

namespace lenet5 {

class CIFAR10Loader {
public:
    static void load(bool train, std::vector<std::vector<float>>& images, std::vector<uint8_t>& labels, const std::string& dataset_path_override = "") {
        namespace fs = std::filesystem;

        std::string dataset_base_path = dataset_path_override.empty() ? "../datasets/CIFAR-10/" : dataset_path_override;
        std::vector<std::string> filenames;

        if (train) {
            filenames = {
                "data_batch_1.bin",
                "data_batch_2.bin",
                "data_batch_3.bin",
                "data_batch_4.bin",
                "data_batch_5.bin"
            };
        } else {
            filenames = {
                "test_batch.bin"
            };
        }

        const int IMAGE_SIZE_BYTES = 3073; // 1 label byte + 3072 image bytes (32*32*3)
        const int IMAGE_WIDTH = 32;
        const int IMAGE_HEIGHT = 32;
        const int IMAGE_CHANNELS = 3;
        const int IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

        // CIFAR-10 normalization constants
        const std::array<float, 3> mean = {0.4914f, 0.4822f, 0.4465f};
        const std::array<float, 3> std = {0.2023f, 0.1994f, 0.2010f};

        for (const auto& filename : filenames) {
            fs::path filepath = fs::path(dataset_base_path) / filename;
            std::ifstream file(filepath, std::ios::binary);

            if (!file.is_open()) {
                throw std::runtime_error("Could not open file " + filepath.string());
            }

            // Get file size and validate
            file.seekg(0, std::ios::end);
            long long file_size = file.tellg();
            file.seekg(0, std::ios::beg);

            if (file_size % IMAGE_SIZE_BYTES != 0) {
                throw std::runtime_error("File size is not a multiple of " + std::to_string(IMAGE_SIZE_BYTES) + " bytes for file: " + filepath.string());
            }

            int num_records = file_size / IMAGE_SIZE_BYTES;

            for (int i = 0; i < num_records; ++i) {
                uint8_t label_byte;
                file.read(reinterpret_cast<char*>(&label_byte), 1);
                labels.push_back(label_byte);

                std::vector<uint8_t> image_data_raw(IMAGE_SIZE_BYTES - 1);
                file.read(reinterpret_cast<char*>(image_data_raw.data()), IMAGE_SIZE_BYTES - 1);

                // Create a host-side buffer for the processed image data
                std::vector<float> host_image_data(IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH);

                // Convert to float and normalize per-channel, storing in host_image_data
                for (int c = 0; c < IMAGE_CHANNELS; ++c) {
                    for (int h = 0; h < IMAGE_HEIGHT; ++h) {
                        for (int w = 0; w < IMAGE_WIDTH; ++w) {
                            // Calculate linear index for the output buffer [C, H, W]
                            size_t output_linear_index = c * IMAGE_PIXELS + h * IMAGE_WIDTH + w;

                            // Data is stored RRR...GGG...BBB...
                            // So, the raw_index needs to be adjusted based on the channel
                            int pixel_index_in_channel = h * IMAGE_WIDTH + w;
                            int original_raw_index = 0;
                            if (c == 0) original_raw_index = pixel_index_in_channel; // Red channel
                            else if (c == 1) original_raw_index = IMAGE_PIXELS + pixel_index_in_channel; // Green channel
                            else if (c == 2) original_raw_index = 2 * IMAGE_PIXELS + pixel_index_in_channel; // Blue channel

                            float pixel_value = static_cast<float>(image_data_raw[original_raw_index]) / 255.0f;
                            pixel_value = (pixel_value - mean[c]) / std[c];
                            host_image_data[output_linear_index] = pixel_value;
                        }
                    }
                }
                images.push_back(host_image_data);
            }
        }
    }
};

} // namespace lenet5 
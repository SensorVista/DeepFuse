#include <gtest/gtest.h>

#include <dnn/utils/mnist_loader.cuh>

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <random>

namespace dnn {
namespace test {

class MNISTLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test files
        create_test_images_file();
        create_test_labels_file();
    }

    void TearDown() override {
        // Clean up test files
        std::filesystem::remove(images_filepath);
        std::filesystem::remove(labels_filepath);
    }

    void create_test_images_file() {
        std::ofstream file(images_filepath, std::ios::binary);
        
        // Write magic number (2051)
        uint32_t magic = 2051;
        file.write(reinterpret_cast<char*>(&magic), sizeof(magic));
        
        // Write number of images (10)
        uint32_t num_images = 10;
        file.write(reinterpret_cast<char*>(&num_images), sizeof(num_images));
        
        // Write rows (28) and columns (28)
        uint32_t rows = 28;
        uint32_t cols = 28;
        file.write(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<char*>(&cols), sizeof(cols));
        
        // Write test image data
        std::vector<uint8_t> image_data(28*28, 0);
        for (int i = 0; i < 10; ++i) {
            // Fill with some pattern
            for (int j = 0; j < 28*28; ++j) {
                image_data[j] = static_cast<uint8_t>((i + j) % 256);
            }
            file.write(reinterpret_cast<char*>(image_data.data()), 28*28);
        }
    }

    void create_test_labels_file() {
        std::ofstream file(labels_filepath, std::ios::binary);
        
        // Write magic number (2049)
        uint32_t magic = 2049;
        file.write(reinterpret_cast<char*>(&magic), sizeof(magic));
        
        // Write number of labels (10)
        uint32_t num_labels = 10;
        file.write(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
        
        // Write test labels (0-9)
        std::vector<uint8_t> labels(10);
        for (int i = 0; i < 10; ++i) {
            labels[i] = static_cast<uint8_t>(i % 10);
        }
        file.write(reinterpret_cast<char*>(labels.data()), 10);
    }

    const std::string images_filepath = "test_images.idx3-ubyte";
    const std::string labels_filepath = "test_labels.idx1-ubyte";
    const int test_num_items = 10;
    const int image_size = 28*28;
};

TEST_F(MNISTLoaderTest, LoadImagesSuccess) {
    auto images = MNISTLoader::load_images(images_filepath, test_num_items, image_size);
    
    EXPECT_EQ(images.size(), test_num_items);
    for (const auto& image : images) {
        EXPECT_EQ(image.size(), image_size);
    }
    
    for (int i = 0; i < image_size; ++i) {
        float expected = static_cast<float>(i % 256) / 255.0f;
        EXPECT_NEAR(images[0][i], expected, 1e-5);
    }
}

TEST_F(MNISTLoaderTest, LoadLabelsSuccess) {
    auto labels = MNISTLoader::load_labels(labels_filepath, test_num_items);
    
    EXPECT_EQ(labels.size(), test_num_items);
    for (int i = 0; i < test_num_items; ++i) {
        EXPECT_EQ(labels[i], i % 10);
    }
}

TEST_F(MNISTLoaderTest, LoadImagesInvalidFile) {
    EXPECT_THROW(
        MNISTLoader::load_images("nonexistent_file", test_num_items, image_size),
        std::runtime_error
    );
}

TEST_F(MNISTLoaderTest, LoadLabelsInvalidFile) {
    EXPECT_THROW(
        MNISTLoader::load_labels("nonexistent_file", test_num_items),
        std::runtime_error
    );
}

TEST_F(MNISTLoaderTest, LoadImagesPartial) {
    const int partial_num = 5;
    auto images = MNISTLoader::load_images(images_filepath, partial_num, image_size);
    EXPECT_EQ(images.size(), partial_num);
}

TEST_F(MNISTLoaderTest, LoadLabelsPartial) {
    const int partial_num = 5;
    auto labels = MNISTLoader::load_labels(labels_filepath, partial_num);
    EXPECT_EQ(labels.size(), partial_num);
}

TEST_F(MNISTLoaderTest, ImageNormalization) {
    auto images = MNISTLoader::load_images(images_filepath, 1, image_size);
    for (float pixel : images[0]) {
        EXPECT_GE(pixel, 0.0f);
        EXPECT_LE(pixel, 1.0f);
    }
}

TEST_F(MNISTLoaderTest, HeaderSkipping) {
    std::ifstream file(images_filepath, std::ios::binary);
    
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    EXPECT_EQ(magic, 2051);
    
    uint32_t num_images;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    EXPECT_EQ(num_images, test_num_items);
    
    uint32_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    EXPECT_EQ(rows, 28);
    EXPECT_EQ(cols, 28);
    
    auto images = MNISTLoader::load_images(images_filepath, 1, image_size);
    EXPECT_NEAR(images[0][0], 0.0f, 1e-5);
}

} // namespace test
} // namespace dnn
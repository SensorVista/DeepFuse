#include <dnn/core/cuda.cuh>
#include <dnn/losses/cross_entropy.cuh>
#include <dnn/optimizers/sgd_optimizer.cuh>
#include <dnn/utils/cifar10_loader.cuh>
#include <dnn/utils/common.cuh>

#include <iostream>
#include <filesystem>
#include <numeric>
#include <random>
#include <vector>

void print_usage(const char* program_name) {
    std::string name = program_name;
    size_t last_slash = name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        name = name.substr(last_slash + 1);
    }
    size_t last_dot = name.find_last_of('.');
    if (last_dot != std::string::npos) {
        name = name.substr(0, last_dot);
    }

    std::cerr << "+----------------------------------------------------------+" << std::endl;
    std::cerr << "|                        ResNet20                          |" << std::endl;
    std::cerr << "+----------------------------------------------------------+" << std::endl;
    std::cerr << "\nUsage: " << name << " <datasets_root_path>" << std::endl;
    std::cerr << "Example: " << name << " ../datasets" << std::endl;
    std::cerr << "\nExpected CIFAR-10 dataset structure:" << std::endl;
    std::cerr << "<datasets_root_path>/" << std::endl;
    std::cerr << "    +-- CIFAR-10/" << std::endl;
    std::cerr << "        +-- data_batch_1.bin" << std::endl;
    std::cerr << "        +-- data_batch_2.bin" << std::endl;
    std::cerr << "        +-- data_batch_3.bin" << std::endl;
    std::cerr << "        +-- data_batch_4.bin" << std::endl;
    std::cerr << "        +-- data_batch_5.bin" << std::endl;
    std::cerr << "        +-- test_batch.bin" << std::endl;
    std::cerr << "\nDataset sizes:" << std::endl;
    std::cerr << "- Training: 50,000 images" << std::endl;
    std::cerr << "- Test: 10,000 images" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // Initialize CUDA device
        dnn::Cuda cuda;
        cuda.dump_info();

        if (argc != 2) {
            print_usage(argv[0]);
            return 1;
        }

        std::filesystem::path datasets_root = argv[1];
        if (!std::filesystem::exists(datasets_root)) {
            std::cerr << "Error: Dataset root path '" << datasets_root << "' does not exist." << std::endl;
            print_usage(argv[0]);
            return 1;
        }

        const std::string cifar10_train_path = (datasets_root / "CIFAR-10").string();
        const std::string cifar10_test_path = (datasets_root / "CIFAR-10").string();

        // Load CIFAR-10 dataset
        const int num_train_images = 50000;
        const int num_test_images = 10000;
        const int image_dim = 32;
        const int channels = 3;
        const int output_classes = 10;

        std::vector<std::vector<float>> train_images;
        std::vector<uint8_t> train_labels;
        std::vector<std::vector<float>> test_images;
        std::vector<uint8_t> test_labels;

        std::cout << "Loading training data from: " << cifar10_train_path << std::endl;
        dnn::CIFAR10Loader::load(true, train_images, train_labels, cifar10_train_path);
        std::cout << "Loaded " << train_images.size() << " training images and " << train_labels.size() << " training labels." << std::endl;

        std::cout << "Loading test data from: " << cifar10_test_path << std::endl;
        dnn::CIFAR10Loader::load(false, test_images, test_labels, cifar10_test_path);
        std::cout << "Loaded " << test_images.size() << " test images and " << test_labels.size() << " test labels." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
} 

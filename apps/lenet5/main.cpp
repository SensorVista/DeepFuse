#include <dnn/models/lenet5.cuh>
#include <dnn/utils/mnist_loader.cuh>
#include <dnn/utils/common.cuh>
#include <dnn/core/device.cuh>

#include <iostream>
#include <filesystem>
#include <numeric>
#include <vector>
#include <random>

void print_usage(const char* program_name) {
    // Extract just the filename without path and extension
    std::string name = program_name;
    int last_slash = name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        name = name.substr(last_slash + 1);
    }
    int last_dot = name.find_last_of('.');
    if (last_dot != std::string::npos) {
        name = name.substr(0, last_dot);
    }

    std::cerr << "+----------------------------------------------------------+" << std::endl;
    std::cerr << "|                        LeNet-5                           |" << std::endl;
    std::cerr << "+----------------------------------------------------------+" << std::endl;
    std::cerr << "\nUsage: " << name << " <datasets_root_path>" << std::endl;
    std::cerr << "Example: " << name << " ../datasets" << std::endl;
    std::cerr << "\nExpected MNIST dataset structure:" << std::endl;
    std::cerr << "<datasets_root_path>/" << std::endl;
    std::cerr << "    +-- MNIST/" << std::endl;
    std::cerr << "        +-- training/" << std::endl;
    std::cerr << "        |   +-- train-images.idx3-ubyte  (Training images)" << std::endl;
    std::cerr << "        |   +-- train-labels.idx1-ubyte  (Training labels)" << std::endl;
    std::cerr << "        +-- test/" << std::endl;
    std::cerr << "            +-- t10k-images.idx3-ubyte   (Test images)" << std::endl;
    std::cerr << "            +-- t10k-labels.idx1-ubyte   (Test labels)" << std::endl;
    std::cerr << "\nFile formats:" << std::endl;
    std::cerr << "- Images: IDX3 format, 28x28 grayscale images" << std::endl;
    std::cerr << "- Labels: IDX1 format, single byte per label (0-9)" << std::endl;
    std::cerr << "\nDataset sizes:" << std::endl;
    std::cerr << "- Training: 60,000 images" << std::endl;
    std::cerr << "- Test: 10,000 images" << std::endl;
}

void train(
    dnn::TrainingModel<float>* network, 
    int num_epochs, int batch_size, int channels, int input_dim, int output_dim,
    int num_train_images, std::vector<std::vector<float>> train_images, std::vector<uint8_t> train_labels,
    int num_test_images, std::vector<std::vector<float>> test_images, std::vector<uint8_t> test_labels) {   
    
    // Radomize samples
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    std::mt19937 rng(std::random_device{}());
    std::vector<int> indices(num_train_images);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0;

        // // Learning rate schedule
        // if (epoch == 10 || epoch == 20) {
        //     network->set_learning_rate(network->learning_rate() * 0.1f);
        //     std::cout << "Learning rate reduced to: " << network->learning_rate() << std::endl;
        // }

        // Ensure every epoch gets a new sample order
        std::shuffle(indices.begin(), indices.end(), rng);

        // Training
        for (int i = 0; i < num_train_images; i += batch_size) {
            int current_batch_size = std::min(batch_size, num_train_images - i);

            dnn::tensor<float> input({ current_batch_size, channels, input_dim, input_dim });
            dnn::tensor<float> target({ current_batch_size, output_dim });

            // Convert batch data to float
            std::vector<float> input_data(current_batch_size * input_dim * input_dim);
            std::vector<float> target_data(current_batch_size * output_dim, 0.0f);

            for (int b = 0; b < current_batch_size; ++b) {
                for (int j = 0; j < input_dim * input_dim; ++j) {
                    input_data[b * input_dim * input_dim + j] = (train_images[indices[i + b]][j] - 127.5f) / 127.5f;
                }
                
                int label = train_labels[indices[i + b]];
                if (label >= output_dim)
                    throw std::out_of_range("Label index exceeds output_dim");
                target_data[b * output_dim + label] = 1.0f;
            }

            // Upload batch data
            input.upload(input_data.data());
            target.upload(target_data.data());

            // Train on batch
            network->train_step(input, target);

            // Accumlate losses
            total_loss += network->loss();
        }

        // Validation
        int correct_predictions = 0;
        for (int i = 0; i < num_test_images; ++i) {
            dnn::tensor<float> input({ 1, 1, input_dim, input_dim });
            // Normalize test images similarly
            std::vector<float> single_test_image_data(input_dim * input_dim);
            for(int j = 0; j < input_dim * input_dim; ++j) {
                single_test_image_data[j] = (test_images[i][j] - 127.5f) / 127.5f;
            }
            input.upload(single_test_image_data.data());

            auto output = network->forward(input);
            std::vector<float> host_output(output.size());
            output.download(host_output.data());

            // Determine predicted class
            int predicted_class = std::distance(host_output.begin(), std::max_element(host_output.begin(), host_output.end()));
            if (predicted_class == test_labels[i]) {
                ++correct_predictions;
            }
        }

        float accuracy = static_cast<float>(correct_predictions) / num_test_images;
        std::cout
            << "Epoch " << (epoch + 1) << "/" << num_epochs << ": "
            << "Accuracy = " << (accuracy * 100.0f) << "%" << ", "
            << "Loss = " << (total_loss / static_cast<float>(num_train_images))
            << std::endl;
    }
}

int main(int argc, char* argv[]) {
    int result = 0;

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

        const std::string train_images_path = (datasets_root / "MNIST/training/train-images.idx3-ubyte").string();
        const std::string train_labels_path = (datasets_root / "MNIST/training/train-labels.idx1-ubyte").string();
        const std::string test_images_path = (datasets_root / "MNIST/test/t10k-images.idx3-ubyte").string();
        const std::string test_labels_path = (datasets_root / "MNIST/test/t10k-labels.idx1-ubyte").string();

        // Verify all required files exist
        for (const auto& path : {train_images_path, train_labels_path, test_images_path, test_labels_path}) {
            if (!std::filesystem::exists(path)) {
                std::cerr << "Error: Required file not found: " << path << std::endl;
                return 1;
            }
        }

        // Load MNIST dataset
        const int num_train_images = 60000;
        const int num_test_images = 10000;
        int image_dim = 28;
        int channels = 1;
        int image_size = image_dim * image_dim * channels;

        std::cout << "Loading training images from: " << train_images_path << std::endl;
        auto train_images = dnn::MNISTLoader::load_images(train_images_path, num_train_images, image_size);
        std::cout << "Loaded " << train_images.size() << " training images." << std::endl;

        std::cout << "Loading training labels from: " << train_labels_path << std::endl;
        auto train_labels = dnn::MNISTLoader::load_labels(train_labels_path, num_train_images);
        std::cout << "Loaded " << train_labels.size() << " training labels." << std::endl;

        std::cout << "Loading test images from: " << test_images_path << std::endl;
        auto test_images = dnn::MNISTLoader::load_images(test_images_path, num_test_images, image_size);
        std::cout << "Loaded " << test_images.size() << " test images." << std::endl;

        std::cout << "Loading test labels from: " << test_labels_path << std::endl;
        auto test_labels = dnn::MNISTLoader::load_labels(test_labels_path, num_test_images);
        std::cout << "Loaded " << test_labels.size() << " test labels." << std::endl;

        // Create LeNet-5 network with float precision
        dnn::LeNet5<float> network;
        network.set_loss(std::make_unique<dnn::CrossEntropyLoss<float>>());
        network.set_optimizer(std::make_unique<dnn::SGDOptimizer<float>>(0.01f, 0.9f, 0.0f));

        // Begin training
        const int num_epochs = 10;
        const int batch_size = 64;

        auto pad_images_to_32x32 = [](const std::vector<std::vector<float>>& images28, int pad_y, int pad_x) {
            const int new_dim = 28 + 2 * pad_y;
            std::vector<std::vector<float>> padded(images28.size(), std::vector<float>(new_dim * new_dim, 0.0f));
            for (int i = 0; i < images28.size(); ++i) {
                for (int y = 0; y < 28; ++y) {
                    for (int x = 0; x < 28; ++x) {
                        padded[i][(y + pad_y) * new_dim + (x + pad_x)] = images28[i][y * 28 + x];
                    }
                }
            }
            return padded;
        };

        // Pad images from 28x28 to 32x32
        train_images = pad_images_to_32x32(train_images, 2, 2);
        test_images  = pad_images_to_32x32(test_images,  2, 2);
        image_dim = 32;

        train(&network, num_epochs, batch_size, channels, image_dim, 10, num_train_images, train_images, train_labels, num_test_images, test_images, test_labels);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        result = 1;
    }

    return result;
} 

#include <dnn/core/cuda.cuh>
#include <dnn/models/perceptron.cuh>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

int main() {
    try {
        // Initialize CUDA device
        dnn::Cuda cuda;
        cuda.dump_info();

        // Generate synthetic training data for binary classification
        int num_train_samples = 1000;
        int num_test_samples = 200;
        int input_dim = 2;  // 2D input features
        int output_dim = 1; // Binary classification

        // Create random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // Generate training data
        std::vector<std::vector<float>> train_images(num_train_samples);
        std::vector<float> train_labels(num_train_samples);
        
        for (int i = 0; i < num_train_samples; ++i) {
            // Generate 2D input features
            train_images[i] = {dist(gen), dist(gen)};
            
            // Generate binary label based on XOR-like pattern
            // Label is 1 if points are in first or third quadrant
            train_labels[i] = (train_images[i][0] * train_images[i][1] > 0) ? 1.0f : 0.0f;
        }

        // Generate test data
        std::vector<std::vector<float>> test_images(num_test_samples);
        std::vector<float> test_labels(num_test_samples);
        
        for (int i = 0; i < num_test_samples; ++i) {
            test_images[i] = {dist(gen), dist(gen)};
            test_labels[i] = (test_images[i][0] * test_images[i][1] > 0) ? 1.0f : 0.0f;
        }

        // Create Perceptron network
        dnn::Perceptron<float> network(input_dim, 4, output_dim);

        // Training parameters
        int num_epochs = 1000;
        int batch_size = 32;

        // Randomize samples
        std::vector<int> indices(num_train_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(std::random_device{}());

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            float total_loss = 0;
            
            // Ensure every epoch gets a new sample order
            std::shuffle(indices.begin(), indices.end(), rng);

            // Training
            for (int i = 0; i < num_train_samples; i += batch_size) {
                int current_batch_size = std::min(batch_size, num_train_samples - i);

                dnn::tensor<float> input({current_batch_size, input_dim});
                dnn::tensor<float> target({current_batch_size, output_dim});

                // Convert batch data
                std::vector<float> input_data(current_batch_size * input_dim);
                std::vector<float> target_data(current_batch_size * output_dim);

                for (int b = 0; b < current_batch_size; ++b) {
                    // Copy input features
                    input_data[b * input_dim] = train_images[indices[i + b]][0];
                    input_data[b * input_dim + 1] = train_images[indices[i + b]][1];
                    
                    // Set target
                    target_data[b] = train_labels[indices[i + b]];
                }

                // Upload batch data
                input.upload(input_data.data());
                target.upload(target_data.data());

                // Train on batch
                network.train_step(input, target);
                total_loss += network.loss();
            }

            // Validation
            int correct_predictions = 0;
            for (int i = 0; i < num_test_samples; ++i) {
                dnn::tensor<float> input({1, input_dim});
                input.upload(test_images[i].data());

                auto output = network.forward(input);
                std::vector<float> host_output(output.size());
                output.download(host_output.data());

                // Determine predicted class
                float predicted = host_output[0] > 0.5f ? 1.0f : 0.0f;
                if (predicted == test_labels[i]) {
                    ++correct_predictions;
                }
            }

            float accuracy = correct_predictions / static_cast<float>(num_test_samples);
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << ": "
                        << "Accuracy = " << (accuracy * 100.0f) << "%" << ", "
                        << "Loss = " << (total_loss / (num_train_samples / batch_size))
                        << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
} 

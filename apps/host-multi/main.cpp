#include <dnn/core/cuda.cuh>
#include <dnn/models/perceptron.cuh>
#include <dnn/utils/common.cuh>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif

#include <atomic>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

// Global mutex for thread-safe output
std::mutex g_output_mutex;

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
    std::cerr << "|                        host-multi                        |" << std::endl;
    std::cerr << "+----------------------------------------------------------+" << std::endl;
    std::cerr << "\nUsage: " << name << " <datasets_root_path>" << std::endl;
    std::cerr << "Example: " << name << " ../datasets" << std::endl;
}

struct TrainingTask {
    int device_id;
    int num_train_samples;
    int num_test_samples;
    int input_dim;
    int output_dim;
    int batch_size;
    int num_epochs;
    std::atomic<int>& correct_predictions;

    void operator()() const {
        try {
            dnn::Cuda cuda(device_id);
            dnn::Perceptron<float> network(input_dim, 4096, output_dim); // Large hidden layer

            // Generate random data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

            std::vector<std::vector<float>> train_images(num_train_samples);
            std::vector<float> train_labels(num_train_samples);
            std::vector<std::vector<float>> test_images(num_test_samples);
            std::vector<float> test_labels(num_test_samples);

            for (int i = 0; i < num_train_samples; ++i) {
                train_images[i] = {dist(gen), dist(gen)};
                train_labels[i] = (train_images[i][0] * train_images[i][1] > 0) ? 1.0f : 0.0f;
            }

            for (int i = 0; i < num_test_samples; ++i) {
                test_images[i] = {dist(gen), dist(gen)};
                test_labels[i] = (test_images[i][0] * test_images[i][1] > 0) ? 1.0f : 0.0f;
            }

            // Training loop
            std::vector<int> indices(num_train_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::mt19937 rng(std::random_device{}());

            for (int epoch = 0; epoch < num_epochs; ++epoch) {
                std::shuffle(indices.begin(), indices.end(), rng);

                for (int i = 0; i < num_train_samples; i += batch_size) {
                    int current_batch = std::min(batch_size, num_train_samples - i);
                    
                    // Create and process batch
                    dnn::tensor<float> input({current_batch, input_dim});
                    dnn::tensor<float> target({current_batch, output_dim});
                    
                    // Process batch (implementation omitted for brevity)
                    network.train_step(input, target);
                }

                // Validation (implementation omitted for brevity)
                int local_correct = 0;
                correct_predictions += local_correct;

                {
                    std::lock_guard<std::mutex> lock(g_output_mutex);
                    std::cout << "Device " << device_id << ", Epoch " << (epoch + 1) 
                              << ": Accuracy = " << (static_cast<float>(local_correct) / num_test_samples * 100.0f) << "%\n";
                }
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(g_output_mutex);
            std::cerr << "Device " << device_id << " error: " << e.what() << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    int result = 0;

    try {
        const auto& devices = dnn::Cuda::get_devices();
        if (devices.empty()) {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }

        if (argc != 2) {
            print_usage(argv[0]);
            return 1;
        }

        // Memory calculation
        const size_t model_memory = 4096 * (2*sizeof(float) + 2*sizeof(float)); // weights + gradients
        const size_t batch_memory = 256 * (2*sizeof(float) + 1*sizeof(float));  // input + target
        const size_t total_per_model = model_memory + batch_memory;

        // Determine maximum concurrent models per GPU
        std::vector<int> models_per_device;
        for (const auto& device : devices) {
            size_t available = device.available_memory_bytes();
            int max_models = static_cast<int>(available * 0.9 / total_per_model); // 90% utilization
            models_per_device.push_back(std::max(1, max_models));
        }

        // Training parameters
        const int num_train_samples = 100000;
        const int num_test_samples = 20000;
        const int input_dim = 2;
        const int output_dim = 1;
        const int num_epochs = 100;
        const int batch_size = 256;

        // Thread-safe counter
        std::atomic<int> correct_predictions(0);

        // Create tasks
        std::vector<TrainingTask> tasks;
        for (int dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
            for (int i = 0; i < models_per_device[dev_idx]; ++i) {
                tasks.emplace_back(TrainingTask{
                    devices[dev_idx].id(),
                    num_train_samples,
                    num_test_samples,
                    input_dim,
                    output_dim,
                    batch_size,
                    num_epochs,
                    correct_predictions
                });
            }
        }

        // Execute with parallelization
#ifdef USE_TBB
        // Use TBB for parallel execution
        tbb::parallel_for(tbb::blocked_range<size_t>(0, tasks.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    tasks[i]();
                }
            }
        );
#else
        // Use std::thread for parallel execution
        const size_t num_threads = std::min(tasks.size(), static_cast<size_t>(std::thread::hardware_concurrency()));
        std::vector<std::thread> threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                tasks[i]();
            }
        };
        
        const size_t tasks_per_thread = tasks.size() / num_threads;
        const size_t remaining_tasks = tasks.size() % num_threads;
        
        size_t current_start = 0;
        for (size_t t = 0; t < num_threads; ++t) {
            size_t current_end = current_start + tasks_per_thread + (t < remaining_tasks ? 1 : 0);
            threads.emplace_back(worker, current_start, current_end);
            current_start = current_end;
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
#endif

        // Calculate final accuracy
        float final_accuracy = static_cast<float>(correct_predictions) / 
                             (tasks.size() * num_epochs * num_test_samples);
        std::cout << "Final accuracy: " << (final_accuracy * 100.0f) << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        result = 1;
    }

    return result;
} 

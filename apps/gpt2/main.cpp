#include <dnn/core/cuda.cuh>
#include <dnn/models/gpt2.cuh>
#include <dnn/tokens/bpe_tokenizer.cuh>
#include <dnn/tokens/vocab_loader.cuh>
#include <dnn/utils/common.cuh>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

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
    std::cerr << "|                        GPT-2                             |" << std::endl;
    std::cerr << "+----------------------------------------------------------+" << std::endl;
    std::cerr << "\nUsage: " << name << " <datasets_root_path>" << std::endl;
    std::cerr << "Example: " << name << " ../datasets" << std::endl;
    std::cerr << "\nExpected dataset structure:" << std::endl;
    std::cerr << "<datasets_root_path>/" << std::endl;
    std::cerr << "    +-- NLP-LLM/" << std::endl;
    std::cerr << "    |   +-- vocab.json   (Vocabulary file)" << std::endl;
    std::cerr << "    |   +-- merges.txt   (Merges file)" << std::endl;
    std::cerr << "    |   +-- training/" << std::endl;
    std::cerr << "            +-- *.txt   (Text files for training)" << std::endl;
    std::cerr << "\nText files should contain multiple paragraphs separated by blank lines.\n\n";
}

// Multithreaded: Collect all .txt file paths in a directory tree
std::vector<std::filesystem::path> get_txt_file_paths(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> result;
    std::mutex result_mutex;
    std::queue<std::filesystem::path> work_queue;
    work_queue.push(dir);
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> done{false};
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 8; // fallback
    auto worker = [&]() {
        while (true) {
            std::filesystem::path current;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [&] { return !work_queue.empty() || done; });
                if (work_queue.empty()) break;
                current = work_queue.front();
                work_queue.pop();
            }
            for (const auto& entry : std::filesystem::directory_iterator(current)) {
                if (entry.is_directory()) {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    work_queue.push(entry.path());
                    cv.notify_all();
                } else if (entry.path().extension() == ".txt") {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    result.push_back(entry.path());
                }
            }
        }
    };
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < num_threads; ++i)
        threads.emplace_back(worker);
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        done = false;
    }
    // Wait for all work to finish
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        while (!work_queue.empty()) {
            cv.wait(lock);
        }
        done = true;
        cv.notify_all();
    }
    for (auto& t : threads) t.join();
    return result;
}

// Helper: Split file paths into train/val sets (shuffles for randomness)
std::pair<std::vector<std::filesystem::path>, std::vector<std::filesystem::path>>
split_train_val_paths(const std::vector<std::filesystem::path>& paths, float val_ratio = 0.01f) {
    std::vector<std::filesystem::path> shuffled = paths;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffled.begin(), shuffled.end(), g); // Shuffle for randomness
    size_t val_size = static_cast<size_t>(shuffled.size() * val_ratio);
    return {
        std::vector<std::filesystem::path>(shuffled.begin() + val_size, shuffled.end()), // train
        std::vector<std::filesystem::path>(shuffled.begin(), shuffled.begin() + val_size) // val
    };
}

// Helper: Read file contents as UTF-8
std::string read_file_contents(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + path.string());
    
    // Read the entire file into a string
    std::string content;
    file.seekg(0, std::ios::end);
    content.reserve(file.tellg());
    file.seekg(0, std::ios::beg);
    content.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    return content;
}

struct TrainingConfig {
    int max_seq_len = 1024;      // Kept at 1024 for full context
    int batch_size = 16;          // Batch size for training
    int num_epochs = 1;          // Number of training epochs
    float learning_rate = 5e-5;  // Initial learning rate
    float warmup_steps = 4000;   // Warmup steps for learning rate
    float max_grad_norm = 1.0f;  // Maximum gradient norm for clipping
    int stride = 128;            // Stride for sliding window
    int eval_steps = 100;        // Steps between evaluations
    int save_steps = 1000;       // Steps between model saves
    float early_stop_patience = 3.0f;  // Early stopping patience
    std::string output_dir = "checkpoints";  // Directory for saving models
    float data_fraction = 0.01f;  // Fraction of data to use (1.0 = all, 0.25 = quarter)  USed for testing/debugging and should be 1.0
};

float get_learning_rate(int step, const TrainingConfig& config) {
    if (step < config.warmup_steps) {
        return config.learning_rate * (step / config.warmup_steps);
    }
    return config.learning_rate * (1.0f - (step - config.warmup_steps) / 
           (config.num_epochs * config.warmup_steps));
}

// Create batches from a sequence of tokens
std::vector<std::pair<std::vector<int>, std::vector<int>>> 
create_batches(const std::vector<int>& tokens, const TrainingConfig& config) {
    std::vector<std::pair<std::vector<int>, std::vector<int>>> batches;
    if (tokens.size() < config.max_seq_len + 1) {
        return batches; // Not enough tokens for a single batch
    }
    for (size_t i = 0; i + config.max_seq_len + 1 <= tokens.size(); i += config.stride) {
        std::vector<int> input_tokens(tokens.begin() + i, tokens.begin() + i + config.max_seq_len);
        std::vector<int> target_tokens(tokens.begin() + i + 1, tokens.begin() + i + config.max_seq_len + 1);
        batches.emplace_back(input_tokens, target_tokens);
        if (batches.size() >= static_cast<size_t>(config.batch_size)) {
            break;
        }
    }
    return batches;
}

float evaluate(dnn::Gpt2<float>* model,
               const std::vector<std::string>& val_docs,
               const TrainingConfig& config) {
    float total_loss = 0.0f;
    int num_steps = 0;
    for (const auto& doc : val_docs) {
        auto tokens = model->tokenizer()->encode(doc, true);
        auto batches = create_batches(tokens, config);
        for (const auto& [input_tokens, target_tokens] : batches) {
            model->train_step(input_tokens, target_tokens);
            total_loss += model->loss();
            num_steps++;
        }
    }
    return num_steps > 0 ? total_loss / num_steps : 0.0f;
}

// Streaming/chunked train: processes only a small batch of files at a time
void train(dnn::Gpt2<float>* model,
           const std::vector<std::filesystem::path>& train_paths,
           const std::vector<std::filesystem::path>& val_paths,
           const TrainingConfig& config) {
    std::filesystem::create_directories(config.output_dir); // Ensure output dir exists
    int total_steps = 0;
    float total_loss = 0.0f;
    float best_val_loss = std::numeric_limits<float>::infinity();
    int no_improve_steps = 0;
    model->set_attention_mask(config.max_seq_len); // Set attention mask once
    auto start_time = std::chrono::high_resolution_clock::now();
    std::random_device rd;
    std::mt19937 g(rd());
    // Limit data to a fraction if specified
    size_t train_limit = static_cast<size_t>(train_paths.size() * config.data_fraction);
    size_t val_limit = static_cast<size_t>(val_paths.size() * config.data_fraction);
    for (int epoch = 0; epoch < config.num_epochs; ++epoch) {
        std::cout << "[DEBUG] Starting epoch " << (epoch + 1) << "/" << config.num_epochs << std::endl;
        std::vector<std::filesystem::path> shuffled = train_paths;
        std::shuffle(shuffled.begin(), shuffled.end(), g); // Shuffle each epoch
        if (train_limit < shuffled.size()) shuffled.resize(train_limit);
        for (size_t i = 0; i < shuffled.size(); i += config.batch_size) {
            std::cout << "[DEBUG] Loading batch " << (i / config.batch_size + 1)
                      << " (" << i << "/" << shuffled.size() << ")" << std::endl;
            std::vector<std::string> batch_docs;
            // Read a batch of files into memory (only this batch)
            for (size_t j = i; j < i + config.batch_size && j < shuffled.size(); ++j) {
                batch_docs.push_back(read_file_contents(shuffled[j]));
            }
            std::vector<std::vector<int>> batch_input_tokens;
            std::vector<std::vector<int>> batch_target_tokens;
            for (const auto& doc : batch_docs) {
                auto tokens = model->tokenizer()->encode(doc, true); // On-the-fly tokenization
                auto batches = create_batches(tokens, config); // Create sliding window batches
                for (const auto& [input_tokens, target_tokens] : batches) {
                    batch_input_tokens.push_back(input_tokens);
                    batch_target_tokens.push_back(target_tokens);
                    if (batch_input_tokens.size() == static_cast<size_t>(config.batch_size)) {
                        for (size_t b = 0; b < batch_input_tokens.size(); ++b) {
                            model->train_step(batch_input_tokens[b], batch_target_tokens[b]);
                            total_loss += model->loss();
                            total_steps++;
                            if (total_steps % 10 == 0) { // Print every 10 steps
                                std::cout << "[DEBUG] Step " << total_steps
                                          << " | Loss: " << model->loss() << std::endl;
                            }
                        }
                        batch_input_tokens.clear();
                        batch_target_tokens.clear();
                        // Periodic evaluation
                        if (total_steps % config.eval_steps == 0) {
                            std::cout << "[DEBUG] Starting validation..." << std::endl;
                            float avg_train_loss = total_loss / config.eval_steps;
                            float val_loss = 0.0f;
                            int val_steps = 0;
                            int val_count = 0;
                            for (const auto& val_path : val_paths) {
                                if (val_count++ >= val_limit) break;
                                std::string val_doc = read_file_contents(val_path);
                                auto val_tokens = model->tokenizer()->encode(val_doc, true);
                                auto val_batches = create_batches(val_tokens, config);
                                for (const auto& [vin, vtarget] : val_batches) {
                                    model->train_step(vin, vtarget);
                                    val_loss += model->loss();
                                    val_steps++;
                                }
                            }
                            std::cout << "[DEBUG] Validation complete." << std::endl;
                            val_loss = val_steps > 0 ? val_loss / val_steps : 0.0f;
                            auto current_time = std::chrono::high_resolution_clock::now();
                            auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
                            std::cout << "[Epoch " << (epoch + 1) << "/" << config.num_epochs
                                      << "] Step " << total_steps
                                      << " Train Loss: " << avg_train_loss
                                      << " Val Loss: " << val_loss
                                      << " LR: " << get_learning_rate(total_steps, config)
                                      << " Time: " << duration.count() << "s" << std::endl;
                            total_loss = 0.0f;
                            // Early stopping
                            if (val_loss < best_val_loss) {
                                best_val_loss = val_loss;
                                no_improve_steps = 0;
                                std::string model_path = config.output_dir + "/best_model.onnx";
                                model->save(model_path); // Save best model
                            } else {
                                no_improve_steps++;
                                if (no_improve_steps >= config.early_stop_patience * (config.eval_steps / config.batch_size)) {
                                    std::cout << "Early stopping triggered after " << total_steps << " steps" << std::endl;
                                    return;
                                }
                            }
                        }
                        // Periodic checkpointing
                        if (total_steps % config.save_steps == 0) {
                            std::string checkpoint_path = config.output_dir + "/checkpoint_" + std::to_string(total_steps) + ".onnx";
                            model->save(checkpoint_path);
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        dnn::Cuda cuda;
        cuda.dump_info(); // Print CUDA device info
        if (argc != 2) {
            print_usage(argv[0]);
            return 1;
        }
        std::filesystem::path datasets_root = argv[1];
        std::filesystem::path vocab_path = datasets_root / "NLP-LLM/vocab.json";
        std::filesystem::path training_path = datasets_root / "NLP-LLM/training";
        if (!std::filesystem::exists(vocab_path) || !std::filesystem::exists(training_path)) {
            std::cerr << "Missing vocab.json or training/ directory." << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        auto vocab_loader = std::make_shared<dnn::VocabLoader>();
        try {
            vocab_loader->load_from_file(vocab_path.string());
            if (vocab_loader->size() == 0) {
                throw std::runtime_error("Vocabulary file is empty");
            }
            if (vocab_loader->token_to_id("<|endoftext|>") == -1) {
                throw std::runtime_error("Vocabulary missing required special token: <|endoftext|>");
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading vocabulary: " << e.what() << std::endl;
            return 1;
        }
        auto tokenizer = std::make_shared<dnn::BpeTokenizer>(vocab_loader);
        auto all_paths = get_txt_file_paths(training_path); // Gather all .txt file paths
        if (all_paths.empty()) {
            std::cerr << "No .txt files found in training directory." << std::endl;
            return 1;
        }

        auto [train_paths, val_paths] = split_train_val_paths(all_paths, 0.01f); // 1% for validation
        std::cout << "Training files: " << train_paths.size() << ", Validation files: " << val_paths.size() << std::endl;
        TrainingConfig config;  
        auto model = std::make_unique<dnn::Gpt2<float>>(tokenizer, 50257, 1024, 12, 12, 768, 3072, 1e-4f, 0.9f, 0.98f, 1e-8f, true);
        train(model.get(), train_paths, val_paths, config); // Start training
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

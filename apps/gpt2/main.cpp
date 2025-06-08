#include <dnn/core/cuda.cuh>
#include <dnn/models/gpt2.cuh>
#include <dnn/tokens/tokenizer.cuh>
#include <dnn/tokens/vocab_loader.cuh>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>

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
    std::cerr << "    +-- vocab.txt   (Vocabulary file)" << std::endl;
    std::cerr << "    +-- training/" << std::endl;
    std::cerr << "        +-- *.txt   (Text files for training)" << std::endl;
    std::cerr << "\nText files should contain multiple paragraphs separated by blank lines.\n\n";
}

std::vector<std::string> load_text_files(const std::filesystem::path& root_path) {
    std::vector<std::string> documents;
    std::filesystem::path text_dir = root_path / "text";
    
    if (!std::filesystem::exists(text_dir)) {
        throw std::runtime_error("Text directory not found: " + text_dir.string());
    }

    for (const auto& entry : std::filesystem::directory_iterator(text_dir)) {
        if (entry.path().extension() == ".txt") {
            std::ifstream file(entry.path());
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file: " + entry.path().string());
            }

            std::string line;
            std::string current_doc;
            while (std::getline(file, line)) {
                if (line.empty()) {
                    if (!current_doc.empty()) {
                        documents.push_back(current_doc);
                        current_doc.clear();
                    }
                } else {
                    if (!current_doc.empty()) {
                        current_doc += "\n";
                    }
                    current_doc += line;
                }
            }
            if (!current_doc.empty()) {
                documents.push_back(current_doc);
            }
        }
    }

    if (documents.empty()) {
        throw std::runtime_error("No text documents found in: " + text_dir.string());
    }

    return documents;
}

struct TrainingConfig {
    int max_seq_len = 1024;      // Kept at 1024 for full context
    int batch_size = 4;          // Reduced for memory efficiency
    int num_epochs = 3;          // Fewer epochs due to faster convergence
    float learning_rate = 5e-5;  // Lower learning rate for stability
    float warmup_steps = 4000;   // Longer warmup for better adaptation
    float max_grad_norm = 1.0f;  // Unchanged (good default)
    int stride = 256;            // Smaller stride for more training examples
};

float get_learning_rate(int step, const TrainingConfig& config) {
    if (step < config.warmup_steps) {
        return config.learning_rate * (step / config.warmup_steps);
    }
    return config.learning_rate * (1.0f - (step - config.warmup_steps) / 
           (config.num_epochs * config.warmup_steps));
}

void train(dnn::NaturalLanguageModel<float>* model,
           const std::vector<std::string>& documents,
           const TrainingConfig& config) {
    
    int total_steps = 0;
    float total_loss = 0.0f;
    
    for (int epoch = 0; epoch < config.num_epochs; ++epoch) {
        for (const auto& doc : documents) {
            // Tokenize the document
            auto tokens = model->tokenizer()->encode(doc, true);
            
            // Create sliding windows
            for (size_t i = 0; i < tokens.size() - config.max_seq_len; i += config.stride) {
                // Create input sequence
                std::vector<int> input_tokens(tokens.begin() + i, 
                                            tokens.begin() + i + config.max_seq_len);
                
                // Create target sequence (shifted by 1)
                std::vector<int> target_tokens(tokens.begin() + i + 1, 
                                             tokens.begin() + i + config.max_seq_len + 1);
                
                // Create attention mask
                std::vector<float> attention_mask(config.max_seq_len, 1.0f);
                
                // Update learning rate
                float lr = get_learning_rate(total_steps, config);
                model->set_learning_rate(lr);
                
                // Forward pass
                float loss = model->train_step(input_tokens, target_tokens, attention_mask);
                
                // Gradient clipping
                model->clip_gradients(config.max_grad_norm);
                
                total_loss += loss;
                total_steps++;
                
                if (total_steps % 100 == 0) {
                    float avg_loss = total_loss / 100;
                    std::cout << "[Epoch " << (epoch + 1) << "/" << config.num_epochs
                              << "] Step " << total_steps 
                              << " Loss: " << avg_loss
                              << " LR: " << lr << std::endl;
                    total_loss = 0.0f;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        dnn::Cuda cuda;
        cuda.dump_info();

        if (argc != 2) {
            print_usage(argv[0]);
            return 1;
        }

        std::filesystem::path datasets_root = argv[1];
        std::filesystem::path vocab_path = datasets_root / "vocab.txt";
        std::filesystem::path training_path = datasets_root / "training";

        if (!std::filesystem::exists(vocab_path) || !std::filesystem::exists(training_path)) {
            std::cerr << "Missing vocab.txt or training/ directory." << std::endl;
            print_usage(argv[0]);
            return 1;
        }

        auto vocab_loader = std::make_shared<dnn::VocabLoader>();
        vocab_loader->load_from_file(vocab_path.string());

        auto tokenizer = std::make_shared<dnn::Tokenizer>(vocab_loader);
        tokenizer->set_special_tokens(vocab_loader->token_to_id("<BOS>"),
                                    vocab_loader->token_to_id("<EOS>"));

        auto documents = load_text_files(training_path);
        std::cout << "Loaded " << documents.size() << " documents.\n";

        TrainingConfig config;
        auto model = std::make_unique<dnn::Gpt2<float>>(tokenizer);
        
        train(model.get(), documents, config);

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#include <dnn/core/cuda.cuh>
#include <dnn/rag/rag_model.cuh>
#include <dnn/rag/document_store.cuh>
#include <dnn/models/gpt2.cuh>
#include <dnn/tokens/bpe_tokenizer.cuh>
#include <dnn/tokens/vocab_loader.cuh>
#include <dnn/utils/text_processing.cuh>
#include <dnn/utils/common.cuh>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <stdexcept>
#include <iomanip>

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
    std::cerr << "|                        RAG Demo                          |" << std::endl;
    std::cerr << "+----------------------------------------------------------+" << std::endl;
    std::cerr << "\nUsage: " << name << " <datasets_root_path>" << std::endl;
    std::cerr << "Example: " << name << " ../datasets" << std::endl;
    std::cerr << "\nExpected dataset structure:" << std::endl;
    std::cerr << "<datasets_root_path>/" << std::endl;
    std::cerr << "    +-- NLP-LLM/" << std::endl;
    std::cerr << "    |   +-- vocab.json   (Vocabulary file)" << std::endl;
    std::cerr << "    |   +-- knowledge_base/" << std::endl;
    std::cerr << "    |       +-- *.txt   (Knowledge base documents)" << std::endl;
    std::cerr << "\nThis demo shows RAG (Retrieval-Augmented Generation) capabilities.\n\n";
}

// Load knowledge base documents from directory
std::vector<std::pair<std::string, std::string>> load_knowledge_base(const std::filesystem::path& kb_path) {
    std::vector<std::pair<std::string, std::string>> documents;
    
    if (!std::filesystem::exists(kb_path)) {
        std::cerr << "Knowledge base directory not found: " << kb_path << std::endl;
        return documents;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(kb_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::ifstream file(entry.path());
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                std::string content = buffer.str();
                
                if (!content.empty()) {
                    // Clean the content
                    content = dnn::utils::TextProcessor::preprocess_for_rag(content);
                    
                    // Use filename as document ID
                    std::string doc_id = entry.path().stem().string();
                    documents.emplace_back(doc_id, content);
                    
                    std::cout << "Loaded document: " << doc_id << " (" << content.length() << " chars)" << std::endl;
                }
            }
        }
    }
    
    return documents;
}

// Interactive RAG demo
void run_rag_demo(std::unique_ptr<dnn::RAGModel<float>>& rag_model) {
    std::cout << "\n=== RAG Interactive Demo ===" << std::endl;
    std::cout << "Type your questions (or 'quit' to exit):" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    std::string query;
    while (true) {
        std::cout << "\nQuestion: ";
        std::getline(std::cin, query);
        
        if (query == "quit" || query == "exit") {
            break;
        }
        
        if (query.empty()) {
            continue;
        }
        
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Generate response using RAG
            std::string response = rag_model->generate(query, 150);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "\nAnswer: " << response << std::endl;
            std::cout << "\n[Generated in " << duration.count() << "ms]" << std::endl;
            
            // Show retrieved documents
            std::vector<int> query_tokens = rag_model->get_tokenizer()->encode(query, true);
            dnn::tensor<int> query_tensor({1, static_cast<int>(query_tokens.size())});
            query_tensor.upload(query_tokens.data());

            // Convert to model's data type (one-hot encoding)
            auto query_float = dnn::utils::to_one_hot<float>(query_tensor, rag_model->get_tokenizer()->vocab_size());

            auto retrieved = rag_model->get_document_store()->search_similar(
                rag_model->get_embedding_model()->forward(query_float).clone(), 3, 0.0f
            );
            
            if (!retrieved.empty()) {
                std::cout << "\nRetrieved documents:" << std::endl;
                for (size_t i = 0; i < retrieved.size(); ++i) {
                    const auto& [doc, score] = retrieved[i];
                    std::cout << "  " << (i + 1) << ". " << doc->id 
                              << " (similarity: " << std::fixed << std::setprecision(3) << score << ")" << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error generating response: " << e.what() << std::endl;
        }
    }
}

// Demo training with RAG
void demo_rag_training(std::unique_ptr<dnn::RAGModel<float>>& rag_model) {
    std::cout << "\n=== RAG Training Demo ===" << std::endl;
    
    // Sample training data
    std::vector<std::pair<std::string, std::string>> training_data = {
        {"What is DeepFuse?", "DeepFuse is a CUDA-based deep learning framework designed for scalable transformer training on consumer GPUs."},
        {"How does RAG work?", "RAG (Retrieval-Augmented Generation) combines document retrieval with text generation to provide more accurate and contextual responses."},
        {"What are the benefits of RAG?", "RAG provides more accurate answers by retrieving relevant information from a knowledge base before generating responses."}
    };
    
    std::cout << "Training on " << training_data.size() << " question-answer pairs..." << std::endl;
    
    for (int epoch = 0; epoch < 3; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/3" << std::endl;
        
        for (const auto& [question, answer] : training_data) {
            try {
                rag_model->train_step(question, answer);
                std::cout << "  Trained on: " << question << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "  Training error: " << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "Training completed!" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // Initialize CUDA
        dnn::Cuda cuda;
        cuda.dump_info();
        
        if (argc != 2) {
            print_usage(argv[0]);
            return 1;
        }
        
        std::filesystem::path datasets_root = argv[1];
        std::filesystem::path vocab_path = datasets_root / "NLP-LLM/vocab.json";
        std::filesystem::path kb_path = datasets_root / "NLP-LLM/knowledge_base";
        
        if (!std::filesystem::exists(vocab_path)) {
            std::cerr << "Vocabulary file not found: " << vocab_path << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        
        // Load vocabulary and tokenizer
        std::cout << "Loading vocabulary..." << std::endl;
        auto vocab_loader = std::make_shared<dnn::VocabLoader>();
        vocab_loader->load_from_file(vocab_path.string());
        
        auto tokenizer = std::make_shared<dnn::BpeTokenizer>(vocab_loader);
        std::cout << "Vocabulary loaded: " << vocab_loader->size() << " tokens" << std::endl;
        
        // Create document store
        std::cout << "Creating document store..." << std::endl;
        auto doc_store = std::make_shared<dnn::DocumentStore>(tokenizer, 768, 512);
        
        // Load knowledge base
        std::cout << "Loading knowledge base..." << std::endl;
        auto documents = load_knowledge_base(kb_path);
        
        if (documents.empty()) {
            std::cout << "No documents found in knowledge base. Creating sample documents..." << std::endl;
            
            // Create sample documents
            documents = {
                {"deepfuse_intro", "DeepFuse is a high-performance C++17/CUDA deep learning framework designed to enable large-scale transformer training on consumer GPUs. By implementing innovative memory management techniques and layer-serialized execution, DeepFuse breaks the VRAM ceiling and enables billion-parameter models on hardware previously limited to much smaller networks."},
                {"rag_explanation", "RAG (Retrieval-Augmented Generation) is a technique that combines document retrieval with text generation. It works by first retrieving relevant documents from a knowledge base based on the input query, then using those documents as context to generate more accurate and informative responses."},
                {"cuda_benefits", "CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that enables dramatic increases in computing performance by harnessing the power of the graphics processing unit (GPU). CUDA allows developers to use C++ to program GPUs for general-purpose computing tasks."},
                {"transformer_architecture", "The Transformer architecture is a neural network architecture based on the attention mechanism. It was introduced in the paper 'Attention Is All You Need' and has become the foundation for many state-of-the-art natural language processing models, including GPT, BERT, and T5."}
            };
        }
        
        // Add documents to store
        doc_store->add_documents(documents);
        std::cout << "Added " << doc_store->size() << " documents to knowledge base" << std::endl;
        
        // Create models
        std::cout << "Creating models..." << std::endl;
        auto generator_model = std::make_unique<dnn::Gpt2<float>>(
            tokenizer, 50257, 1024, 6, 6, 384, 1536, 1e-4f, 0.9f, 0.98f, 1e-8f, true
        );
        
        auto embedding_model = std::make_unique<dnn::Gpt2<float>>(
            tokenizer, 50257, 512, 4, 4, 256, 1024, 1e-4f, 0.9f, 0.98f, 1e-8f, false
        );
        
        // Create RAG model
        std::cout << "Creating RAG model..." << std::endl;
        auto rag_model = std::make_unique<dnn::RAGModel<float>>(
            std::move(generator_model),
            std::move(embedding_model),
            doc_store,
            tokenizer,
            1024,  // max context length
            3,     // retrieval top-k
            true   // training enabled
        );
        
        // Update embeddings
        std::cout << "Computing document embeddings..." << std::endl;
        rag_model->update_document_embeddings();
        std::cout << "Embeddings computed for " << doc_store->size() << " documents" << std::endl;
        
        // Demo training
        demo_rag_training(rag_model);
        
        // Optional: Demonstrate end-to-end training
        if (false) {  // Set to true to enable training demo
            std::cout << "\n=== Training Demo ===" << std::endl;

            // Create sample training data
            std::vector<std::pair<std::string, std::string>> train_data = {
                {"What is DeepFuse?", "DeepFuse is a CUDA-based deep learning framework for scalable transformer training on consumer GPUs."},
                {"How does RAG work?", "RAG combines retrieval from a knowledge base with generative language modeling."},
                {"What are the benefits?", "RAG provides more accurate and contextual responses by retrieving relevant information."}
            };

            std::vector<std::pair<std::string, std::string>> val_data = {
                {"What is CUDA?", "CUDA is NVIDIA's parallel computing platform for GPU acceleration."}
            };

            // Train the model
            rag_model->train_with_text_data(train_data, val_data, 5, 1, true);
        }

        // Interactive demo
        run_rag_demo(rag_model);

        std::cout << "\nRAG demo completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

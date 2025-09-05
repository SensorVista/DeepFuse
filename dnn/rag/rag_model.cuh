#pragma once

#include "../models/training_model.cuh"
#include "retriever.cuh"
#include "../utils/text_processing.cuh"
#include <memory>
#include <string>
#include <vector>

namespace dnn {

// Forward declarations
class DocumentStore;

// RAG orchestrator combining retrieval and generation. Manages knowledge base, retriever layer,
// and generator model for context-aware text generation with real-time document retrieval.
template<typename T>
class RAGModel : public dnn::TrainingModel<T> {
public:
    RAGModel(std::shared_ptr<dnn::TrainingModel<T>> generator_model,
             std::shared_ptr<dnn::TrainingModel<T>> embedding_model,
             std::shared_ptr<DocumentStore> document_store,
             std::shared_ptr<dnn::Tokenizer> tokenizer,
             int max_context_length = 1024,
             int retrieval_top_k = 5,
             bool training_enabled = false);
    
    // Training interface
    void train_step(const tensor<T>& input, const tensor<T>& target) override;  // Joint training: retriever + generator
    void train_step(const std::string& query, const std::string& target_answer);

    // End-to-end training pipeline
    void train(const std::vector<std::pair<tensor<T>, tensor<T>>>& train_data,
               const std::vector<std::pair<tensor<T>, tensor<T>>>& val_data,
               int epochs, int batch_size = 1, bool verbose = true);
    void train_with_text_data(const std::vector<std::pair<std::string, std::string>>& train_queries,
                             const std::vector<std::pair<std::string, std::string>>& val_queries,
                             int epochs, int batch_size = 1, bool verbose = true);

    // Training utilities
    T validate(const std::vector<std::pair<tensor<T>, tensor<T>>>& val_data);
    void save_checkpoint(const std::string& path, int epoch, T loss);
    bool load_checkpoint(const std::string& path);
    void set_learning_rate(T lr);
    T get_learning_rate() const;
    
    // Optimizer setup
    void set_optimizer(std::unique_ptr<dnn::Optimizer<T>> optimizer) override;
    void set_retriever_optimizer(std::unique_ptr<dnn::Optimizer<T>> optimizer);
    void set_generator_optimizer(std::unique_ptr<dnn::Optimizer<T>> optimizer);
    
    // Inference interface
    tensor<T> forward(const tensor<T>& input) override;  // Forward pass through retriever + generator
    std::string generate(const std::string& query, int max_length = 100);  // RAG generation: retrieve then generate

    // RAG-specific methods
    void add_knowledge_base(const std::vector<std::pair<std::string, std::string>>& documents);  // Add documents to knowledge base
    void update_embeddings();  // Compute embeddings for all documents
    void update_document_embeddings();  // Wire embedding model and update embeddings

    // Context management
    std::string build_context(const std::string& query, const std::vector<std::pair<Document*, float>>& retrieved);  // Format retrieved docs as context

    // Autoregressive generation helpers
    int sample_next_token_argmax(const tensor<T>& logits, int vocab_size);
    int sample_next_token_temperature(const tensor<T>& logits, int vocab_size, float temperature);
    int sample_next_token_topk(const tensor<T>& logits, int vocab_size, float temperature, int top_k);
    int sample_next_token_nucleus(const tensor<T>& logits, int vocab_size, float temperature, float top_p);
    tensor<T> append_token_to_input(tensor<T> current_input, int new_token);

    // Sampling utilities
    std::vector<float> softmax(const std::vector<float>& logits);
    int sample_from_distribution(const std::vector<float>& probabilities);
    void apply_repetition_penalty(tensor<T>& logits, const std::vector<int>& generated_tokens,
                                std::unordered_map<int, int>& token_counts, float penalty);
    
    // Save/load
    void save(const std::string& path) const override;
    static std::unique_ptr<RAGModel<T>> load(const std::string& path, bool training_enabled);
    
    std::vector<dnn::BaseLayer*> layers() override;
    
    // Getters
    std::shared_ptr<DocumentStore> get_document_store() const { return document_store_; }
    std::shared_ptr<dnn::TrainingModel<T>> get_generator_model() const { return generator_model_; }
    std::shared_ptr<dnn::TrainingModel<T>> get_embedding_model() const { return embedding_model_; }
    std::shared_ptr<dnn::Tokenizer> get_tokenizer() const { return tokenizer_; }

private:
    std::shared_ptr<dnn::TrainingModel<T>> generator_model_;
    std::shared_ptr<dnn::TrainingModel<T>> embedding_model_;
    std::shared_ptr<DocumentStore> document_store_;
    std::shared_ptr<dnn::Tokenizer> tokenizer_;

    std::unique_ptr<RetrieverLayer<T>> retriever_layer_;

    // Optimizers for joint training
    std::unique_ptr<dnn::Optimizer<T>> retriever_optimizer_;
    std::unique_ptr<dnn::Optimizer<T>> generator_optimizer_;

    int max_context_length_;
    int retrieval_top_k_;

    // Context formatting
    std::string format_context_template_;

    // Training state
    int current_epoch_;
    T best_val_loss_;
    T current_learning_rate_;
    std::vector<T> training_losses_;
    std::vector<T> validation_losses_;
    
    tensor<T> prepare_rag_input(const std::string& query, const std::string& context);
    std::vector<std::string> extract_retrieved_content(const std::vector<std::pair<Document*, float>>& retrieved);
};

} // namespace dnn

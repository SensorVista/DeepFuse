#pragma once

#include "../core/layer.cuh"
#include "../core/tensor.cuh"
#include "../models/training_model.cuh"
#include "document_store.cuh"
#include <memory>
#include <optional>

// Forward declarations
namespace dnn {
    class DocumentStore;
    struct Document;
}

namespace dnn {

// RAG retrieval component. Embeds queries and searches knowledge base for relevant documents.
// Performs similarity matching between query embeddings and document embeddings for context retrieval.
template<typename T>
class RetrieverLayer : public dnn::Layer<T> {
public:
    RetrieverLayer(std::shared_ptr<DocumentStore> doc_store,
                   std::shared_ptr<dnn::TrainingModel<T>> embedding_model,
                   int top_k = 5,
                   float similarity_threshold = 0.0f,
                   bool training_enabled = false);
    
    // Layer interface
    tensor<T> forward(const tensor<T>& input) override;  // Retrieve relevant documents for query embedding
    tensor<T> backward(const tensor<T>& grad_output) override;  // Backprop through retrieval (currently no-op)
    std::string name() const override { return "RetrieverLayer"; }

    // RAG-specific methods
    void set_document_store(std::shared_ptr<DocumentStore> doc_store);  // Connect to knowledge base
    void set_top_k(int top_k) { top_k_ = top_k; }  // Set number of documents to retrieve
    void set_similarity_threshold(float threshold) { similarity_threshold_ = threshold; }  // Minimum similarity threshold

    // Get retrieved documents for current query
    std::vector<std::pair<Document*, float>> get_last_retrieved() const;  // Access cached retrieval results

    // Perform real-time retrieval for a text query
    std::vector<std::pair<Document*, float>> retrieve_for_query(const std::string& query);  // RAG retrieval step
    
    // Save/load
    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

private:
    std::shared_ptr<DocumentStore> doc_store_;
    std::shared_ptr<dnn::TrainingModel<T>> embedding_model_;
    int top_k_;
    float similarity_threshold_;
    
    // Cache for last retrieval results
    mutable std::vector<std::pair<Document*, float>> last_retrieved_;
    mutable std::optional<tensor<T>> last_query_embedding_;
    
    tensor<T> compute_query_embedding(const tensor<T>& input);
    tensor<T> create_retrieval_output(const std::vector<std::pair<Document*, float>>& retrieved);
};

} // namespace dnn

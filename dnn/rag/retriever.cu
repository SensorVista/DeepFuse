#include "retriever.cuh"
#include "document_store.cuh"
#include "../utils/common.cuh"
#include "../core/tensor.cuh"
#include "../tokens/tokenizer.cuh"
#include <stdexcept>
#include <algorithm>

namespace dnn {

template<typename T>
RetrieverLayer<T>::RetrieverLayer(std::shared_ptr<DocumentStore> doc_store,
                                std::shared_ptr<dnn::TrainingModel<T>> embedding_model,
                                int top_k, float similarity_threshold, bool training_enabled)
    : Layer<T>(training_enabled), doc_store_(doc_store), embedding_model_(embedding_model),
      top_k_(top_k), similarity_threshold_(similarity_threshold) {
    
    if (!doc_store_) {
        throw std::invalid_argument("Document store cannot be null");
    }
    if (!embedding_model_) {
        throw std::invalid_argument("Embedding model cannot be null");
    }
    if (top_k <= 0) {
        throw std::invalid_argument("Top-k must be positive");
    }
}

template<typename T>
tensor<T> RetrieverLayer<T>::forward(const tensor<T>& input) {
    if (!doc_store_ || !embedding_model_) {
        throw std::runtime_error("RetrieverLayer not properly initialized");
    }

    // Compute query embedding
    tensor<T> query_embedding = compute_query_embedding(input);

    // Cache the query embedding (clone necessary for caching)
    last_query_embedding_ = query_embedding.clone();

    // Convert to float for similarity computation
    std::optional<tensor<float>> query_embedding_float_opt;
    if constexpr (std::is_same_v<T, float>) {
        // Same type - move to avoid copy
        query_embedding_float_opt = std::move(query_embedding);
    } else {
        // Different type - convert via buffer and initialize properly
        std::vector<T> query_data(query_embedding.size());
        std::vector<float> query_float_data(query_embedding.size());

        query_embedding.download(query_data.data());
        for (size_t i = 0; i < query_data.size(); ++i) {
            query_float_data[i] = static_cast<float>(query_data[i]);
        }

        // Create and initialize tensor in one step
        tensor<float> query_embedding_float({static_cast<int>(query_embedding.size())});
        query_embedding_float.upload(query_float_data.data());
        query_embedding_float_opt = std::move(query_embedding_float);
    }

    // Extract the tensor (now guaranteed to be initialized)
    tensor<float> query_embedding_float = std::move(*query_embedding_float_opt);

    // Search for similar documents
    last_retrieved_ = doc_store_->search_similar(query_embedding_float, top_k_, similarity_threshold_);

    // Create output tensor representing retrieved documents
    return create_retrieval_output(last_retrieved_);
}

template<typename T>
std::vector<std::pair<Document*, float>> RetrieverLayer<T>::retrieve_for_query(const std::string& query) {
    if (!doc_store_ || !embedding_model_) {
        throw std::runtime_error("RetrieverLayer not properly initialized");
    }

    if (query.empty()) {
        return {};
    }

    // Get tokenizer from document store
    auto tokenizer = doc_store_->get_tokenizer();
    if (!tokenizer) {
        throw std::runtime_error("No tokenizer available in document store");
    }

    // Tokenize query into tokens
    std::vector<int> query_tokens = tokenizer->encode(query, true);

    // Convert tokens to one-hot encoding for embedding model
    tensor<int> token_tensor({1, static_cast<int>(query_tokens.size())});
    token_tensor.upload(query_tokens.data());
    tensor<T> query_input = dnn::utils::to_one_hot<T>(token_tensor, tokenizer->vocab_size());

    // Generate query embedding using embedding model
    tensor<T> query_embedding = embedding_model_->forward(query_input);

    // Convert to float for similarity computation (following the same pattern as forward())
    std::optional<tensor<float>> query_embedding_float_opt;
    if constexpr (std::is_same_v<T, float>) {
        query_embedding_float_opt = std::move(query_embedding);
    } else {
        // Convert to float
        std::vector<T> embedding_data(query_embedding.size());
        std::vector<float> float_data(query_embedding.size());

        query_embedding.download(embedding_data.data());
        for (size_t i = 0; i < embedding_data.size(); ++i) {
            float_data[i] = static_cast<float>(embedding_data[i]);
        }

        tensor<float> query_embedding_float({static_cast<int>(query_embedding.size())});
        query_embedding_float.upload(float_data.data());
        query_embedding_float_opt = std::move(query_embedding_float);
    }

    // Extract the tensor
    tensor<float> query_embedding_float = std::move(*query_embedding_float_opt);

    // Search knowledge base for most similar documents
    return doc_store_->search_similar(query_embedding_float, top_k_, similarity_threshold_);
}

template<typename T>
tensor<T> RetrieverLayer<T>::backward(const tensor<T>& grad_output) {
    // For now, we don't backpropagate through retrieval
    // In a full implementation, we might want to learn better embeddings
    // Return zero gradient for the input
    tensor<T> grad_input(grad_output.shape());
    grad_input.zero();
    return grad_input;
}

template<typename T>
void RetrieverLayer<T>::set_document_store(std::shared_ptr<DocumentStore> doc_store) {
    if (!doc_store) {
        throw std::invalid_argument("Document store cannot be null");
    }
    doc_store_ = doc_store;
}

template<typename T>
std::vector<std::pair<Document*, float>> RetrieverLayer<T>::get_last_retrieved() const {
    return last_retrieved_;
}

template<typename T>
void RetrieverLayer<T>::save(std::ostream& out) const {
    // Save configuration
    out.write(reinterpret_cast<const char*>(&top_k_), sizeof(top_k_));
    out.write(reinterpret_cast<const char*>(&similarity_threshold_), sizeof(similarity_threshold_));
    
    // Note: We don't save the document store or embedding model here
    // as they should be saved separately and loaded independently
}

template<typename T>
void RetrieverLayer<T>::load(std::istream& in) {
    // Load configuration
    in.read(reinterpret_cast<char*>(&top_k_), sizeof(top_k_));
    in.read(reinterpret_cast<char*>(&similarity_threshold_), sizeof(similarity_threshold_));
}

template<typename T>
tensor<T> RetrieverLayer<T>::compute_query_embedding(const tensor<T>& input) {
    // Use the embedding model to compute query embedding
    // For now, we assume the input is already in the right format
    // In a full implementation, we might need to preprocess the input
    
    return embedding_model_->forward(input).clone();
}

template<typename T>
tensor<T> RetrieverLayer<T>::create_retrieval_output(const std::vector<std::pair<Document*, float>>& retrieved) {
    // Create a tensor that represents the retrieved documents
    // For simplicity, we'll create a tensor with shape [num_retrieved, embedding_dim]
    // containing the embeddings of retrieved documents
    
    if (retrieved.empty()) {
        // Return empty tensor if no documents retrieved
        return tensor<T>({0, doc_store_->embedding_dim()});
    }
    
    int num_retrieved = static_cast<int>(retrieved.size());
    int embedding_dim = doc_store_->embedding_dim();
    
    tensor<T> output({num_retrieved, embedding_dim});
    
    // Convert document embeddings to output tensor
    std::vector<T> output_data(num_retrieved * embedding_dim);
    
    for (int i = 0; i < num_retrieved; ++i) {
        const Document* doc = retrieved[i].first;
        if (!doc || doc->embedding->size() == 0) {
            continue;
        }
        
        // Convert float embedding to T
        std::vector<float> embedding_data(doc->embedding->size());
        doc->embedding->download(embedding_data.data());
        
        for (int j = 0; j < embedding_dim; ++j) {
            output_data[i * embedding_dim + j] = static_cast<T>(embedding_data[j]);
        }
    }
    
    output.upload(output_data.data());
    return output;
}

// Explicit template instantiations
template class RetrieverLayer<float>;
template class RetrieverLayer<__half>;

} // namespace dnn

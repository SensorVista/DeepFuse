#include "document_store.cuh"
#include "../core/tensor.cuh"
#include "../tokens/tokenizer.cuh"
#include "../models/training_model.cuh"
#include "../utils/common.cuh"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dnn {

DocumentStore::DocumentStore(std::shared_ptr<dnn::Tokenizer> tokenizer, 
                           int embedding_dim, int max_doc_length)
    : tokenizer_(tokenizer), embedding_dim_(embedding_dim), max_doc_length_(max_doc_length) {
    if (!tokenizer_) {
        throw std::invalid_argument("Tokenizer cannot be null");
    }
    if (embedding_dim <= 0) {
        throw std::invalid_argument("Embedding dimension must be positive");
    }
    if (max_doc_length <= 0) {
        throw std::invalid_argument("Max document length must be positive");
    }
}

template<typename T>
void DocumentStore::set_embedding_model(std::shared_ptr<dnn::TrainingModel<T>> model) {
    if constexpr (std::is_same_v<T, float>) {
        embed_f32_ = model;
    } else if constexpr (std::is_same_v<T, __half>) {
        embed_f16_ = model;
    } else {
        throw std::invalid_argument("Unsupported embedding model type");
    }
}

void DocumentStore::add_document(const std::string& id, const std::string& content,
                               const std::unordered_map<std::string, std::string>& metadata) {
    if (id.empty()) {
        throw std::invalid_argument("Document ID cannot be empty");
    }
    if (content.empty()) {
        throw std::invalid_argument("Document content cannot be empty");
    }
    
    Document doc(id, content);
    doc.metadata = metadata;
    tokenize_document(doc);
    
    documents_[id] = std::move(doc);
    index_dirty_ = true;
}

void DocumentStore::remove_document(const std::string& id) {
    auto it = documents_.find(id);
    if (it != documents_.end()) {
        documents_.erase(it);
        index_dirty_ = true;
    }
}

Document* DocumentStore::get_document(const std::string& id) {
    auto it = documents_.find(id);
    return (it != documents_.end()) ? &it->second : nullptr;
}

const Document* DocumentStore::get_document(const std::string& id) const {
    auto it = documents_.find(id);
    return (it != documents_.end()) ? &it->second : nullptr;
}

void DocumentStore::add_documents(const std::vector<std::pair<std::string, std::string>>& docs) {
    for (const auto& [id, content] : docs) {
        add_document(id, content);
    }
}

tensor<float> DocumentStore::encode_ids_with_float(const std::vector<int>& ids) {
    if (!embed_f32_) {
        throw std::runtime_error("Float embedding model not set");
    }

    // Convert token IDs to tensor
    tensor<int> input_ids({1, static_cast<int>(ids.size())});
    input_ids.upload(ids.data());

    // Convert to one-hot encoding
    tensor<float> input = dnn::utils::to_one_hot<float>(input_ids, tokenizer_->vocab_size());

    // Forward pass through embedding model
    tensor<float> embeddings = embed_f32_->forward(input);

    // Apply pooling based on output shape
    const auto& shape = embeddings.shape();
    if (shape.size() == 2 && shape[0] == static_cast<int>(ids.size())) {
        // [L, D] -> average pooling to [D]
        int L = shape[0], D = shape[1];
        std::vector<float> pooled(D, 0.0f), buf(embeddings.size());
        embeddings.download(buf.data());

        for (int t = 0; t < L; ++t) {
            const float* row = &buf[t * D];
            for (int j = 0; j < D; ++j) {
                pooled[j] += row[j];
            }
        }

        float invL = L > 0 ? 1.0f / static_cast<float>(L) : 1.0f;
        for (int j = 0; j < D; ++j) {
            pooled[j] *= invL;
        }

        tensor<float> result({D});
        result.upload(pooled.data());
        return result;
    } else if (shape.size() == 2 && shape[0] == 1) {
        // [1, D] -> flatten to [D]
        std::vector<float> buf(embeddings.size());
        embeddings.download(buf.data());
        tensor<float> result({shape[1]});
        result.upload(buf.data());
        return result;
    } else {
        // Return as-is if shape doesn't match expected patterns
        return embeddings;
    }
}

tensor<float> DocumentStore::encode_ids_with_half(const std::vector<int>& ids) {
    if (!embed_f16_) {
        throw std::runtime_error("Half embedding model not set");
    }

    // Convert token IDs to tensor
    tensor<int> input_ids({1, static_cast<int>(ids.size())});
    input_ids.upload(ids.data());

    // Convert to one-hot encoding as float first, then convert to half
    tensor<float> input_float = dnn::utils::to_one_hot<float>(input_ids, tokenizer_->vocab_size());

    // Convert to half precision for the model
    std::vector<int> shape = input_float.shape();
    tensor<__half> input_half(shape);
    std::vector<float> float_data(input_float.size());
    std::vector<__half> half_data(input_float.size());
    input_float.download(float_data.data());
    for (size_t i = 0; i < float_data.size(); ++i) {
        half_data[i] = __half(float_data[i]);
    }
    input_half.upload(half_data.data());

    // Forward pass through embedding model
    tensor<__half> embeddings_half = embed_f16_->forward(input_half);

    // Convert back to float for consistency
    tensor<float> embeddings(embeddings_half.shape());
    std::vector<__half> half_output_data(embeddings_half.size());
    std::vector<float> float_output_data(embeddings_half.size());

    embeddings_half.download(half_output_data.data());
    for (size_t i = 0; i < half_output_data.size(); ++i) {
        float_output_data[i] = static_cast<float>(half_output_data[i]);
    }
    embeddings.upload(float_output_data.data());

    // Apply pooling based on output shape
    const auto& output_shape = embeddings.shape();
    if (output_shape.size() == 2 && output_shape[0] == static_cast<int>(ids.size())) {
        // [L, D] -> average pooling to [D]
        int L = output_shape[0], D = output_shape[1];
        std::vector<float> pooled(D, 0.0f), output_buf(embeddings.size());
        embeddings.download(output_buf.data());

        for (int t = 0; t < L; ++t) {
            const float* row = &output_buf[t * D];
            for (int j = 0; j < D; ++j) {
                pooled[j] += row[j];
            }
        }

        float invL = L > 0 ? 1.0f / static_cast<float>(L) : 1.0f;
        for (int j = 0; j < D; ++j) {
            pooled[j] *= invL;
        }

        tensor<float> result({D});
        result.upload(pooled.data());
        return result;
    } else if (output_shape.size() == 2 && output_shape[0] == 1) {
        // [1, D] -> flatten to [D]
        std::vector<float> output_data(embeddings.size());
        embeddings.download(output_data.data());
        tensor<float> result({output_shape[1]});
        result.upload(output_data.data());
        return result;
    } else {
        // Return as-is if shape doesn't match expected patterns
        return embeddings;
    }
}

void DocumentStore::l2_normalize_inplace(tensor<float>& v) {
    std::vector<float> data(v.size());
    v.download(data.data());

    double sum_squares = 0.0;
    for (float x : data) {
        sum_squares += static_cast<double>(x) * static_cast<double>(x);
    }

    if (sum_squares <= 0.0) {
        return; // Already zero vector
    }

    float inv_norm = 1.0f / std::sqrt(static_cast<float>(sum_squares));
    for (auto& x : data) {
        x *= inv_norm;
    }

    v.upload(data.data());
}

void DocumentStore::update_embeddings_internal() {
    if (!embed_f32_ && !embed_f16_) {
        throw std::runtime_error("No embedding model set");
    }

    for (auto& kv : documents_) {
        auto& doc = kv.second;
        if (doc.token_ids.empty()) {
            continue;
        }

        // Compute embedding using appropriate model
        tensor<float> embedding = embed_f32_
            ? encode_ids_with_float(doc.token_ids)
            : encode_ids_with_half(doc.token_ids);

        // Ensure correct dimension
        if (static_cast<int>(embedding.size()) != embedding_dim_) {
            std::vector<float> buf(embedding.size());
            embedding.download(buf.data());
            buf.resize(embedding_dim_, 0.0f);
            tensor<float> fixed({embedding_dim_});
            fixed.upload(buf.data());
            embedding = std::move(fixed);
        }

        // L2 normalize
        l2_normalize_inplace(embedding);

        // Store embedding (move instead of clone to avoid unnecessary copy)
        doc.set_embedding(std::make_unique<tensor<float>>(std::move(embedding)));
    }

    index_dirty_ = true;
}

std::vector<std::pair<Document*, float>> DocumentStore::search_similar(
    const tensor<float>& query_embedding, 
    int top_k, 
    float threshold) const {
    
    if (documents_.empty()) {
        return {};
    }
    
    rebuild_index();
    
    std::vector<std::pair<Document*, float>> results;
    results.reserve(documents_.size());
    
    for (const auto& [doc_id, embedding_ptr] : embedding_index_) {
        Document* doc = const_cast<Document*>(get_document(doc_id));
        if (!doc || !doc->has_embedding() || !embedding_ptr) {
            continue;
        }
        
        float similarity = compute_similarity(query_embedding, *embedding_ptr);
        if (similarity >= threshold) {
            results.emplace_back(doc, similarity);
        }
    }
    
    // Sort by similarity (descending)
    std::sort(results.begin(), results.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return top_k results
    if (top_k > 0 && results.size() > static_cast<size_t>(top_k)) {
        results.resize(top_k);
    }
    
    return results;
}

void DocumentStore::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for saving: " + path);
    }
    
    // Save metadata
    size_t num_docs = documents_.size();
    out.write(reinterpret_cast<const char*>(&num_docs), sizeof(num_docs));
    out.write(reinterpret_cast<const char*>(&embedding_dim_), sizeof(embedding_dim_));
    out.write(reinterpret_cast<const char*>(&max_doc_length_), sizeof(max_doc_length_));
    
    // Save documents
    for (const auto& [id, doc] : documents_) {
        // Save document ID
        size_t id_len = id.length();
        out.write(reinterpret_cast<const char*>(&id_len), sizeof(id_len));
        out.write(id.c_str(), id_len);
        
        // Save content
        size_t content_len = doc.content.length();
        out.write(reinterpret_cast<const char*>(&content_len), sizeof(content_len));
        out.write(doc.content.c_str(), content_len);
        
        // Save token IDs
        size_t token_count = doc.token_ids.size();
        out.write(reinterpret_cast<const char*>(&token_count), sizeof(token_count));
        if (token_count > 0) {
            out.write(reinterpret_cast<const char*>(doc.token_ids.data()), 
                     token_count * sizeof(int));
        }
        
        // Save embedding if present
        bool has_embedding = doc.has_embedding();
        out.write(reinterpret_cast<const char*>(&has_embedding), sizeof(has_embedding));
        if (has_embedding) {
            doc.get_embedding().save(out);
        }
        
        // Save metadata
        size_t metadata_size = doc.metadata.size();
        out.write(reinterpret_cast<const char*>(&metadata_size), sizeof(metadata_size));
        for (const auto& [key, value] : doc.metadata) {
            size_t key_len = key.length();
            out.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
            out.write(key.c_str(), key_len);
            
            size_t value_len = value.length();
            out.write(reinterpret_cast<const char*>(&value_len), sizeof(value_len));
            out.write(value.c_str(), value_len);
        }
    }
}

void DocumentStore::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for loading: " + path);
    }
    
    // Clear existing documents
    documents_.clear();
    index_dirty_ = true;
    
    // Load metadata
    size_t num_docs;
    in.read(reinterpret_cast<char*>(&num_docs), sizeof(num_docs));
    in.read(reinterpret_cast<char*>(&embedding_dim_), sizeof(embedding_dim_));
    in.read(reinterpret_cast<char*>(&max_doc_length_), sizeof(max_doc_length_));
    
    // Load documents
    for (size_t i = 0; i < num_docs; ++i) {
        Document doc;  // Now this should work with proper default constructor
        
        // Load document ID
        size_t id_len;
        in.read(reinterpret_cast<char*>(&id_len), sizeof(id_len));
        doc.id.resize(id_len);
        in.read(&doc.id[0], id_len);
        
        // Load content
        size_t content_len;
        in.read(reinterpret_cast<char*>(&content_len), sizeof(content_len));
        doc.content.resize(content_len);
        in.read(&doc.content[0], content_len);
        
        // Load token IDs
        size_t token_count;
        in.read(reinterpret_cast<char*>(&token_count), sizeof(token_count));
        doc.token_ids.resize(token_count);
        if (token_count > 0) {
            in.read(reinterpret_cast<char*>(doc.token_ids.data()), 
                   token_count * sizeof(int));
        }
        
        // Load embedding if present
        bool has_embedding;
        in.read(reinterpret_cast<char*>(&has_embedding), sizeof(has_embedding));
        if (has_embedding) {
            // Create tensor with dummy shape first, then load will resize it
            auto embedding = std::make_unique<tensor<float>>(std::vector<int>{1});
            embedding->load(in);
            doc.set_embedding(std::move(embedding));
        }
        
        // Load metadata
        size_t metadata_size;
        in.read(reinterpret_cast<char*>(&metadata_size), sizeof(metadata_size));
        for (size_t j = 0; j < metadata_size; ++j) {
            size_t key_len;
            in.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
            std::string key(key_len, '\0');
            in.read(&key[0], key_len);
            
            size_t value_len;
            in.read(reinterpret_cast<char*>(&value_len), sizeof(value_len));
            std::string value(value_len, '\0');
            in.read(&value[0], value_len);
            
            doc.metadata[key] = value;
        }
        
        documents_[doc.id] = std::move(doc);
    }
}

void DocumentStore::rebuild_index() const {
    if (!index_dirty_) {
        return;
    }
    
    embedding_index_.clear();
    embedding_index_.reserve(documents_.size());
    
    for (const auto& [id, doc] : documents_) {
        if (doc.has_embedding()) {
            // Clone the embedding tensor for the index
            embedding_index_.emplace_back(id, std::make_unique<tensor<float>>(doc.get_embedding().clone()));
        }
    }
    
    index_dirty_ = false;
}

float DocumentStore::compute_similarity(const tensor<float>& a, const tensor<float>& b) const {
    if (a.size() != b.size()) {
        return 0.0f;
    }
    
    // Simple cosine similarity implementation
    std::vector<float> a_data(a.size());
    std::vector<float> b_data(b.size());
    a.download(a_data.data());
    b.download(b_data.data());
    
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < a_data.size(); ++i) {
        dot_product += a_data[i] * b_data[i];
        norm_a += a_data[i] * a_data[i];
        norm_b += b_data[i] * b_data[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }
    
    return dot_product / (norm_a * norm_b);
}

void DocumentStore::tokenize_document(Document& doc) {
    try {
        doc.token_ids = tokenizer_->encode(doc.content, false);
        
        // Truncate if too long
        if (static_cast<int>(doc.token_ids.size()) > max_doc_length_) {
            doc.token_ids.resize(max_doc_length_);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to tokenize document '" + doc.id + "': " + e.what());
    }
}


// Template instantiations for set_embedding_model
template void DocumentStore::set_embedding_model(std::shared_ptr<dnn::TrainingModel<float>> model);
template void DocumentStore::set_embedding_model(std::shared_ptr<dnn::TrainingModel<__half>> model);

} // namespace dnn

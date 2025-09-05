#pragma once

#include "../core/tensor.cuh"
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>

namespace dnn::utils {

// Vector similarity search utilities for RAG retrieval. Computes cosine similarity, euclidean distance,
// and performs efficient nearest neighbor search for document ranking and context retrieval.
template<typename T>
class VectorSearch {
public:
    // Similarity metrics for document ranking
    static float cosine_similarity(const tensor<T>& a, const tensor<T>& b);  // Cosine similarity for embeddings
    static float euclidean_distance(const tensor<T>& a, const tensor<T>& b);
    static float dot_product(const tensor<T>& a, const tensor<T>& b);
    
    // Batch similarity computation
    static tensor<T> batch_cosine_similarity(const tensor<T>& queries,  // [N, D]
                                           const tensor<T>& documents); // [M, D]
    
    // Top-K search
    template<typename ScoreType>
    static std::vector<std::pair<int, ScoreType>> top_k_similar(
        const tensor<T>& query,
        const std::vector<tensor<T>>& candidates,
        int k,
        float (*similarity_func)(const tensor<T>&, const tensor<T>&) = cosine_similarity);
    
    // Index building for faster search
    static void build_embedding_index(const std::vector<tensor<T>>& embeddings);
    static std::vector<std::pair<int, float>> search_index(
        const tensor<T>& query, int top_k);
    
    // Utility functions
    static tensor<T> normalize_vectors(const tensor<T>& vectors);
    static float compute_norm(const tensor<T>& vector);

private:
    static std::vector<tensor<T>> embedding_index_;
    static bool index_built_;
    
    static void build_index_internal(const std::vector<tensor<T>>& embeddings);
};

} // namespace dnn::utils

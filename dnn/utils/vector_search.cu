#include "vector_search.cuh"
#include "common.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace dnn::utils {

// Static member definitions
template<typename T>
std::vector<tensor<T>> VectorSearch<T>::embedding_index_;

template<typename T>
bool VectorSearch<T>::index_built_ = false;

// CUDA kernel for cosine similarity computation
template<typename T>
__global__ void cosine_similarity_kernel(const T* a, const T* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        float val_a = static_cast<float>(a[i]);
        float val_b = static_cast<float>(b[i]);
        dot += val_a * val_b;
        norm_a += val_a * val_a;
        norm_b += val_b * val_b;
    }
    
    if (norm_a > 0.0f && norm_b > 0.0f) {
        *result = dot / (sqrtf(norm_a) * sqrtf(norm_b));
    } else {
        *result = 0.0f;
    }
}

// CUDA kernel for batch cosine similarity
template<typename T>
__global__ void batch_cosine_similarity_kernel(
    const T* queries,     // [N, D]
    const T* documents,   // [M, D]
    float* results,       // [N, M]
    int N, int M, int D) {
    
    int n = blockIdx.x;
    int m = blockIdx.y;
    
    if (n >= N || m >= M) return;
    
    float dot = 0.0f;
    float norm_q = 0.0f;
    float norm_d = 0.0f;
    
    for (int d = 0; d < D; ++d) {
        float q_val = static_cast<float>(queries[n * D + d]);
        float d_val = static_cast<float>(documents[m * D + d]);
        dot += q_val * d_val;
        norm_q += q_val * q_val;
        norm_d += d_val * d_val;
    }
    
    if (norm_q > 0.0f && norm_d > 0.0f) {
        results[n * M + m] = dot / (sqrtf(norm_q) * sqrtf(norm_d));
    } else {
        results[n * M + m] = 0.0f;
    }
}

template<typename T>
float VectorSearch<T>::cosine_similarity(const tensor<T>& a, const tensor<T>& b) {
    if (a.size() != b.size()) {
        return 0.0f;
    }
    
    // Allocate result on device
    float* d_result;
    CHECK_CUDA_EX(cudaMalloc(&d_result, sizeof(float)));
    
    // Launch kernel
    const int block_size = 256;
    const int grid_size = 1;
    
    cosine_similarity_kernel<<<grid_size, block_size>>>(
        a.data(), b.data(), d_result, a.size()
    );
    THROW_CUDA_EX();
    
    // Copy result back
    float result;
    CHECK_CUDA_EX(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_EX(cudaFree(d_result));
    
    return result;
}

template<typename T>
float VectorSearch<T>::euclidean_distance(const tensor<T>& a, const tensor<T>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::infinity();
    }
    
    // Download data to host for computation
    std::vector<T> a_data(a.size());
    std::vector<T> b_data(b.size());
    a.download(a_data.data());
    b.download(b_data.data());
    
    float distance = 0.0f;
    for (size_t i = 0; i < a_data.size(); ++i) {
        float diff = static_cast<float>(a_data[i]) - static_cast<float>(b_data[i]);
        distance += diff * diff;
    }
    
    return std::sqrt(distance);
}

template<typename T>
float VectorSearch<T>::dot_product(const tensor<T>& a, const tensor<T>& b) {
    if (a.size() != b.size()) {
        return 0.0f;
    }
    
    // Download data to host for computation
    std::vector<T> a_data(a.size());
    std::vector<T> b_data(b.size());
    a.download(a_data.data());
    b.download(b_data.data());
    
    float dot = 0.0f;
    for (size_t i = 0; i < a_data.size(); ++i) {
        dot += static_cast<float>(a_data[i]) * static_cast<float>(b_data[i]);
    }
    
    return dot;
}

template<typename T>
tensor<T> VectorSearch<T>::batch_cosine_similarity(const tensor<T>& queries, const tensor<T>& documents) {
    const auto& q_shape = queries.shape();
    const auto& d_shape = documents.shape();
    
    if (q_shape.size() != 2 || d_shape.size() != 2) {
        throw std::invalid_argument("Input tensors must be 2D");
    }
    
    int N = q_shape[0];  // Number of queries
    int M = d_shape[0];  // Number of documents
    int D = q_shape[1];  // Embedding dimension
    
    if (d_shape[1] != D) {
        throw std::invalid_argument("Embedding dimensions must match");
    }
    
    // Create result tensor
    tensor<T> results({N, M});
    
    // Allocate device memory for results
    float* d_results;
    CHECK_CUDA_EX(cudaMalloc(&d_results, N * M * sizeof(float)));
    
    // Launch kernel
    dim3 block_size(1, 1);
    dim3 grid_size(N, M);
    
    batch_cosine_similarity_kernel<<<grid_size, block_size>>>(
        queries.data(), documents.data(), d_results, N, M, D
    );
    THROW_CUDA_EX();
    
    // Copy results back
    std::vector<float> results_data(N * M);
    CHECK_CUDA_EX(cudaMemcpy(results_data.data(), d_results, 
                            N * M * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_EX(cudaFree(d_results));
    
    // Convert to tensor<T> and upload
    std::vector<T> results_t_data(N * M);
    for (int i = 0; i < N * M; ++i) {
        results_t_data[i] = static_cast<T>(results_data[i]);
    }
    results.upload(results_t_data.data());
    
    return results;
}

template<typename T>
template<typename ScoreType>
std::vector<std::pair<int, ScoreType>> VectorSearch<T>::top_k_similar(
    const tensor<T>& query,
    const std::vector<tensor<T>>& candidates,
    int k,
    float (*similarity_func)(const tensor<T>&, const tensor<T>&)) {
    
    if (candidates.empty() || k <= 0) {
        return {};
    }
    
    std::vector<std::pair<int, ScoreType>> results;
    results.reserve(candidates.size());
    
    // Compute similarities
    for (size_t i = 0; i < candidates.size(); ++i) {
        float similarity = similarity_func(query, candidates[i]);
        results.emplace_back(static_cast<int>(i), static_cast<ScoreType>(similarity));
    }
    
    // Sort by similarity (descending)
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return top-k
    if (k < static_cast<int>(results.size())) {
        results.resize(k);
    }
    
    return results;
}

template<typename T>
void VectorSearch<T>::build_embedding_index(const std::vector<tensor<T>>& embeddings) {
    build_index_internal(embeddings);
    index_built_ = true;
}

template<typename T>
std::vector<std::pair<int, float>> VectorSearch<T>::search_index(
    const tensor<T>& query, int top_k) {
    
    if (!index_built_ || embedding_index_.empty()) {
        return {};
    }
    
    // Use cosine similarity by default
    auto results = top_k_similar<float>(query, embedding_index_, top_k, cosine_similarity);
    
    return results;
}

template<typename T>
tensor<T> VectorSearch<T>::normalize_vectors(const tensor<T>& vectors) {
    const auto& shape = vectors.shape();
    if (shape.size() != 2) {
        throw std::invalid_argument("Input tensor must be 2D");
    }
    
    int N = shape[0];  // Number of vectors
    int D = shape[1];  // Dimension
    
    tensor<T> normalized(shape);
    
    // Download data
    std::vector<T> data(vectors.size());
    vectors.download(data.data());
    
    // Normalize each vector
    for (int i = 0; i < N; ++i) {
        float norm = 0.0f;
        for (int j = 0; j < D; ++j) {
            float val = static_cast<float>(data[i * D + j]);
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 0.0f) {
            for (int j = 0; j < D; ++j) {
                data[i * D + j] = static_cast<T>(static_cast<float>(data[i * D + j]) / norm);
            }
        }
    }
    
    normalized.upload(data.data());
    return normalized;
}

template<typename T>
float VectorSearch<T>::compute_norm(const tensor<T>& vector) {
    std::vector<T> data(vector.size());
    vector.download(data.data());
    
    float norm = 0.0f;
    for (const auto& val : data) {
        float f_val = static_cast<float>(val);
        norm += f_val * f_val;
    }
    
    return std::sqrt(norm);
}

template<typename T>
void VectorSearch<T>::build_index_internal(const std::vector<tensor<T>>& embeddings) {
    embedding_index_.clear();
    embedding_index_.reserve(embeddings.size());
    
    // Clone each tensor to avoid copy constructor issues
    for (const auto& embedding : embeddings) {
        embedding_index_.push_back(embedding.clone());
    }
}

// Explicit template instantiations
template class VectorSearch<float>;
template class VectorSearch<__half>;

} // namespace dnn::utils

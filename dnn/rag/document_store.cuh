#pragma once

#include "../core/tensor.cuh"
#include "../tokens/tokenizer.cuh"
#include "../models/training_model.cuh"
#include "../utils/common.cuh"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <stdexcept>

namespace dnn {

// Forward declarations removed - using actual includes above

// Document container storing text content, tokenized IDs, embeddings, and metadata.
// Serves as the basic unit of knowledge in RAG systems for retrieval and context generation.
class Document {
public:
	std::string id;
	std::string content;
	std::vector<int> token_ids;
	std::unique_ptr<tensor<float>> embedding;  // Smart pointer to avoid constructor issues
	std::unordered_map<std::string, std::string> metadata;

	Document() : id(""), content(""), token_ids(), embedding(nullptr), metadata() {}
	Document(const std::string& id, const std::string& content)
		: id(id), content(content), token_ids(), embedding(nullptr), metadata() {
	}
	
	// Destructor (default is fine since unique_ptr handles cleanup)
	~Document() = default;
	
	// Move constructor and assignment
	Document(Document&&) = default;
	Document& operator=(Document&&) = default;
	
	// Delete copy constructor and assignment to avoid tensor copy issues
	Document(const Document&) = delete;
	Document& operator=(const Document&) = delete;
	
	// Helper methods
	bool has_embedding() const { return embedding != nullptr; }
	void set_embedding(std::unique_ptr<tensor<float>> emb) { embedding = std::move(emb); }
	const tensor<float>& get_embedding() const { 
		if (!embedding) throw std::runtime_error("No embedding available");
		return *embedding; 
	}
};

// RAG knowledge base manager. Stores documents, computes embeddings, and performs similarity search.
// Provides vector database functionality for efficient document retrieval in RAG applications.
class DocumentStore {
public:
	DocumentStore(std::shared_ptr<dnn::Tokenizer> tokenizer,
		int embedding_dim = 768,
		int max_doc_length = 512);

	// Document management
	void add_document(const std::string& id, const std::string& content,
		const std::unordered_map<std::string, std::string>& metadata = {});  // Add document to knowledge base
	void remove_document(const std::string& id);  // Remove document from knowledge base
	Document* get_document(const std::string& id);  // Access document by ID
	const Document* get_document(const std::string& id) const;

	// Batch operations
	void add_documents(const std::vector<std::pair<std::string, std::string>>& docs);  // Add multiple documents at once

	// Search interface - Core RAG retrieval
	std::vector<std::pair<Document*, float>> search_similar(
		const tensor<float>& query_embedding,
		int top_k = 5,
		float threshold = 0.0f) const;  // Find most similar documents to query

	// Persistence
	void save(const std::string& path) const;
	void load(const std::string& path);

	// Statistics
	size_t size() const { return documents_.size(); }
	int embedding_dim() const { return embedding_dim_; }
	bool has_embeddings() const { return !embedding_index_.empty(); }

	// Get all documents (for iteration)
	const std::unordered_map<std::string, Document>& get_documents() const { return documents_; }

	// Get tokenizer for query processing
	std::shared_ptr<dnn::Tokenizer> get_tokenizer() const { return tokenizer_; }
	
	// Embedding model setup
	template<typename T>
	void set_embedding_model(std::shared_ptr<dnn::TrainingModel<T>> model);

	// Public methods for embedding updates (to avoid template issues)
	void update_embeddings_internal();

private:
	std::shared_ptr<dnn::Tokenizer> tokenizer_;
	std::unordered_map<std::string, Document> documents_;
	int embedding_dim_;
	int max_doc_length_;

	// Embedding models (support both float and half precision)
	std::shared_ptr<dnn::TrainingModel<float>> embed_f32_;
	std::shared_ptr<dnn::TrainingModel<__half>> embed_f16_;

	// Vector search index (simple L2 distance for now)
	mutable std::vector<std::pair<std::string, std::unique_ptr<tensor<float>>>> embedding_index_;
	mutable bool index_dirty_ = true;

	// Helper methods for embedding computation
	tensor<float> encode_ids_with_float(const std::vector<int>& ids);
	tensor<float> encode_ids_with_half(const std::vector<int>& ids);
	static void l2_normalize_inplace(tensor<float>& v);

	void rebuild_index() const;
	float compute_similarity(const tensor<float>& a, const tensor<float>& b) const;
	void tokenize_document(Document& doc);
};

} // namespace dnn

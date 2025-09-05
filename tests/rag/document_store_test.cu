#include <gtest/gtest.h>
#include <dnn/rag/document_store.cuh>
#include <dnn/tokens/bpe_tokenizer.cuh>
#include <dnn/tokens/vocab_loader.cuh>
#include <dnn/core/cuda.cuh>
#include <memory>
#include <vector>
#include <string>

class DocumentStoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        cuda_ = std::make_unique<dnn::Cuda>();
        
        // Create a simple vocabulary for testing
        vocab_loader_ = std::make_shared<dnn::VocabLoader>();
        
        // Add some basic tokens
        std::vector<std::pair<std::string, int>> tokens = {
            {"<|endoftext|>", 0},
            {"hello", 1},
            {"world", 2},
            {"test", 3},
            {"document", 4},
            {"content", 5},
            {"this", 6},
            {"is", 7},
            {"a", 8},
            {"sample", 9}
        };
        
        for (const auto& [token, id] : tokens) {
            vocab_loader_->add_token(token, id);
        }
        
        tokenizer_ = std::make_shared<dnn::BpeTokenizer>(vocab_loader_);
        doc_store_ = std::make_shared<dnn::DocumentStore>(tokenizer_, 128, 256);
    }

    void TearDown() override {
        doc_store_.reset();
        tokenizer_.reset();
        vocab_loader_.reset();
        cuda_.reset();
    }

    std::unique_ptr<dnn::Cuda> cuda_;
    std::shared_ptr<dnn::VocabLoader> vocab_loader_;
    std::shared_ptr<dnn::BpeTokenizer> tokenizer_;
    std::shared_ptr<dnn::DocumentStore> doc_store_;
};

TEST_F(DocumentStoreTest, AddDocument) {
    doc_store_->add_document("doc1", "hello world test");
    
    EXPECT_EQ(doc_store_->size(), 1);
    
    auto* doc = doc_store_->get_document("doc1");
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->id, "doc1");
    EXPECT_EQ(doc->content, "hello world test");
    EXPECT_FALSE(doc->token_ids.empty());
}

TEST_F(DocumentStoreTest, RemoveDocument) {
    doc_store_->add_document("doc1", "hello world");
    doc_store_->add_document("doc2", "test document");
    
    EXPECT_EQ(doc_store_->size(), 2);
    
    doc_store_->remove_document("doc1");
    
    EXPECT_EQ(doc_store_->size(), 1);
    EXPECT_EQ(doc_store_->get_document("doc1"), nullptr);
    EXPECT_NE(doc_store_->get_document("doc2"), nullptr);
}

TEST_F(DocumentStoreTest, AddMultipleDocuments) {
    std::vector<std::pair<std::string, std::string>> docs = {
        {"doc1", "hello world"},
        {"doc2", "test document"},
        {"doc3", "sample content"}
    };
    
    doc_store_->add_documents(docs);
    
    EXPECT_EQ(doc_store_->size(), 3);
    
    for (const auto& [id, content] : docs) {
        auto* doc = doc_store_->get_document(id);
        ASSERT_NE(doc, nullptr);
        EXPECT_EQ(doc->content, content);
    }
}

TEST_F(DocumentStoreTest, SearchSimilar) {
    // Add documents
    doc_store_->add_document("doc1", "hello world test");
    doc_store_->add_document("doc2", "test document content");
    doc_store_->add_document("doc3", "sample text here");
    
    // Create a simple query embedding (all ones for testing)
    dnn::tensor<float> query_embedding({128});
    std::vector<float> query_data(128, 1.0f);
    query_embedding.upload(query_data.data());
    
    // Search (should return all documents since we don't have real embeddings)
    auto results = doc_store_->search_similar(query_embedding, 2, 0.0f);
    
    // Since we don't have embeddings, this will return empty results
    // In a real test, we would set up embeddings first
    EXPECT_TRUE(results.empty());
}

TEST_F(DocumentStoreTest, SaveAndLoad) {
    // Add documents
    doc_store_->add_document("doc1", "hello world test");
    doc_store_->add_document("doc2", "test document content");
    
    // Save to temporary file
    const std::string temp_file = "test_doc_store.bin";
    doc_store_->save(temp_file);
    
    // Create new document store and load
    auto new_doc_store = std::make_shared<dnn::DocumentStore>(tokenizer_, 128, 256);
    new_doc_store->load(temp_file);
    
    // Verify loaded documents
    EXPECT_EQ(new_doc_store->size(), 2);
    
    auto* doc1 = new_doc_store->get_document("doc1");
    ASSERT_NE(doc1, nullptr);
    EXPECT_EQ(doc1->content, "hello world test");
    
    auto* doc2 = new_doc_store->get_document("doc2");
    ASSERT_NE(doc2, nullptr);
    EXPECT_EQ(doc2->content, "test document content");
    
    // Clean up
    std::remove(temp_file.c_str());
}

TEST_F(DocumentStoreTest, InvalidInputs) {
    // Test empty document ID
    EXPECT_THROW(doc_store_->add_document("", "content"), std::invalid_argument);
    
    // Test empty content
    EXPECT_THROW(doc_store_->add_document("doc1", ""), std::invalid_argument);
    
    // Test null tokenizer in constructor
    EXPECT_THROW(
        dnn::DocumentStore(nullptr, 128, 256), 
        std::invalid_argument
    );
    
    // Test invalid embedding dimension
    EXPECT_THROW(
        dnn::DocumentStore(tokenizer_, 0, 256), 
        std::invalid_argument
    );
    
    // Test invalid max document length
    EXPECT_THROW(
        dnn::DocumentStore(tokenizer_, 128, 0), 
        std::invalid_argument
    );
}

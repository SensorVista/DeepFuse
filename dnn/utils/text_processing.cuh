#pragma once

#include <string>
#include <vector>
#include <regex>
#include <algorithm>
#include <sstream>
#include <cctype>

namespace dnn::utils {

// Text processing utilities for RAG systems. Handles text cleaning, chunking, and context formatting.
// Prepares documents and queries for embedding and ensures consistent text representation.
class TextProcessor {
public:
    // Text cleaning and preprocessing
    static std::string clean_text(const std::string& text);
    static std::string normalize_whitespace(const std::string& text);
    static std::string remove_special_chars(const std::string& text);
    static std::string to_lowercase(const std::string& text);
    
    // Text chunking for documents
    static std::vector<std::string> chunk_text(const std::string& text, 
                                              int max_chunk_size = 512,
                                              int overlap = 50);
    
    // Context formatting - RAG context assembly
    static std::string format_rag_context(const std::string& query,
                                        const std::vector<std::string>& retrieved_docs,
                                        const std::string& template_str = "Context: {}\n\nQuestion: {}\n\nAnswer:");  // Combine query + retrieved docs

    // Text similarity (simple word overlap)
    static float compute_text_similarity(const std::string& a, const std::string& b);

    // Text preprocessing for RAG - Clean text for retrieval
    static std::string preprocess_for_rag(const std::string& text);  // Normalize text for knowledge base
    
    // Document splitting
    static std::vector<std::string> split_into_sentences(const std::string& text);
    static std::vector<std::string> split_into_paragraphs(const std::string& text);
    
private:
    static const std::regex whitespace_regex_;
    static const std::regex special_chars_regex_;
    static const std::regex sentence_end_regex_;
    
    static std::vector<std::string> tokenize_words(const std::string& text);
    static std::string join_strings(const std::vector<std::string>& strings, const std::string& delimiter);
};

} // namespace dnn::utils

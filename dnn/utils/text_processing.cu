#include "text_processing.cuh"
#include <stdexcept>
#include <unordered_set>
#include <cmath>

namespace dnn::utils {

// Static regex patterns
const std::regex TextProcessor::whitespace_regex_(R"(\s+)");
const std::regex TextProcessor::special_chars_regex_(R"([^\w\s])");
const std::regex TextProcessor::sentence_end_regex_(R"([.!?]+)");

std::string TextProcessor::clean_text(const std::string& text) {
    if (text.empty()) {
        return text;
    }
    
    std::string cleaned = text;
    
    // Remove excessive whitespace
    cleaned = normalize_whitespace(cleaned);
    
    // Remove leading/trailing whitespace
    cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r"));
    cleaned.erase(cleaned.find_last_not_of(" \t\n\r") + 1);
    
    return cleaned;
}

std::string TextProcessor::normalize_whitespace(const std::string& text) {
    if (text.empty()) {
        return text;
    }
    
    // Replace multiple whitespace characters with single space
    std::string normalized = std::regex_replace(text, whitespace_regex_, " ");
    
    return normalized;
}

std::string TextProcessor::remove_special_chars(const std::string& text) {
    if (text.empty()) {
        return text;
    }
    
    // Remove special characters but keep alphanumeric and whitespace
    std::string cleaned = std::regex_replace(text, special_chars_regex_, "");
    
    return cleaned;
}

std::string TextProcessor::to_lowercase(const std::string& text) {
    if (text.empty()) {
        return text;
    }
    
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    return lower;
}

std::vector<std::string> TextProcessor::chunk_text(const std::string& text, 
                                                 int max_chunk_size, int overlap) {
    if (text.empty() || max_chunk_size <= 0) {
        return {};
    }
    
    if (overlap < 0) {
        overlap = 0;
    }
    if (overlap >= max_chunk_size) {
        overlap = max_chunk_size / 2;
    }
    
    std::vector<std::string> chunks;
    std::vector<std::string> words = tokenize_words(text);
    
    if (words.empty()) {
        return chunks;
    }
    
    size_t start = 0;
    while (start < words.size()) {
        size_t end = std::min(start + max_chunk_size, words.size());
        
        std::vector<std::string> chunk_words(words.begin() + start, words.begin() + end);
        chunks.push_back(join_strings(chunk_words, " "));
        
        if (end >= words.size()) {
            break;
        }
        
        // Move start position with overlap
        start = end - overlap;
    }
    
    return chunks;
}

std::string TextProcessor::format_rag_context(const std::string& query,
                                            const std::vector<std::string>& retrieved_docs,
                                            const std::string& template_str) {
    if (retrieved_docs.empty()) {
        return template_str;
    }
    
    // Join retrieved documents
    std::string context = join_strings(retrieved_docs, "\n\n");
    
    // Replace placeholders in template
    std::string formatted = template_str;
    
    // Replace {context} placeholder
    size_t context_pos = formatted.find("{}");
    if (context_pos != std::string::npos) {
        formatted.replace(context_pos, 2, context);
    }
    
    // Replace {query} placeholder if it exists
    size_t query_pos = formatted.find("{query}");
    if (query_pos != std::string::npos) {
        formatted.replace(query_pos, 7, query);
    }
    
    return formatted;
}

float TextProcessor::compute_text_similarity(const std::string& a, const std::string& b) {
    if (a.empty() && b.empty()) {
        return 1.0f;
    }
    if (a.empty() || b.empty()) {
        return 0.0f;
    }
    
    // Simple word overlap similarity
    std::vector<std::string> words_a = tokenize_words(to_lowercase(a));
    std::vector<std::string> words_b = tokenize_words(to_lowercase(b));
    
    if (words_a.empty() && words_b.empty()) {
        return 1.0f;
    }
    if (words_a.empty() || words_b.empty()) {
        return 0.0f;
    }
    
    // Create word sets
    std::unordered_set<std::string> set_a(words_a.begin(), words_a.end());
    std::unordered_set<std::string> set_b(words_b.begin(), words_b.end());
    
    // Compute intersection
    std::unordered_set<std::string> intersection;
    for (const auto& word : set_a) {
        if (set_b.find(word) != set_b.end()) {
            intersection.insert(word);
        }
    }
    
    // Compute union
    std::unordered_set<std::string> union_set = set_a;
    union_set.insert(set_b.begin(), set_b.end());
    
    // Jaccard similarity
    if (union_set.empty()) {
        return 0.0f;
    }
    
    return static_cast<float>(intersection.size()) / static_cast<float>(union_set.size());
}

std::string TextProcessor::preprocess_for_rag(const std::string& text) {
    if (text.empty()) {
        return text;
    }
    
    std::string processed = text;
    
    // Clean the text
    processed = clean_text(processed);
    
    // Normalize whitespace
    processed = normalize_whitespace(processed);
    
    // Remove excessive punctuation
    // Keep basic punctuation but remove excessive repetition
    std::string result;
    char prev_char = '\0';
    int repeat_count = 0;
    
    for (char c : processed) {
        if (c == prev_char && (c == '.' || c == '!' || c == '?' || c == ',')) {
            repeat_count++;
            if (repeat_count < 3) {  // Allow up to 2 repetitions
                result += c;
            }
        } else {
            result += c;
            repeat_count = 0;
        }
        prev_char = c;
    }
    
    return result;
}

std::vector<std::string> TextProcessor::split_into_sentences(const std::string& text) {
    if (text.empty()) {
        return {};
    }
    
    std::vector<std::string> sentences;
    std::string current_sentence;
    
    for (char c : text) {
        current_sentence += c;
        
        if (c == '.' || c == '!' || c == '?') {
            std::string cleaned = clean_text(current_sentence);
            if (!cleaned.empty()) {
                sentences.push_back(cleaned);
            }
            current_sentence.clear();
        }
    }
    
    // Add remaining text as last sentence
    if (!current_sentence.empty()) {
        std::string cleaned = clean_text(current_sentence);
        if (!cleaned.empty()) {
            sentences.push_back(cleaned);
        }
    }
    
    return sentences;
}

std::vector<std::string> TextProcessor::split_into_paragraphs(const std::string& text) {
    if (text.empty()) {
        return {};
    }
    
    std::vector<std::string> paragraphs;
    std::istringstream stream(text);
    std::string line;
    
    while (std::getline(stream, line)) {
        std::string cleaned = clean_text(line);
        if (!cleaned.empty()) {
            paragraphs.push_back(cleaned);
        }
    }
    
    return paragraphs;
}

std::vector<std::string> TextProcessor::tokenize_words(const std::string& text) {
    if (text.empty()) {
        return {};
    }
    
    std::vector<std::string> words;
    std::istringstream stream(text);
    std::string word;
    
    while (stream >> word) {
        // Remove punctuation from word boundaries
        if (!word.empty()) {
            // Remove leading punctuation
            while (!word.empty() && !std::isalnum(word.front())) {
                word.erase(0, 1);
            }
            
            // Remove trailing punctuation
            while (!word.empty() && !std::isalnum(word.back())) {
                word.pop_back();
            }
            
            if (!word.empty()) {
                words.push_back(word);
            }
        }
    }
    
    return words;
}

std::string TextProcessor::join_strings(const std::vector<std::string>& strings, const std::string& delimiter) {
    if (strings.empty()) {
        return "";
    }
    
    std::ostringstream result;
    for (size_t i = 0; i < strings.size(); ++i) {
        if (i > 0) {
            result << delimiter;
        }
        result << strings[i];
    }
    
    return result.str();
}

} // namespace dnn::utils

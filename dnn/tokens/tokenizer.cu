#include "tokenizer.cuh"

#include <sstream>
#include <cctype>
#include <algorithm>
#include <stdexcept>
#include <regex>

namespace dnn {

// Initialize static regex patterns
const std::regex Tokenizer::whitespace_regex(R"(\s+)");
const std::regex Tokenizer::control_chars_regex(R"([\x00-\x1f\x7f-\x9f])");

Tokenizer::Tokenizer(std::shared_ptr<VocabLoader> vocab)
    : vocab_(vocab) {
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_special_tokens) const {
    std::vector<int> token_ids;
    
    if (add_special_tokens) {
        token_ids.push_back(get_bos_token_id());
    }
    
    // Clean and split text
    std::string cleaned_text = clean_text(text);
    auto words = split_into_words(cleaned_text);
    
    // Process each word
    for (const auto& word : words) {
        // Apply BPE to word
        auto subwords = apply_bpe_to_word(word);
        
        // Convert subwords to token IDs
        for (const auto& subword : subwords) {
            int token_id = vocab_->token_to_id(subword);
            if (token_id == -1) {
                token_id = get_unk_token_id();
            }
            token_ids.push_back(token_id);
        }
    }
    
    if (add_special_tokens) {
        token_ids.push_back(get_eos_token_id());
    }
    
    return token_ids;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids, bool skip_special_tokens) const {
    std::string text;
    
    for (size_t i = 0; i < token_ids.size(); ++i) {
        int token_id = token_ids[i];
        
        // Skip special tokens if requested
        if (skip_special_tokens && 
            (token_id == get_bos_token_id() || 
             token_id == get_eos_token_id() || 
             token_id == get_unk_token_id())) {
            continue;
        }
        
        // Get token string
        std::string token = vocab_->id_to_token(token_id);
        
        // Handle special tokens
        if (token == "<|endoftext|>") {
            continue;
        }
        
        // Add token to text
        text += token;
    }
    
    return text;
}

std::vector<std::string> Tokenizer::split_into_words(const std::string& text) const {
    std::vector<std::string> words;
    std::string word;
    
    for (char c : text) {
        if (std::isspace(c)) {
            if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        } else {
            word += c;
        }
    }
    
    if (!word.empty()) {
        words.push_back(word);
    }
    
    return words;
}

std::vector<std::string> Tokenizer::split_into_chars(const std::string& word) const {
    std::vector<std::string> chars;
    for (char c : word) {
        chars.push_back(std::string(1, c));
    }
    return chars;
}

std::vector<std::string> Tokenizer::apply_bpe_to_word(const std::string& word) const {
    // Start with individual characters
    std::vector<std::string> subwords = split_into_chars(word);
    
    // Apply BPE merge rules
    bool merged;
    do {
        merged = false;
        for (const auto& merge : vocab_->get_bpe_merges()) {
            for (size_t i = 0; i < subwords.size() - 1; ++i) {
                std::string pair = subwords[i] + subwords[i + 1];
                if (pair == merge.first) {
                    // Merge the pair
                    subwords[i] = merge.second;
                    subwords.erase(subwords.begin() + i + 1);
                    merged = true;
                    break;
                }
            }
            if (merged) break;
        }
    } while (merged);
    
    return subwords;
}

std::string Tokenizer::clean_text(const std::string& text) const {
    // Replace multiple whitespace with single space
    std::string cleaned = std::regex_replace(text, whitespace_regex, " ");
    
    // Remove control characters
    cleaned = std::regex_replace(cleaned, control_chars_regex, "");
    
    // Trim whitespace
    cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r\f\v"));
    cleaned.erase(cleaned.find_last_not_of(" \t\n\r\f\v") + 1);
    
    return cleaned;
}

}  // namespace dnn

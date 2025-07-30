#include "bpe_tokenizer.cuh"

#include <sstream>
#include <cctype>
#include <algorithm>
#include <stdexcept>
#include <regex>

namespace dnn {

BpeTokenizer::BpeTokenizer(std::shared_ptr<VocabLoader> vocab)
    : vocab_(vocab) {
    if (!vocab_) {
        throw std::invalid_argument("VocabLoader cannot be null");
    }
}

std::string BpeTokenizer::text_to_unicode(const std::string& text) const {
    std::string result;
    for (unsigned char c : text) {
        auto it = vocab_->bytes_to_unicode().find(c);
        if (it != vocab_->bytes_to_unicode().end()) {
            result += static_cast<char>(it->second);
        } else {
            result += c;  // Fallback to original character if not found
        }
    }
    return result;
}

std::string BpeTokenizer::unicode_to_text(const std::string& unicode) const {
    std::string result;
    for (unsigned char c : unicode) {
        auto it = vocab_->unicode_to_bytes().find(c);
        if (it != vocab_->unicode_to_bytes().end()) {
            result += static_cast<char>(it->second);
        } else {
            result += c;  // Fallback to original character if not found
        }
    }
    return result;
}

std::string BpeTokenizer::clean_text(const std::string& text) const {
    // Replace multiple whitespace with single space
    std::string cleaned = std::regex_replace(text, whitespace_regex, " ");
    
    // Remove control characters and split on them
    std::string result;
    for (char c : cleaned) {
        if (c >= 0x20 && c != 0x7F) {  // Keep printable characters
            result += c;
        } else {
            // Replace control characters with space to ensure proper word splitting
            result += ' ';
        }
    }
    
    // Trim whitespace
    result.erase(0, result.find_first_not_of(" \t\n\r\f\v"));
    result.erase(result.find_last_not_of(" \t\n\r\f\v") + 1);
    
    return result;
}

std::vector<int> BpeTokenizer::encode(const std::string& text, bool add_special_tokens) const {
    std::vector<int> token_ids;
    
    if (add_special_tokens) {
        token_ids.push_back(get_bos_token_id());
    }
    
    // Clean and convert text to unicode
    std::string cleaned_text = clean_text(text);
    std::string unicode_text = text_to_unicode(cleaned_text);
    
    // Split into words and process each word
    auto words = split_into_words(unicode_text);
    for (const auto& word : words) {
        // First check if the word exists as a whole
        int token_id = vocab_->token_to_id(word);
        if (token_id != vocab_->get_unk_token_id()) {
            // Word exists as a whole, use it directly
            token_ids.push_back(token_id);
        } else {
            // Word doesn't exist, try to tokenize it character by character
            for (char c : word) {
                std::string char_str(1, c);
                token_id = vocab_->token_to_id(char_str);
                if (token_id != vocab_->get_unk_token_id()) {
                    token_ids.push_back(token_id);
                } else {
                    // If character is unknown, use UNK token
                    token_ids.push_back(vocab_->get_unk_token_id());
                }
            }
        }
    }
    
    if (add_special_tokens) {
        token_ids.push_back(get_eos_token_id());
    }
    
    return token_ids;
}

std::string BpeTokenizer::decode(const std::vector<int>& token_ids, bool skip_special_tokens) const {
    std::string text;
    
    for (size_t i = 0; i < token_ids.size(); ++i) {
        int token_id = token_ids[i];
        
        // Skip special tokens if requested
        if (skip_special_tokens && 
            (token_id == get_bos_token_id() || 
             token_id == get_eos_token_id() || 
             token_id == get_unk_token_id() ||
             token_id == get_pad_token_id())) {
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
    
    // Convert unicode back to text
    return unicode_to_text(text);
}

std::vector<std::string> BpeTokenizer::split_into_words(const std::string& text) const {
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

std::vector<std::string> BpeTokenizer::split_into_chars(const std::string& word) const {
    std::vector<std::string> chars;
    for (char c : word) {
        chars.push_back(std::string(1, c));
    }
    return chars;
}

std::vector<std::string> BpeTokenizer::apply_bpe_to_word(const std::string& word) const {
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

void BpeTokenizer::save(std::ostream& out) const {
    vocab_->save(out);
}

void BpeTokenizer::load(std::istream& in) {
    vocab_->load(in);
}

}  // namespace dnn 
#pragma once

#include "vocab_loader.cuh"

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <regex>
#include <algorithm>

namespace dnn {

class Tokenizer {
public:
    explicit Tokenizer(std::shared_ptr<VocabLoader> vocab);

    // Encode text into token IDs
    std::vector<int> encode(const std::string& text, bool add_special_tokens = false) const;
    
    // Decode token IDs back into text
    std::string decode(const std::vector<int>& token_ids, bool skip_special_tokens = true) const;

    // Get special token IDs
    int get_bos_token_id() const { return vocab_->get_bos_token_id(); }
    int get_eos_token_id() const { return vocab_->get_eos_token_id(); }
    int get_unk_token_id() const { return vocab_->get_unk_token_id(); }

private:
    std::shared_ptr<VocabLoader> vocab_;

    // Split text into words
    std::vector<std::string> split_into_words(const std::string& text) const;
    
    // Split word into characters
    std::vector<std::string> split_into_chars(const std::string& word) const;
    
    // Apply BPE to a word
    std::vector<std::string> apply_bpe_to_word(const std::string& word) const;
    
    // Clean text before tokenization
    std::string clean_text(const std::string& text) const;
    
    // Regex patterns for text cleaning
    static const std::regex whitespace_regex;
    static const std::regex control_chars_regex;
};

}  // namespace dnn

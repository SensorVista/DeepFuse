#pragma once

#include "tokenizer.cuh"
#include "vocab_loader.cuh"

#include <string>
#include <vector>
#include <memory>
#include <regex>
#include <iostream>

namespace dnn {

/// BpeTokenizer implements GPT-2's BPE tokenization.
/// Handles byte-to-unicode conversion, BPE merging, and special tokens.

class BpeTokenizer : public Tokenizer {
public:
    explicit BpeTokenizer(std::shared_ptr<VocabLoader> vocab);

    // Tokenizer interface implementation
    std::vector<int> encode(const std::string& text, bool add_special_tokens = false) const override;
    std::string decode(const std::vector<int>& token_ids, bool skip_special_tokens = true) const override;

    int get_bos_token_id() const override { return vocab_->get_bos_token_id(); }
    int get_eos_token_id() const override { return vocab_->get_eos_token_id(); }
    int get_unk_token_id() const override { return vocab_->get_unk_token_id(); }
    int get_pad_token_id() const override { return vocab_->get_pad_token_id(); }
    int vocab_size() const override { return vocab_->size(); }

    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

    // Convert text to bytes and then to unicode
    std::string text_to_unicode(const std::string& text) const;
    
    // Convert unicode back to bytes and then to text
    std::string unicode_to_text(const std::string& unicode) const;

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
    static inline const std::regex whitespace_regex{R"(\s+)"};
    static inline const std::regex control_chars_regex{R"([\x00-\x1f\x7f-\x9f])"};
};

}  // namespace dnn 
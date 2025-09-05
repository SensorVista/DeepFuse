#pragma once

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace dnn {

/// VocabLoader maps tokens to indices and vice versa.
/// Implements GPT-2's BPE vocabulary with proper special tokens and byte-to-unicode mapping.

class VocabLoader {
public:
    VocabLoader();
    ~VocabLoader();

    // Add token manually
    void add_token(const std::string& token);
    void add_token(const std::string& token, int id);

    // Load vocabulary from file
    void load_from_file(const std::string& path);

    void save(std::ostream& out) const;
    void load(std::istream& in);

    // Get token ID
    int token_to_id(const std::string& token) const;

    // Get token string
    std::string id_to_token(int id) const;

    // Get vocabulary size
    int size() const;

    // Reset all mappings
    void clear();

    // Get special token IDs
    int get_bos_token_id() const { return token_to_id("<|endoftext|>"); }
    int get_eos_token_id() const { return token_to_id("<|endoftext|>"); }
    int get_unk_token_id() const { return token_to_id("<|unk|>"); }
    int get_pad_token_id() const { return token_to_id("<|pad|>"); }

    // Get BPE merge rules
    const std::vector<std::pair<std::string, std::string>>& get_bpe_merges() const;

    // Byte-to-unicode conversion
    const std::unordered_map<int, int>& bytes_to_unicode() const { return byte_to_unicode_; }
    const std::unordered_map<int, int>& unicode_to_bytes() const { return unicode_to_byte_; }

private:
    std::unordered_map<std::string, int> token_to_index_;
    std::vector<std::string> index_to_token_;
    
    // BPE merge rules
    std::vector<std::pair<std::string, std::string>> bpe_merges_;
    
    // Byte-to-unicode mapping
    std::unordered_map<int, int> byte_to_unicode_;
    std::unordered_map<int, int> unicode_to_byte_;
    
    // Load BPE merge rules
    void load_bpe_merges(const std::string& path);
    
    // Initialize byte-to-unicode mapping
    void initialize_byte_to_unicode();
};

}  // namespace dnn

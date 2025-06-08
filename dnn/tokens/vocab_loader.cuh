#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <regex>
#include <stdexcept>

namespace dnn {

/// VocabLoader maps tokens to indices and vice versa.
/// For GPT-2, this uses a BPE vocabulary file.
/// Stubbed for Phase 1: no JSON or disk I/O yet.

class VocabLoader {
public:
    VocabLoader();
    ~VocabLoader();

    // Add token manually
    void add_token(const std::string& token);

    // Load vocabulary from file
    void load_from_file(const std::string& path);

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
    int get_unk_token_id() const { return token_to_id("<|endoftext|>"); }

    // Get BPE merge rules
    const std::vector<std::pair<std::string, std::string>>& get_bpe_merges() const;

private:
    std::unordered_map<std::string, int> token_to_index_;
    std::vector<std::string> index_to_token_;
    
    // BPE merge rules
    std::vector<std::pair<std::string, std::string>> bpe_merges_;
    
    // Load BPE merge rules
    void load_bpe_merges(const std::string& path);
    
    // Apply BPE merge rules
    std::string apply_bpe(const std::string& token) const;
};

}  // namespace dnn

#include "vocab_loader.cuh"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>

namespace dnn {

VocabLoader::VocabLoader() {
}

VocabLoader::~VocabLoader() {
}

void VocabLoader::add_token(const std::string& token) {
    if (token_to_index_.count(token) == 0) {
        int id = static_cast<int>(index_to_token_.size());
        token_to_index_[token] = id;
        index_to_token_.push_back(token);
    }
}

int VocabLoader::token_to_id(const std::string& token) const {
    auto it = token_to_index_.find(token);
    if (it != token_to_index_.end()) {
        return it->second;
    }
    return -1;  // Token not found
}

std::string VocabLoader::id_to_token(int id) const {
    if (id >= 0 && id < static_cast<int>(index_to_token_.size())) {
        return index_to_token_[id];
    }
    return "<UNK>";  // Invalid ID
}

int VocabLoader::size() const {
    return static_cast<int>(index_to_token_.size());
}

void VocabLoader::clear() {
    token_to_index_.clear();
    index_to_token_.clear();
    bpe_merges_.clear();
}

void VocabLoader::load_from_file(const std::string& path) {
    clear();
    
    // Load vocabulary
    std::ifstream vocab_file(path);
    if (!vocab_file.is_open()) {
        throw std::runtime_error("Could not open vocabulary file: " + path);
    }
    
    std::string line;
    int index = 0;
    while (std::getline(vocab_file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Add token to vocabulary
        token_to_index_[line] = index;
        index_to_token_.push_back(line);
        index++;
    }
    
    // Load BPE merge rules
    std::string merges_path = path + ".merges";
    load_bpe_merges(merges_path);
}

void VocabLoader::load_bpe_merges(const std::string& path) {
    std::ifstream merges_file(path);
    if (!merges_file.is_open()) {
        throw std::runtime_error("Could not open BPE merges file: " + path);
    }
    
    std::string line;
    while (std::getline(merges_file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Parse merge rule
        std::istringstream iss(line);
        std::string pair, merged;
        iss >> pair >> merged;
        
        if (!pair.empty() && !merged.empty()) {
            bpe_merges_.emplace_back(pair, merged);
        }
    }
}

const std::vector<std::pair<std::string, std::string>>& VocabLoader::get_bpe_merges() const {
    return bpe_merges_;
}

}  // namespace dnn

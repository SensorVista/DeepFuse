#include "vocab_loader.cuh"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace dnn {

VocabLoader::VocabLoader() {
    initialize_byte_to_unicode();
}

VocabLoader::~VocabLoader() {
}

void VocabLoader::initialize_byte_to_unicode() {
    // GPT-2's byte-to-unicode mapping
    // Maps bytes to unicode characters that are valid in Python 3
    for (int b = 0; b < 256; b++) {
        if ((b >= 0x41 && b <= 0x5A) || // A-Z
            (b >= 0x61 && b <= 0x7A) || // a-z
            (b >= 0x30 && b <= 0x39) || // 0-9
            b == 0x20 || b == 0x21 || b == 0x22 || b == 0x23 || b == 0x24 ||
            b == 0x25 || b == 0x26 || b == 0x27 || b == 0x28 || b == 0x29 ||
            b == 0x2A || b == 0x2B || b == 0x2C || b == 0x2D || b == 0x2E ||
            b == 0x2F || b == 0x3A || b == 0x3B || b == 0x3C || b == 0x3D ||
            b == 0x3E || b == 0x3F || b == 0x40 || b == 0x5B || b == 0x5C ||
            b == 0x5D || b == 0x5E || b == 0x5F || b == 0x60 || b == 0x7B ||
            b == 0x7C || b == 0x7D || b == 0x7E) {
            byte_to_unicode_[b] = b;
            unicode_to_byte_[b] = b;
        } else {
            byte_to_unicode_[b] = b + 256;
            unicode_to_byte_[b + 256] = b;
        }
    }
}

void VocabLoader::add_token(const std::string& token) {
    if (token_to_index_.count(token) == 0) {
        int id = static_cast<int>(index_to_token_.size());
        token_to_index_[token] = id;
        index_to_token_.push_back(token);
    }
}

void VocabLoader::add_token(const std::string& token, int id) {
    if (token_to_index_.count(token) == 0) {
        // Ensure the index_to_token_ vector is large enough
        if (id >= static_cast<int>(index_to_token_.size())) {
            index_to_token_.resize(id + 1);
        }
        token_to_index_[token] = id;
        index_to_token_[id] = token;
    }
}

int VocabLoader::token_to_id(const std::string& token) const {
    auto it = token_to_index_.find(token);
    if (it != token_to_index_.end()) {
        return it->second;
    }
    return get_unk_token_id();  // Return UNK token ID instead of -1
}

std::string VocabLoader::id_to_token(int id) const {
    if (id >= 0 && id < static_cast<int>(index_to_token_.size())) {
        return index_to_token_[id];
    }
    return "<|endoftext|>";  // Return endoftext token for invalid IDs
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
    
    // Read vocabulary file
    std::ifstream vocab_file(path);
    if (!vocab_file.is_open()) {
        throw std::runtime_error("Could not open vocabulary file: " + path);
    }
    
    // Read entire file into string
    std::stringstream buffer;
    buffer << vocab_file.rdbuf();
    std::string vocab_json = buffer.str();
    
    // Parse JSON
    try {
        auto vocab_data = nlohmann::json::parse(vocab_json);
        
        // Add special tokens first
        add_token("<|endoftext|>");
        add_token("<|unk|>");
        add_token("<|pad|>");
        
        // Parse the JSON object
        for (const auto& [token, id] : vocab_data.items()) {
            if (id.is_number()) {
                // Add token to vocabulary
                token_to_index_[token] = id.get<int>();
                // Ensure index_to_token_ has enough space
                while (index_to_token_.size() <= static_cast<size_t>(id.get<int>())) {
                    index_to_token_.push_back("");
                }
                index_to_token_[id.get<int>()] = token;
            }
        }
        
        // Verify vocabulary size
        if (token_to_index_.empty()) {
            throw std::runtime_error("Vocabulary file is empty or invalid");
        }
        
        // Try to load BPE merge rules if the file exists
        std::filesystem::path vocab_path(path);
        std::filesystem::path merges_path = vocab_path.parent_path() / "merges.txt";
        std::ifstream merges_file(merges_path);
        if (merges_file.is_open()) {
            merges_file.close();
            load_bpe_merges(merges_path.string());
        }
        
    } 
    catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Failed to parse vocabulary JSON: " + std::string(e.what()));
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Unhandled exception while parsing vocabulary JSON: " + std::string(e.what()));
    }
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
        
        // Parse merge rule - format is "token1 token2 merged"
        size_t first_space = line.find(' ');
        if (first_space == std::string::npos) {
            continue;  // Skip invalid lines
        }
        
        std::string first = line.substr(0, first_space);
        std::string second = line.substr(first_space + 1);
        
        // Create the merged token by concatenating without space
        std::string merged = first + second;
        
        // Store the merge rule
        bpe_merges_.emplace_back(first + " " + second, merged);
    }
}

const std::vector<std::pair<std::string, std::string>>& VocabLoader::get_bpe_merges() const {
    return bpe_merges_;
}

void VocabLoader::save(std::ostream& out) const {
    int token_count = static_cast<int>(token_to_index_.size());
    out.write(reinterpret_cast<const char*>(&token_count), sizeof(token_count));
    for (const auto& kv : token_to_index_) {
        int key_len = static_cast<int>(kv.first.size());
        out.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
        out.write(kv.first.data(), key_len);
        out.write(reinterpret_cast<const char*>(&kv.second), sizeof(kv.second));
    }
    int idx_count = static_cast<int>(index_to_token_.size());
    out.write(reinterpret_cast<const char*>(&idx_count), sizeof(idx_count));
    for (const auto& token : index_to_token_) {
        int tok_len = static_cast<int>(token.size());
        out.write(reinterpret_cast<const char*>(&tok_len), sizeof(tok_len));
        out.write(token.data(), tok_len);
    }
    int merge_count = static_cast<int>(bpe_merges_.size());
    out.write(reinterpret_cast<const char*>(&merge_count), sizeof(merge_count));
    for (const auto& merge : bpe_merges_) {
        int first_len = static_cast<int>(merge.first.size());
        int second_len = static_cast<int>(merge.second.size());
        out.write(reinterpret_cast<const char*>(&first_len), sizeof(first_len));
        out.write(merge.first.data(), first_len);
        out.write(reinterpret_cast<const char*>(&second_len), sizeof(second_len));
        out.write(merge.second.data(), second_len);
    }
    int btu_count = static_cast<int>(byte_to_unicode_.size());
    out.write(reinterpret_cast<const char*>(&btu_count), sizeof(btu_count));
    for (const auto& kv : byte_to_unicode_) {
        out.write(reinterpret_cast<const char*>(&kv.first), sizeof(kv.first));
        out.write(reinterpret_cast<const char*>(&kv.second), sizeof(kv.second));
    }
    int utb_count = static_cast<int>(unicode_to_byte_.size());
    out.write(reinterpret_cast<const char*>(&utb_count), sizeof(utb_count));
    for (const auto& kv : unicode_to_byte_) {
        out.write(reinterpret_cast<const char*>(&kv.first), sizeof(kv.first));
        out.write(reinterpret_cast<const char*>(&kv.second), sizeof(kv.second));
    }
}

void VocabLoader::load(std::istream& in) {
    int token_count;
    in.read(reinterpret_cast<char*>(&token_count), sizeof(token_count));
    token_to_index_.clear();
    for (int i = 0; i < token_count; ++i) {
        int key_len;
        in.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        std::string key(key_len, '\0');
        in.read(&key[0], key_len);
        int val;
        in.read(reinterpret_cast<char*>(&val), sizeof(val));
        token_to_index_[key] = val;
    }
    int idx_count;
    in.read(reinterpret_cast<char*>(&idx_count), sizeof(idx_count));
    index_to_token_.resize(idx_count);
    for (int i = 0; i < idx_count; ++i) {
        int tok_len;
        in.read(reinterpret_cast<char*>(&tok_len), sizeof(tok_len));
        std::string token(tok_len, '\0');
        in.read(&token[0], tok_len);
        index_to_token_[i] = token;
    }
    int merge_count;
    in.read(reinterpret_cast<char*>(&merge_count), sizeof(merge_count));
    bpe_merges_.resize(merge_count);
    for (int i = 0; i < merge_count; ++i) {
        int first_len, second_len;
        in.read(reinterpret_cast<char*>(&first_len), sizeof(first_len));
        std::string first(first_len, '\0');
        in.read(&first[0], first_len);
        in.read(reinterpret_cast<char*>(&second_len), sizeof(second_len));
        std::string second(second_len, '\0');
        in.read(&second[0], second_len);
        bpe_merges_[i] = {first, second};
    }
    int btu_count;
    in.read(reinterpret_cast<char*>(&btu_count), sizeof(btu_count));
    byte_to_unicode_.clear();
    for (int i = 0; i < btu_count; ++i) {
        int key, val;
        in.read(reinterpret_cast<char*>(&key), sizeof(key));
        in.read(reinterpret_cast<char*>(&val), sizeof(val));
        byte_to_unicode_[key] = val;
    }
    int utb_count;
    in.read(reinterpret_cast<char*>(&utb_count), sizeof(utb_count));
    unicode_to_byte_.clear();
    for (int i = 0; i < utb_count; ++i) {
        int key, val;
        in.read(reinterpret_cast<char*>(&key), sizeof(key));
        in.read(reinterpret_cast<char*>(&val), sizeof(val));
        unicode_to_byte_[key] = val;
    }
}

}  // namespace dnn

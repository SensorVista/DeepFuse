#pragma once

#include <string>
#include <vector>
#include <iostream>

namespace dnn {

/// Base class for tokenizers.
/// Defines the interface for different tokenization schemes (BPE, WordPiece, etc.).

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    // Encode text into token IDs
    virtual std::vector<int> encode(const std::string& text, bool add_special_tokens = false) const = 0;
    
    // Decode token IDs back into text
    virtual std::string decode(const std::vector<int>& token_ids, bool skip_special_tokens = true) const = 0;

    // Get special token IDs
    virtual int get_bos_token_id() const = 0;  // Begin of Sequence
    virtual int get_eos_token_id() const = 0;  // End of Sequence
    virtual int get_unk_token_id() const = 0;  // Unknown
    virtual int get_pad_token_id() const = 0;  // Padding

    // Get vocabulary size
    virtual int vocab_size() const = 0;

    // Save tokenizer metadata and state
    virtual void save(std::ostream& out) const = 0;
    // Load tokenizer metadata and state
    virtual void load(std::istream& in) = 0;
};

}  // namespace dnn

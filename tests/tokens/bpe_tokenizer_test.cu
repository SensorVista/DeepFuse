#include <gtest/gtest.h>

#include <dnn/tokens/bpe_tokenizer.cuh>
#include <dnn/tokens/vocab_loader.cuh>

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <filesystem>

namespace dnn {
namespace test {

class BpeTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test files
        create_test_vocab_file();
        create_test_merges_file();
        
        // Create a test vocabulary
        vocab_ = std::make_shared<VocabLoader>();
        vocab_->load_from_file(vocab_filepath);
        
        // Create tokenizer with test vocabulary
        tokenizer_ = std::make_unique<BpeTokenizer>(vocab_);
    }

    void TearDown() override {
        tokenizer_.reset();
        vocab_.reset();
        
        // Clean up test files
        std::filesystem::remove(vocab_filepath);
        std::filesystem::remove(merges_filepath);
    }

    void create_test_vocab_file() {
        std::ofstream file(vocab_filepath);
        file << "{\n";
        file << "  \"<|endoftext|>\": 0,\n";
        file << "  \"<|unk|>\": 1,\n";
        file << "  \"<|pad|>\": 2,\n";
        file << "  \"h\": 3,\n";
        file << "  \"e\": 4,\n";
        file << "  \"l\": 5,\n";
        file << "  \"o\": 6,\n";
        file << "  \"w\": 7,\n";
        file << "  \"r\": 8,\n";
        file << "  \"d\": 9,\n";
        file << "  \"he\": 10,\n";
        file << "  \"ll\": 11,\n";
        file << "  \"lo\": 12,\n";
        file << "  \"wo\": 13,\n";
        file << "  \"rl\": 14,\n";
        file << "  \"ld\": 15,\n";
        file << "  \"hello\": 16,\n";
        file << "  \"world\": 17\n";
        file << "}\n";
    }

    void create_test_merges_file() {
        std::ofstream file(merges_filepath);
        file << "h e\n";
        file << "l l\n";
        file << "he l\n";
        file << "ll o\n";
        file << "w o\n";
        file << "r l\n";
        file << "wo r\n";
        file << "rl d\n";
    }

    const std::string vocab_filepath = "test_vocab.json";
    const std::string merges_filepath = "test_vocab.json.merges";
    std::shared_ptr<VocabLoader> vocab_;
    std::unique_ptr<BpeTokenizer> tokenizer_;
};

TEST_F(BpeTokenizerTest, Constructor) {
    EXPECT_NO_THROW(BpeTokenizer tokenizer(vocab_));
    EXPECT_THROW(BpeTokenizer tokenizer(nullptr), std::invalid_argument);
}

TEST_F(BpeTokenizerTest, EncodeBasic) {
    std::string text = "hello world";
    auto token_ids = tokenizer_->encode(text);
    
    EXPECT_EQ(token_ids.size(), 2);
    EXPECT_EQ(token_ids[0], vocab_->token_to_id("hello"));
    EXPECT_EQ(token_ids[1], vocab_->token_to_id("world"));
}

TEST_F(BpeTokenizerTest, EncodeWithSpecialTokens) {
    std::string text = "hello world";
    auto token_ids = tokenizer_->encode(text, true);  // add_special_tokens = true
    
    EXPECT_EQ(token_ids.size(), 4);  // BOS + tokens + EOS
    EXPECT_EQ(token_ids[0], tokenizer_->get_bos_token_id());
    EXPECT_EQ(token_ids[1], vocab_->token_to_id("hello"));
    EXPECT_EQ(token_ids[2], vocab_->token_to_id("world"));
    EXPECT_EQ(token_ids[3], tokenizer_->get_eos_token_id());
}

TEST_F(BpeTokenizerTest, DecodeBasic) {
    std::vector<int> token_ids = {
        vocab_->token_to_id("hello"),
        vocab_->token_to_id("world")
    };
    
    std::string text = tokenizer_->decode(token_ids);
    EXPECT_EQ(text, "helloworld");  // No space between tokens
}

TEST_F(BpeTokenizerTest, DecodeWithSpecialTokens) {
    std::vector<int> token_ids = {
        tokenizer_->get_bos_token_id(),
        vocab_->token_to_id("hello"),
        vocab_->token_to_id("world"),
        tokenizer_->get_eos_token_id()
    };
    
    std::string text = tokenizer_->decode(token_ids, true);  // skip_special_tokens = true
    EXPECT_EQ(text, "helloworld");  // No space between tokens
}

TEST_F(BpeTokenizerTest, UnknownToken) {
    std::string text = "xyz";  // Using characters not in our test vocabulary
    auto token_ids = tokenizer_->encode(text);
    
    // Each character should get an UNK token since none are in vocabulary
    EXPECT_EQ(token_ids.size(), text.length());
    for (int token_id : token_ids) {
        EXPECT_EQ(token_id, tokenizer_->get_unk_token_id());
    }
}

TEST_F(BpeTokenizerTest, TextCleaning) {
    // Test that spaces are preserved as they are in GPT-2
    std::string text = "  hello  world  ";  // Extra spaces
    auto token_ids = tokenizer_->encode(text);
    
    EXPECT_EQ(token_ids.size(), 2);
    EXPECT_EQ(token_ids[0], vocab_->token_to_id("hello"));
    EXPECT_EQ(token_ids[1], vocab_->token_to_id("world"));
    
    // Decode should preserve original spacing
    std::string decoded = tokenizer_->decode(token_ids);
    EXPECT_EQ(decoded, "helloworld");  // No spaces in decoded text
}

TEST_F(BpeTokenizerTest, SpecialTokenIds) {
    EXPECT_EQ(tokenizer_->get_bos_token_id(), vocab_->get_bos_token_id());
    EXPECT_EQ(tokenizer_->get_eos_token_id(), vocab_->get_eos_token_id());
    EXPECT_EQ(tokenizer_->get_unk_token_id(), vocab_->get_unk_token_id());
    EXPECT_EQ(tokenizer_->get_pad_token_id(), vocab_->get_pad_token_id());
}

TEST_F(BpeTokenizerTest, VocabSize) {
    EXPECT_EQ(tokenizer_->vocab_size(), vocab_->size());
}

TEST_F(BpeTokenizerTest, ByteToUnicode) {
    std::string text = "hello";
    std::string unicode = tokenizer_->text_to_unicode(text);
    std::string back = tokenizer_->unicode_to_text(unicode);
    EXPECT_EQ(back, text);
}

TEST_F(BpeTokenizerTest, BpeMerging) {
    // Test that BPE merging works correctly for unknown words
    std::string text = "helo";  // Not in vocabulary, should use BPE
    auto token_ids = tokenizer_->encode(text);
    
    // Should be tokenized as: h + e + l + o
    EXPECT_EQ(token_ids.size(), 4);
    EXPECT_EQ(token_ids[0], vocab_->token_to_id("h"));
    EXPECT_EQ(token_ids[1], vocab_->token_to_id("e"));
    EXPECT_EQ(token_ids[2], vocab_->token_to_id("l"));
    EXPECT_EQ(token_ids[3], vocab_->token_to_id("o"));
}

TEST_F(BpeTokenizerTest, ControlCharacters) {
    std::string text = "hello\x01world";  // Contains control character
    auto token_ids = tokenizer_->encode(text);
    EXPECT_EQ(token_ids.size(), 2);
    EXPECT_EQ(token_ids[0], vocab_->token_to_id("hello"));
    EXPECT_EQ(token_ids[1], vocab_->token_to_id("world"));
}

} // namespace test
} // namespace dnn 
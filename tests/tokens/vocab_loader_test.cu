#include <gtest/gtest.h>

#include <dnn/tokens/vocab_loader.cuh>

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

namespace dnn {
namespace test {

class VocabLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary test vocabulary file
        create_test_vocab_file();
        create_test_merges_file();
    }

    void TearDown() override {
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
        file << "  \"hello\": 3,\n";
        file << "  \"world\": 4,\n";
        file << "  \"test\": 5,\n";
        file << "  \"token\": 6\n";
        file << "}\n";
    }

    void create_test_merges_file() {
        std::ofstream file(merges_filepath);
        file << "#version: 0.2\n";
        file << "h e\n";
        file << "he l\n";
        file << "hel l\n";
        file << "hell o\n";
        file << "w o\n";
        file << "wo r\n";
        file << "wor l\n";
        file << "worl d\n";
    }

    const std::string vocab_filepath = "test_vocab.json";
    const std::string merges_filepath = "merges.txt";
};

TEST_F(VocabLoaderTest, Constructor) {
    VocabLoader vocab;
    EXPECT_EQ(vocab.size(), 0);
    
    // Test byte-to-unicode mapping initialization
    const auto& byte_to_unicode = vocab.bytes_to_unicode();
    const auto& unicode_to_byte = vocab.unicode_to_bytes();
    EXPECT_FALSE(byte_to_unicode.empty());
    EXPECT_FALSE(unicode_to_byte.empty());
}

TEST_F(VocabLoaderTest, AddToken) {
    VocabLoader vocab;
    vocab.add_token("test_token");
    EXPECT_EQ(vocab.size(), 1);
    EXPECT_EQ(vocab.token_to_id("test_token"), 0);
}

TEST_F(VocabLoaderTest, LoadFromFile) {
    VocabLoader vocab;
    vocab.load_from_file(vocab_filepath);
    
    EXPECT_EQ(vocab.size(), 7);  // 7 tokens in test file
    EXPECT_EQ(vocab.token_to_id("<|endoftext|>"), 0);
    EXPECT_EQ(vocab.token_to_id("<|unk|>"), 1);
    EXPECT_EQ(vocab.token_to_id("<|pad|>"), 2);
    EXPECT_EQ(vocab.token_to_id("hello"), 3);
    EXPECT_EQ(vocab.token_to_id("world"), 4);
}

TEST_F(VocabLoaderTest, TokenToIdAndIdToToken) {
    VocabLoader vocab;
    vocab.load_from_file(vocab_filepath);
    
    // Test token_to_id
    EXPECT_EQ(vocab.token_to_id("test"), 5);
    EXPECT_EQ(vocab.token_to_id("token"), 6);
    EXPECT_EQ(vocab.token_to_id("unknown"), vocab.get_unk_token_id());
    
    // Test id_to_token
    EXPECT_EQ(vocab.id_to_token(0), "<|endoftext|>");
    EXPECT_EQ(vocab.id_to_token(1), "<|unk|>");
    EXPECT_EQ(vocab.id_to_token(2), "<|pad|>");
    EXPECT_EQ(vocab.id_to_token(3), "hello");
    EXPECT_EQ(vocab.id_to_token(4), "world");
    
    // Test invalid ID
    EXPECT_EQ(vocab.id_to_token(-1), "<|endoftext|>");
    EXPECT_EQ(vocab.id_to_token(100), "<|endoftext|>");
}

TEST_F(VocabLoaderTest, SpecialTokens) {
    VocabLoader vocab;
    vocab.load_from_file(vocab_filepath);
    
    // Test special token IDs
    EXPECT_EQ(vocab.get_bos_token_id(), vocab.token_to_id("<|endoftext|>"));
    EXPECT_EQ(vocab.get_eos_token_id(), vocab.token_to_id("<|endoftext|>"));
    EXPECT_EQ(vocab.get_unk_token_id(), vocab.token_to_id("<|unk|>"));
    EXPECT_EQ(vocab.get_pad_token_id(), vocab.token_to_id("<|pad|>"));
}

TEST_F(VocabLoaderTest, BpeMerges) {
    VocabLoader vocab;
    vocab.load_from_file(vocab_filepath);
    
    const auto& merges = vocab.get_bpe_merges();
    EXPECT_FALSE(merges.empty());
    EXPECT_EQ(merges.size(), 8);  // 8 merge rules in test file
    
    // Test first merge rule
    EXPECT_EQ(merges[0].first, "h e");
    EXPECT_EQ(merges[0].second, "he");
    
    // Test middle merge rule
    EXPECT_EQ(merges[3].first, "hell o");
    EXPECT_EQ(merges[3].second, "hello");
    
    // Test last merge rule
    EXPECT_EQ(merges[7].first, "worl d");
    EXPECT_EQ(merges[7].second, "world");
}

TEST_F(VocabLoaderTest, Clear) {
    VocabLoader vocab;
    vocab.load_from_file(vocab_filepath);
    EXPECT_EQ(vocab.size(), 7);
    
    vocab.clear();
    EXPECT_EQ(vocab.size(), 0);
    EXPECT_TRUE(vocab.get_bpe_merges().empty());
}

TEST_F(VocabLoaderTest, InvalidFile) {
    VocabLoader vocab;
    EXPECT_THROW(vocab.load_from_file("nonexistent_file.json"), std::runtime_error);
}

TEST_F(VocabLoaderTest, InvalidJson) {
    VocabLoader vocab;
    std::ofstream file("invalid.json");
    file << "invalid json content";
    file.close();
    
    EXPECT_THROW(vocab.load_from_file("invalid.json"), std::runtime_error);
    std::filesystem::remove("invalid.json");
}

} // namespace test
} // namespace dnn 
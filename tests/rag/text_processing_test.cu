#include <gtest/gtest.h>
#include <dnn/utils/text_processing.cuh>
#include <string>
#include <vector>

class TextProcessingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TextProcessingTest, CleanText) {
    std::string input = "  Hello    world!   \n\n  Test.  ";
    std::string expected = "Hello world! Test.";
    std::string result = dnn::utils::TextProcessor::clean_text(input);
    EXPECT_EQ(result, expected);
}

TEST_F(TextProcessingTest, NormalizeWhitespace) {
    std::string input = "Hello    world\n\nTest";
    std::string expected = "Hello world Test";
    std::string result = dnn::utils::TextProcessor::normalize_whitespace(input);
    EXPECT_EQ(result, expected);
}

TEST_F(TextProcessingTest, RemoveSpecialChars) {
    std::string input = "Hello, world! How are you?";
    std::string expected = "Hello world How are you";
    std::string result = dnn::utils::TextProcessor::remove_special_chars(input);
    EXPECT_EQ(result, expected);
}

TEST_F(TextProcessingTest, ToLowercase) {
    std::string input = "Hello World TEST";
    std::string expected = "hello world test";
    std::string result = dnn::utils::TextProcessor::to_lowercase(input);
    EXPECT_EQ(result, expected);
}

TEST_F(TextProcessingTest, ChunkText) {
    std::string text = "This is a test document with multiple words that should be chunked properly.";
    std::vector<std::string> chunks = dnn::utils::TextProcessor::chunk_text(text, 10, 2);
    
    EXPECT_GT(chunks.size(), 1);
    
    // Check that chunks don't exceed max size
    for (const auto& chunk : chunks) {
        std::vector<std::string> words = dnn::utils::TextProcessor::chunk_text(chunk, 1000, 0);
        EXPECT_LE(words.size(), 10);
    }
}

TEST_F(TextProcessingTest, FormatRagContext) {
    std::string query = "What is the answer?";
    std::vector<std::string> docs = {"Document 1 content", "Document 2 content"};
    std::string template_str = "Context: {}\n\nQuestion: {}\n\nAnswer:";
    
    std::string result = dnn::utils::TextProcessor::format_rag_context(query, docs, template_str);
    
    EXPECT_TRUE(result.find("Document 1 content") != std::string::npos);
    EXPECT_TRUE(result.find("Document 2 content") != std::string::npos);
    EXPECT_TRUE(result.find("What is the answer?") != std::string::npos);
}

TEST_F(TextProcessingTest, ComputeTextSimilarity) {
    std::string text1 = "hello world test";
    std::string text2 = "hello world example";
    std::string text3 = "completely different content";
    
    float similarity1 = dnn::utils::TextProcessor::compute_text_similarity(text1, text2);
    float similarity2 = dnn::utils::TextProcessor::compute_text_similarity(text1, text3);
    
    EXPECT_GT(similarity1, similarity2);
    EXPECT_GE(similarity1, 0.0f);
    EXPECT_LE(similarity1, 1.0f);
}

TEST_F(TextProcessingTest, PreprocessForRag) {
    std::string input = "  Hello!!!   World...   Test???   ";
    std::string result = dnn::utils::TextProcessor::preprocess_for_rag(input);
    
    // Should be cleaned and normalized
    EXPECT_TRUE(result.find("Hello") != std::string::npos);
    EXPECT_TRUE(result.find("World") != std::string::npos);
    EXPECT_TRUE(result.find("Test") != std::string::npos);
    
    // Should not have excessive punctuation
    EXPECT_TRUE(result.find("!!!") == std::string::npos);
    EXPECT_TRUE(result.find("...") == std::string::npos);
    EXPECT_TRUE(result.find("???") == std::string::npos);
}

TEST_F(TextProcessingTest, SplitIntoSentences) {
    std::string text = "First sentence. Second sentence! Third sentence? Fourth sentence.";
    std::vector<std::string> sentences = dnn::utils::TextProcessor::split_into_sentences(text);
    
    EXPECT_EQ(sentences.size(), 4);
    EXPECT_TRUE(sentences[0].find("First sentence") != std::string::npos);
    EXPECT_TRUE(sentences[1].find("Second sentence") != std::string::npos);
    EXPECT_TRUE(sentences[2].find("Third sentence") != std::string::npos);
    EXPECT_TRUE(sentences[3].find("Fourth sentence") != std::string::npos);
}

TEST_F(TextProcessingTest, SplitIntoParagraphs) {
    std::string text = "First paragraph\n\nSecond paragraph\n\nThird paragraph";
    std::vector<std::string> paragraphs = dnn::utils::TextProcessor::split_into_paragraphs(text);
    
    EXPECT_EQ(paragraphs.size(), 3);
    EXPECT_TRUE(paragraphs[0].find("First paragraph") != std::string::npos);
    EXPECT_TRUE(paragraphs[1].find("Second paragraph") != std::string::npos);
    EXPECT_TRUE(paragraphs[2].find("Third paragraph") != std::string::npos);
}

TEST_F(TextProcessingTest, EmptyInputs) {
    EXPECT_EQ(dnn::utils::TextProcessor::clean_text(""), "");
    EXPECT_EQ(dnn::utils::TextProcessor::normalize_whitespace(""), "");
    EXPECT_EQ(dnn::utils::TextProcessor::remove_special_chars(""), "");
    EXPECT_EQ(dnn::utils::TextProcessor::to_lowercase(""), "");
    
    std::vector<std::string> empty_chunks = dnn::utils::TextProcessor::chunk_text("", 10, 2);
    EXPECT_TRUE(empty_chunks.empty());
    
    std::vector<std::string> empty_sentences = dnn::utils::TextProcessor::split_into_sentences("");
    EXPECT_TRUE(empty_sentences.empty());
    
    std::vector<std::string> empty_paragraphs = dnn::utils::TextProcessor::split_into_paragraphs("");
    EXPECT_TRUE(empty_paragraphs.empty());
}

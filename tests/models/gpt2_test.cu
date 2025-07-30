#include <gtest/gtest.h>

#include <dnn/models/gpt2.cuh>
#include <dnn/tokens/bpe_tokenizer.cuh>
#include <dnn/tokens/vocab_loader.cuh>

#include <vector>
#include <random>
#include <filesystem>
#include <fstream>

using namespace dnn;

namespace dnn {
namespace test {

class GPT2Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }

};

// Helper: Create a minimal vocab and tokenizer for testing
std::shared_ptr<BpeTokenizer> create_test_tokenizer() {
    auto vocab_loader = std::make_shared<VocabLoader>();
    // Add a few tokens for testing
    vocab_loader->add_token("hello");
    vocab_loader->add_token("world");
    vocab_loader->add_token("<|endoftext|>");
    return std::make_shared<BpeTokenizer>(vocab_loader);
}

TEST_F(GPT2Test, Constructor) {
    auto tokenizer = create_test_tokenizer();
    Gpt2<float> model(tokenizer, 3, 8, 2, 2, 8, 16, 1e-4f, 0.9f, 0.98f, 1e-8f, true);
    
    // Just check model is constructed
    SUCCEED();
}

TEST_F(GPT2Test, ForwardPass) {
    auto tokenizer = create_test_tokenizer();
    Gpt2<float> model(tokenizer, 3, 8, 2, 2, 8, 16, 1e-4f, 0.9f, 0.98f, 1e-8f, true);
    
    // Create dummy input (batch of 1, sequence length 8)
    std::vector<int> input_tokens = {0, 1, 2, 0, 1, 2, 0, 1};
    auto output = model.forward(input_tokens);
    
    // Check output shape (should match sequence length and vocab size)
    EXPECT_EQ(output.shape()[0], 1); // sequence length
    EXPECT_EQ(output.shape()[1], 3); // vocab size
}

TEST_F(GPT2Test, TrainingStep) {
    auto tokenizer = create_test_tokenizer();
    Gpt2<float> model(tokenizer, 3, 8, 2, 2, 8, 16, 1e-4f, 0.9f, 0.98f, 1e-8f, true);
    std::vector<int> input_tokens = {0, 1, 2, 0, 1, 2, 0, 1};
    std::vector<int> target_tokens = {1, 2, 0, 1, 2, 0, 1, 2};
    float initial_loss = model.loss();
    model.train_step(input_tokens, target_tokens);
    float new_loss = model.loss();
    EXPECT_NE(initial_loss, new_loss);
}

TEST_F(GPT2Test, MultipleTrainingSteps) {
    auto tokenizer = create_test_tokenizer();
    Gpt2<float> model(tokenizer, 3, 8, 2, 2, 8, 16, 1e-4f, 0.9f, 0.98f, 1e-8f, true);
    std::vector<int> input_tokens = {0, 1, 2, 0, 1, 2, 0, 1};
    std::vector<int> target_tokens = {1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<float> losses;
    for (int i = 0; i < 5; ++i) {
        model.train_step(input_tokens, target_tokens);
        losses.push_back(model.loss());
    }
    // Check that at least one loss decreased
    bool loss_decreased = false;
    for (size_t i = 1; i < losses.size(); ++i) {
        if (losses[i] < losses[i-1]) {
            loss_decreased = true;
            break;
        }
    }
    EXPECT_TRUE(loss_decreased);
}

TEST_F(GPT2Test, SaveLoadRoundtrip) {
    auto tokenizer = create_test_tokenizer();
    // Use hidden_dim=12, num_heads=3, intermediate_dim=24 for compatibility
    Gpt2<float> model(tokenizer, 3, 8, 2, 3, 12, 24, 1e-4f, 0.9f, 0.98f, 1e-8f, true);
    std::vector<int> input_tokens = {0, 1, 2, 0, 1, 2, 0, 1};
    std::vector<int> target_tokens = {1, 2, 0, 1, 2, 0, 1, 2};
    model.train_step(input_tokens, target_tokens);
    std::string path = "gpt2_test_model.bin";
    model.save(path);
    auto loaded = Gpt2<float>::load(path, true);
    auto params1 = model.parameters();
    auto params2 = loaded->parameters();
    ASSERT_EQ(params1.size(), params2.size());
    for (size_t i = 0; i < params1.size(); ++i) {
        EXPECT_TRUE(params1[i]->approx_equal(*params2[i], 1e-5f)) << "Parameter " << i << " mismatch after load/save.";
    }
    std::filesystem::remove(path);
}

TEST_F(GPT2Test, LoadCorruptFileThrows) {
    std::string bad_path = "nonexistent_gpt2_file.bin";
    EXPECT_THROW({
        auto m = Gpt2<float>::load(bad_path, true);
    }, std::runtime_error);
}

} // namespace test
} // namespace dnn

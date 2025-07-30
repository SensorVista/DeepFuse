#include <gtest/gtest.h>

#include <dnn/tokens/token_embedding.cuh>
#include <dnn/core/tensor.cuh>

#include <vector>
#include <random>

namespace dnn {
namespace test {

class TokenEmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }

    // Helper function to create a test input tensor of token IDs
    tensor<int> create_test_input(int batch_size, int seq_len) {
        tensor<int> input({batch_size, seq_len});
        std::vector<int> host_data(input.size());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, vocab_size_ - 1);
        
        for (size_t i = 0; i < host_data.size(); ++i) {
            host_data[i] = dis(gen);
        }
        input.upload(host_data.data());
        return input;
    }

    // Helper function to verify embedding properties
    void verify_embedding(const tensor<float>& embedding, int batch_size, int seq_len, int embed_dim) {
        // Check shape
        EXPECT_EQ(embedding.shape()[0], batch_size);
        EXPECT_EQ(embedding.shape()[1], seq_len);
        EXPECT_EQ(embedding.shape()[2], embed_dim);

        // Check that values are not all zero
        std::vector<float> host_embedding(embedding.size());
        embedding.download(host_embedding.data());
        
        bool has_non_zero = false;
        for (float val : host_embedding) {
            if (std::abs(val) > 1e-6) {
                has_non_zero = true;
                break;
            }
        }
        EXPECT_TRUE(has_non_zero);
    }

    const int vocab_size_ = 1000;
    const int embed_dim_ = 512;
    const int max_seq_len_ = 1024;
};

TEST_F(TokenEmbeddingTest, Constructor) {
    EXPECT_NO_THROW(TokenEmbedding<float>(vocab_size_, embed_dim_));
    EXPECT_THROW(TokenEmbedding<float>(0, embed_dim_), std::invalid_argument);
    EXPECT_THROW(TokenEmbedding<float>(vocab_size_, 0), std::invalid_argument);
}

TEST_F(TokenEmbeddingTest, ForwardPass) {
    int batch_size = 2;
    int seq_len = 4;
    
    TokenEmbedding<float> layer(vocab_size_, embed_dim_);
    tensor<int> input = create_test_input(batch_size, seq_len);
    
    tensor<float> output = layer.forward(input);
    
    // Verify output shape and properties
    verify_embedding(output, batch_size, seq_len, embed_dim_);
}

TEST_F(TokenEmbeddingTest, BackwardPass) {
    int batch_size = 2;
    int seq_len = 4;
    
    TokenEmbedding<float> layer(vocab_size_, embed_dim_, 1024, true);
    tensor<int> input = create_test_input(batch_size, seq_len);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Create gradient tensor with non-zero values
    tensor<float> grad_output(output.shape());
    std::vector<float> host_grad(grad_output.size(), 1.0f);
    grad_output.upload(host_grad.data());
    
    // Backward pass
    tensor<float> grad_input = layer.backward(grad_output);
    
    // Check gradient shape
    EXPECT_EQ(grad_input.shape(), input.shape());
    
    // Verify that embedding gradients are non-zero
    auto grads = layer.gradients();
    std::vector<float> host_grad_weights(grads[0]->size());
    grads[0]->download(host_grad_weights.data());
    
    bool has_non_zero = false;
    for (float val : host_grad_weights) {
        if (std::abs(val) > 1e-6) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero) << "Embedding gradients should be non-zero";
    
    // Verify that bias gradients are non-zero
    std::vector<float> host_grad_bias(grads[1]->size());
    grads[1]->download(host_grad_bias.data());
    
    has_non_zero = false;
    for (float val : host_grad_bias) {
        if (std::abs(val) > 1e-6) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero) << "Bias gradients should be non-zero";
}

TEST_F(TokenEmbeddingTest, WeightInitialization) {
    TokenEmbedding<float> layer(vocab_size_, embed_dim_);
    
    // Get weights
    auto params = layer.parameters();
    EXPECT_EQ(params.size(), 2);  // weights and bias
    
    // Check weights shape
    EXPECT_EQ(params[0]->shape()[0], vocab_size_);
    EXPECT_EQ(params[0]->shape()[1], embed_dim_);
    
    // Check bias shape
    EXPECT_EQ(params[1]->shape()[0], embed_dim_);
    
    // Verify weights are initialized
    std::vector<float> host_weights(params[0]->size());
    params[0]->download(host_weights.data());
    
    bool has_non_zero = false;
    for (float val : host_weights) {
        if (std::abs(val) > 1e-6) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);
}

TEST_F(TokenEmbeddingTest, DifferentSequenceLengths) {
    TokenEmbedding<float> layer(vocab_size_, embed_dim_);
    
    // Test with sequence length less than max_seq_len
    tensor<int> input1 = create_test_input(2, 5);
    tensor<float> output1 = layer.forward(input1);
    EXPECT_EQ(output1.shape()[1], 5);
    
    // Test with sequence length equal to max_seq_len
    tensor<int> input2 = create_test_input(2, max_seq_len_);
    tensor<float> output2 = layer.forward(input2);
    EXPECT_EQ(output2.shape()[1], max_seq_len_);
    
    // Test with sequence length greater than max_seq_len
    tensor<int> input3 = create_test_input(2, max_seq_len_ + 1);
    EXPECT_THROW(layer.forward(input3), std::runtime_error);
}

TEST_F(TokenEmbeddingTest, InvalidTokenIds) {
    TokenEmbedding<float> layer(vocab_size_, embed_dim_);
    
    // Create input with invalid token IDs
    tensor<int> input({2, 4});
    std::vector<int> host_data = {vocab_size_, vocab_size_ + 1, -1, -2};
    input.upload(host_data.data());
    
    EXPECT_THROW(layer.forward(input), std::runtime_error);
}

TEST_F(TokenEmbeddingTest, Name) {
    TokenEmbedding<float> layer(vocab_size_, embed_dim_);
    EXPECT_EQ(layer.name(), "TokenEmbedding");
}

} // namespace test
} // namespace dnn 
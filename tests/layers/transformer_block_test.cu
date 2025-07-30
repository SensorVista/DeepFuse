#include <gtest/gtest.h>

#include <dnn/core/cuda.cuh>
#include <dnn/layers/transformer_block.cuh>

#include <vector>
#include <cmath>
#include <random>

namespace dnn {
namespace test {

class TransformerBlockTest : public ::testing::Test {
protected:
    std::unique_ptr<Cuda> cuda_;

    void SetUp() override {
        cuda_ = std::make_unique<Cuda>(0);
        cudaDeviceSynchronize();
    }

    void TearDown() override {
        cudaDeviceSynchronize();
        cuda_.reset();
    }

    // Helper function to create random input tensor
    tensor<float> create_random_input(int batch_size, int seq_len, int embed_dim) {
        std::vector<int> shape = {batch_size, seq_len, embed_dim};
        tensor<float> input(shape);
        
        std::vector<float> data(batch_size * seq_len * embed_dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = dist(gen);
        }
        
        input.upload(data.data());
        return input;
    }

    // Helper function to create attention mask
    tensor<float> create_attention_mask(int batch_size, int seq_len) {
        std::vector<int> shape = {batch_size, 1, seq_len, seq_len};
        tensor<float> mask(shape);
        
        std::vector<float> data(batch_size * seq_len * seq_len);
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    // Create causal mask (lower triangular)
                    data[b * seq_len * seq_len + i * seq_len + j] = (j <= i) ? 0.0f : -1e9f;
                }
            }
        }
        
        mask.upload(data.data());
        return mask;
    }
};

TEST_F(TransformerBlockTest, ConstructorAndInitialization) {
    int embed_dim = 512;
    int num_heads = 8;
    int mlp_hidden_dim = 2048;
    
    TransformerBlock<float> block(embed_dim, num_heads, mlp_hidden_dim);
    
    // Verify dimensions
    EXPECT_EQ(block.name(), "TransformerBlock");
}

TEST_F(TransformerBlockTest, ForwardPass) {
    int batch_size = 2;
    int seq_len = 10;
    int embed_dim = 512;
    int num_heads = 8;
    int mlp_hidden_dim = 2048;
    
    TransformerBlock<float> block(embed_dim, num_heads, mlp_hidden_dim);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, seq_len, embed_dim);
    
    // Create attention mask
    tensor<float> mask = create_attention_mask(batch_size, seq_len);
    block.set_mask(&mask);
    
    // Forward pass
    tensor<float> output = block.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

TEST_F(TransformerBlockTest, BackwardPass) {
    int batch_size = 2;
    int seq_len = 10;
    int embed_dim = 512;
    int num_heads = 8;
    int mlp_hidden_dim = 2048;
    
    TransformerBlock<float> block(embed_dim, num_heads, mlp_hidden_dim, true);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, seq_len, embed_dim);
    
    // Create attention mask
    tensor<float> mask = create_attention_mask(batch_size, seq_len);
    block.set_mask(&mask);
    
    // Forward pass
    tensor<float> output = block.forward(input);
    
    // Create gradient tensor
    tensor<float> grad_output(output.shape());
    std::vector<float> grad_data(output.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < grad_data.size(); ++i) {
        grad_data[i] = dist(gen);
    }
    grad_output.upload(grad_data.data());
    
    // Backward pass
    tensor<float> grad_input = block.backward(grad_output);
    
    // Verify gradient shape
    EXPECT_EQ(grad_input.shape()[0], batch_size);
    EXPECT_EQ(grad_input.shape()[1], seq_len);
    EXPECT_EQ(grad_input.shape()[2], embed_dim);
}

TEST_F(TransformerBlockTest, DifferentDimensions) {
    int batch_size = 4;
    int seq_len = 20;
    int embed_dim = 256;
    int num_heads = 4;
    int mlp_hidden_dim = 1024;
    
    TransformerBlock<float> block(embed_dim, num_heads, mlp_hidden_dim, true);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, seq_len, embed_dim);
    
    // Create attention mask
    tensor<float> mask = create_attention_mask(batch_size, seq_len);
    block.set_mask(&mask);
    
    // Forward pass
    tensor<float> output = block.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
    
    // Create gradient tensor
    tensor<float> grad_output(output.shape());
    std::vector<float> grad_data(output.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < grad_data.size(); ++i) {
        grad_data[i] = dist(gen);
    }
    grad_output.upload(grad_data.data());
    
    // Backward pass
    tensor<float> grad_input = block.backward(grad_output);
    
    // Verify gradient shape
    EXPECT_EQ(grad_input.shape()[0], batch_size);
    EXPECT_EQ(grad_input.shape()[1], seq_len);
    EXPECT_EQ(grad_input.shape()[2], embed_dim);
}

TEST_F(TransformerBlockTest, MaskHandling) {
    int batch_size = 2;
    int seq_len = 10;
    int embed_dim = 512;
    int num_heads = 8;
    int mlp_hidden_dim = 2048;
    
    TransformerBlock<float> block(embed_dim, num_heads, mlp_hidden_dim);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, seq_len, embed_dim);
    
    // Test without mask
    tensor<float> output1 = block.forward(input);
    
    // Create and set mask
    tensor<float> mask = create_attention_mask(batch_size, seq_len);
    block.set_mask(&mask);
    
    // Test with mask
    tensor<float> output2 = block.forward(input);
    
    // Outputs should be different due to masking
    std::vector<float> data1(output1.size());
    std::vector<float> data2(output2.size());
    output1.download(data1.data());
    output2.download(data2.data());
    
    bool different = false;
    for (size_t i = 0; i < data1.size(); ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-5) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

} // namespace test
} // namespace dnn 
#include <gtest/gtest.h>

#include <dnn/tokens/positional_encoding.cuh>
#include <dnn/core/tensor.cuh>

#include <vector>
#include <cmath>

namespace dnn {
namespace test {

class PositionalEncodingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }

    // Helper function to create a test input tensor
    tensor<float> create_test_input(int batch_size, int seq_len, int embed_dim) {
        tensor<float> input({batch_size, seq_len, embed_dim});
        std::vector<float> host_data(input.size());
        for (size_t i = 0; i < host_data.size(); ++i) {
            host_data[i] = static_cast<float>(i) / host_data.size();
        }
        input.upload(host_data.data());
        return input;
    }

    // Helper function to verify positional encoding properties
    void verify_positional_encoding(const tensor<float>& encoding, int embed_dim, int max_seq_len) {
        std::vector<float> host_encoding(encoding.size());
        encoding.download(host_encoding.data());

        // Check shape
        EXPECT_EQ(encoding.shape()[0], max_seq_len);
        EXPECT_EQ(encoding.shape()[1], embed_dim);

        // Check that values are bounded
        for (float val : host_encoding) {
            EXPECT_GE(val, -1.0f);
            EXPECT_LE(val, 1.0f);
        }

        // Check that different positions have different encodings
        bool has_different_values = false;
        for (int i = 0; i < max_seq_len - 1; ++i) {
            for (int j = 0; j < embed_dim; ++j) {
                if (std::abs(host_encoding[i * embed_dim + j] - 
                            host_encoding[(i + 1) * embed_dim + j]) > 1e-6) {
                    has_different_values = true;
                    break;
                }
            }
            if (has_different_values) break;
        }
        EXPECT_TRUE(has_different_values);
    }
};

TEST_F(PositionalEncodingTest, Constructor) {
    EXPECT_NO_THROW(PositionalEncoding<float>(512, 1024));
    EXPECT_THROW(PositionalEncoding<float>(0, 1024), std::invalid_argument);
    EXPECT_THROW(PositionalEncoding<float>(512, 0), std::invalid_argument);
}

TEST_F(PositionalEncodingTest, ForwardPass) {
    int batch_size = 2;
    int seq_len = 4;
    int embed_dim = 8;
    
    PositionalEncoding<float> layer(embed_dim, seq_len);
    tensor<float> input = create_test_input(batch_size, seq_len, embed_dim);
    
    tensor<float> output = layer.forward(input);
    
    // Check output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
    
    // Verify that output is not equal to input (positional encoding was added)
    std::vector<float> host_input(input.size()), host_output(output.size());
    input.download(host_input.data());
    output.download(host_output.data());
    
    bool has_different_values = false;
    for (size_t i = 0; i < host_input.size(); ++i) {
        if (std::abs(host_input[i] - host_output[i]) > 1e-6) {
            has_different_values = true;
            break;
        }
    }
    EXPECT_TRUE(has_different_values);
}

TEST_F(PositionalEncodingTest, BackwardPass) {
    int batch_size = 2;
    int seq_len = 4;
    int embed_dim = 8;
    
    PositionalEncoding<float> layer(embed_dim, seq_len);
    tensor<float> input = create_test_input(batch_size, seq_len, embed_dim);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Create gradient tensor
    tensor<float> grad_output(output.shape());
    grad_output.fill(1.0f);
    
    // Backward pass
    tensor<float> grad_input = layer.backward(grad_output);
    
    // Check gradient shape
    EXPECT_EQ(grad_input.shape(), input.shape());
    
    // Verify that gradient is not zero
    std::vector<float> host_grad(grad_input.size());
    grad_input.download(host_grad.data());
    
    bool has_non_zero = false;
    for (float val : host_grad) {
        if (std::abs(val) > 1e-6) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);
}

TEST_F(PositionalEncodingTest, DifferentSequenceLengths) {
    int embed_dim = 8;
    int max_seq_len = 10;
    
    PositionalEncoding<float> layer(embed_dim, max_seq_len);
    
    // Test with sequence length less than max_seq_len
    tensor<float> input1 = create_test_input(2, 5, embed_dim);
    tensor<float> output1 = layer.forward(input1);
    EXPECT_EQ(output1.shape()[1], 5);
    
    // Test with sequence length equal to max_seq_len
    tensor<float> input2 = create_test_input(2, max_seq_len, embed_dim);
    tensor<float> output2 = layer.forward(input2);
    EXPECT_EQ(output2.shape()[1], max_seq_len);
}

TEST_F(PositionalEncodingTest, Name) {
    PositionalEncoding<float> layer(512, 1024);
    EXPECT_EQ(layer.name(), "PositionalEncoding");
}

} // namespace test
} // namespace dnn 
#include <gtest/gtest.h>

#include <dnn/core/cuda.cuh>
#include <dnn/layers/softmax_layer.cuh>

#include <vector>
#include <cmath>
#include <random>

namespace dnn {
namespace test {

class SoftmaxLayerTest : public ::testing::Test {
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
    tensor<float> create_random_input(int batch_size, int num_heads, int seq_len) {
        std::vector<int> shape = {batch_size, num_heads, seq_len, seq_len};
        tensor<float> input(shape);
        
        std::vector<float> data(batch_size * num_heads * seq_len * seq_len);
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

    // Helper function to compute expected softmax output
    std::vector<float> compute_expected_output(
        const std::vector<float>& input,
        const std::vector<float>& mask,
        int batch_size,
        int num_heads,
        int seq_len) {
        
        std::vector<float> output(input.size());
        
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < seq_len; ++i) {
                    // Find max for numerical stability
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int j = 0; j < seq_len; ++j) {
                        int idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                        int mask_idx = (b * seq_len + i) * seq_len + j;
                        float val = input[idx] + (mask.empty() ? 0.0f : mask[mask_idx]);
                        max_val = std::max(max_val, val);
                    }
                    
                    // Compute exp and sum
                    float sum = 0.0f;
                    std::vector<float> exp_vals(seq_len);
                    for (int j = 0; j < seq_len; ++j) {
                        int idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                        int mask_idx = (b * seq_len + i) * seq_len + j;
                        float val = input[idx] + (mask.empty() ? 0.0f : mask[mask_idx]);
                        exp_vals[j] = std::exp(val - max_val);
                        sum += exp_vals[j];
                    }
                    
                    // Normalize
                    for (int j = 0; j < seq_len; ++j) {
                        int idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                        output[idx] = exp_vals[j] / (sum + 1e-6f);
                    }
                }
            }
        }
        
        return output;
    }
};

TEST_F(SoftmaxLayerTest, ConstructorAndInitialization) {
    SoftmaxLayer<float> layer;
    EXPECT_EQ(layer.name(), "Softmax");
}

TEST_F(SoftmaxLayerTest, ForwardPass) {
    int batch_size = 2;
    int num_heads = 4;
    int seq_len = 8;
    
    SoftmaxLayer<float> layer;
    
    // Create input
    tensor<float> input = create_random_input(batch_size, num_heads, seq_len);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], num_heads);
    EXPECT_EQ(output.shape()[2], seq_len);
    EXPECT_EQ(output.shape()[3], seq_len);
    
    // Download data for verification
    std::vector<float> output_data(output.size());
    std::vector<float> input_data(input.size());
    output.download(output_data.data());
    input.download(input_data.data());
    
    // Compute expected output
    std::vector<float> expected_output = compute_expected_output(
        input_data, std::vector<float>(), batch_size, num_heads, seq_len);
    
    // Verify output values
    for (size_t i = 0; i < output_data.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-5);
    }
}

TEST_F(SoftmaxLayerTest, ForwardPassWithMask) {
    int batch_size = 2;
    int num_heads = 4;
    int seq_len = 8;
    
    SoftmaxLayer<float> layer;
    
    // Create input and mask
    tensor<float> input = create_random_input(batch_size, num_heads, seq_len);
    tensor<float> mask = create_attention_mask(batch_size, seq_len);
    
    // Set mask
    layer.set_mask(&mask);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], num_heads);
    EXPECT_EQ(output.shape()[2], seq_len);
    EXPECT_EQ(output.shape()[3], seq_len);
    
    // Download data for verification
    std::vector<float> output_data(output.size());
    std::vector<float> input_data(input.size());
    std::vector<float> mask_data(mask.size());
    output.download(output_data.data());
    input.download(input_data.data());
    mask.download(mask_data.data());
    
    // Compute expected output
    std::vector<float> expected_output = compute_expected_output(
        input_data, mask_data, batch_size, num_heads, seq_len);
    
    // Verify output values
    for (size_t i = 0; i < output_data.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-5);
    }
}

TEST_F(SoftmaxLayerTest, BackwardPass) {
    int batch_size = 2;
    int num_heads = 4;
    int seq_len = 8;
    
    SoftmaxLayer<float> layer(true);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, num_heads, seq_len);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
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
    tensor<float> grad_input = layer.backward(grad_output);
    
    // Verify gradient shape
    EXPECT_EQ(grad_input.shape()[0], batch_size);
    EXPECT_EQ(grad_input.shape()[1], num_heads);
    EXPECT_EQ(grad_input.shape()[2], seq_len);
    EXPECT_EQ(grad_input.shape()[3], seq_len);
}

TEST_F(SoftmaxLayerTest, DifferentDimensions) {
    int batch_size = 4;
    int num_heads = 8;
    int seq_len = 16;
    
    SoftmaxLayer<float> layer;
    
    // Create input
    tensor<float> input = create_random_input(batch_size, num_heads, seq_len);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], num_heads);
    EXPECT_EQ(output.shape()[2], seq_len);
    EXPECT_EQ(output.shape()[3], seq_len);
    
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
    tensor<float> grad_input = layer.backward(grad_output);
    
    // Verify gradient shape
    EXPECT_EQ(grad_input.shape()[0], batch_size);
    EXPECT_EQ(grad_input.shape()[1], num_heads);
    EXPECT_EQ(grad_input.shape()[2], seq_len);
    EXPECT_EQ(grad_input.shape()[3], seq_len);
}

TEST_F(SoftmaxLayerTest, InvalidInputShape) {
    int batch_size = 2;
    int num_heads = 4;
    int seq_len = 8;
    
    SoftmaxLayer<float> layer;
    
    // Create input with invalid shape
    std::vector<int> invalid_shape = {batch_size, num_heads, seq_len};
    tensor<float> input(invalid_shape);
    
    // Forward pass with invalid shape should throw
    EXPECT_THROW(layer.forward(input), std::runtime_error);
}

TEST_F(SoftmaxLayerTest, MaskedSoftmaxZeroesMaskedPositions) {
    int batch_size = 1, num_heads = 1, seq_len = 3;
    SoftmaxLayer<float> layer;

    // Input: all zeros
    tensor<float> input({batch_size, num_heads, seq_len, seq_len});
    std::vector<float> input_data(seq_len * seq_len, 0.0f);
    input.upload(input_data.data());

    // Mask: only allow [0,0] position
    tensor<float> mask({batch_size, 1, seq_len, seq_len});
    std::vector<float> mask_data(seq_len * seq_len, -1e9f);
    mask_data[0] = 0.0f; // Only [0,0] is unmasked
    mask.upload(mask_data.data());
    layer.set_mask(&mask);

    tensor<float> output = layer.forward(input);
    std::vector<float> output_data(output.size());
    output.download(output_data.data());

    // Only the first position should be ~1, rest ~0
    EXPECT_NEAR(output_data[0], 1.0f, 1e-5);
    for (int i = 1; i < output_data.size(); ++i)
        EXPECT_NEAR(output_data[i], 0.0f, 1e-5);
}

} // namespace test
} // namespace dnn 
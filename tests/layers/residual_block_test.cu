#include <gtest/gtest.h>

#include <dnn/core/cuda.cuh>
#include <dnn/layers/residual_block.cuh>

#include <vector>
#include <cmath>
#include <random>

namespace dnn {
namespace test {

class ResidualBlockTest : public ::testing::Test {
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
    tensor<float> create_random_input(int batch_size, int channels, int height, int width) {
        std::vector<int> shape = {batch_size, channels, height, width};
        tensor<float> input(shape);
        
        std::vector<float> data(batch_size * channels * height * width);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = dist(gen);
        }
        
        input.upload(data.data());
        return input;
    }
};

TEST_F(ResidualBlockTest, ConstructorAndInitialization) {
    int in_channels = 64;
    int out_channels = 128;
    int stride = 2;
    
    // Test basic residual block
    ResidualBlock<float> block(in_channels, out_channels, stride);
    EXPECT_EQ(block.name(), "ResidualBlock");
    
    // Test bottleneck residual block
    ResidualBlock<float> bottleneck(in_channels, out_channels, stride, true);
    EXPECT_EQ(bottleneck.name(), "ResidualBottleneck");
}

TEST_F(ResidualBlockTest, ForwardPassBasic) {
    int batch_size = 2;
    int in_channels = 64;
    int out_channels = 64;  // Same channels, no projection needed
    int height = 32;
    int width = 32;
    
    ResidualBlock<float> block(in_channels, out_channels);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, in_channels, height, width);
    
    // Forward pass
    tensor<float> output = block.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], out_channels);
    EXPECT_EQ(output.shape()[2], height);
    EXPECT_EQ(output.shape()[3], width);
}

TEST_F(ResidualBlockTest, ForwardPassWithStride) {
    int batch_size = 2;
    int in_channels = 64;
    int out_channels = 128;
    int height = 32;
    int width = 32;
    int stride = 2;
    
    ResidualBlock<float> block(in_channels, out_channels, stride);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, in_channels, height, width);
    
    // Forward pass
    tensor<float> output = block.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], out_channels);
    EXPECT_EQ(output.shape()[2], height / stride);
    EXPECT_EQ(output.shape()[3], width / stride);
}

TEST_F(ResidualBlockTest, ForwardPassBottleneck) {
    int batch_size = 2;
    int in_channels = 64;
    int out_channels = 256;
    int height = 32;
    int width = 32;
    
    ResidualBlock<float> block(in_channels, out_channels, 1, true);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, in_channels, height, width);
    
    // Forward pass
    tensor<float> output = block.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], out_channels);
    EXPECT_EQ(output.shape()[2], height);
    EXPECT_EQ(output.shape()[3], width);
}

TEST_F(ResidualBlockTest, BackwardPass) {
    int batch_size = 2;
    int in_channels = 64;
    int out_channels = 128;
    int height = 32;
    int width = 32;
    int stride = 2;
    
    ResidualBlock<float> block(in_channels, out_channels, stride, false, true);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, in_channels, height, width);
    
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
    EXPECT_EQ(grad_input.shape()[1], in_channels);
    EXPECT_EQ(grad_input.shape()[2], height);
    EXPECT_EQ(grad_input.shape()[3], width);
    
    // Check if gradients are computed for parameters
    auto grads = block.gradients();
    EXPECT_GT(grads.size(), 0);
}

TEST_F(ResidualBlockTest, Parameters) {
    int in_channels = 64;
    int out_channels = 128;
    int stride = 2;
    
    // Test basic residual block
    ResidualBlock<float> block(in_channels, out_channels, stride);
    auto params = block.parameters();
    EXPECT_GT(params.size(), 0);
    
    // Test bottleneck residual block
    ResidualBlock<float> bottleneck(in_channels, out_channels, stride, true);
    auto bottleneck_params = bottleneck.parameters();
    EXPECT_GT(bottleneck_params.size(), 0);
}

TEST_F(ResidualBlockTest, DifferentDimensions) {
    int batch_size = 4;
    int in_channels = 32;
    int out_channels = 64;
    int height = 64;
    int width = 64;
    int stride = 2;
    
    ResidualBlock<float> block(in_channels, out_channels, stride, true, true);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, in_channels, height, width);
    
    // Forward pass
    tensor<float> output = block.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], out_channels);
    EXPECT_EQ(output.shape()[2], height / stride);
    EXPECT_EQ(output.shape()[3], width / stride);
    
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
    EXPECT_EQ(grad_input.shape()[1], in_channels);
    EXPECT_EQ(grad_input.shape()[2], height);
    EXPECT_EQ(grad_input.shape()[3], width);
}

} // namespace test
} // namespace dnn 
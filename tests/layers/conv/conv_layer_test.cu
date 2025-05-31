#include <gtest/gtest.h>
#include "dnn/layers/conv/conv_layer.cuh"
#include <vector>
#include <cmath>
#include <random>

namespace lenet5 {
namespace test {

class ConvLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }

    // Helper function to compute expected output size
    std::vector<size_t> compute_output_shape(
        const std::vector<size_t>& input_shape,
        const std::vector<size_t>& kernel_size,
        size_t stride,
        size_t padding,
        size_t out_channels)
    {
        std::vector<size_t> output_shape(4);
        output_shape[0] = input_shape[0];              // batch size
        output_shape[1] = out_channels;

        output_shape[2] = (input_shape[2] + 2 * padding - kernel_size[0]) / stride + 1;
        output_shape[3] = (input_shape[3] + 2 * padding - kernel_size[1]) / stride + 1;

        return output_shape;
    }
};

TEST_F(ConvLayerTest, ConstructorAndInitialization) {
    size_t in_channels = 3;
    size_t out_channels = 4;
    std::vector<size_t> kernel_size = {3, 3};
    size_t stride = 1;
    size_t padding = 1;
    
    ConvLayer<float> layer(in_channels, out_channels, kernel_size, stride, padding);
    
    // Check if weights and bias are properly initialized
    auto params = layer.parameters();
    EXPECT_EQ(params.size(), 2); // weights and bias
    
    // Check weights shape
    EXPECT_EQ(params[0]->shape()[0], out_channels);
    EXPECT_EQ(params[0]->shape()[1], in_channels);
    EXPECT_EQ(params[0]->shape()[2], kernel_size[0]);
    EXPECT_EQ(params[0]->shape()[3], kernel_size[1]);
    
    // Check bias shape
    EXPECT_EQ(params[1]->shape()[0], out_channels);
}

TEST_F(ConvLayerTest, ForwardPass) {
    size_t in_channels = 2;
    size_t out_channels = 3;
    std::vector<size_t> kernel_size = {3, 3};
    size_t stride = 1;
    size_t padding = 1;
    
    ConvLayer<float> layer(in_channels, out_channels, kernel_size, stride, padding);
    
    // Create input tensor (batch_size=1, channels=2, height=4, width=4)
    std::vector<size_t> input_shape = {1, in_channels, 4, 4};
    tensor<float> input(input_shape);
    
    // Fill input with test data
    std::vector<float> input_data(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i) / input_data.size();
    }
    input.upload(input_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    std::vector<size_t> expected_shape = compute_output_shape(input_shape, kernel_size, stride, padding, out_channels);
    EXPECT_EQ(output.shape(), expected_shape);
    
    // Download and verify output values are not zero
    std::vector<float> output_data(output.size());
    output.download(output_data.data());
    
    bool has_non_zero = false;
    for (float val : output_data) {
        if (std::abs(val) > 1e-6) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);
}

TEST_F(ConvLayerTest, BackwardPass) {
    size_t in_channels = 2;
    size_t out_channels = 3;
    std::vector<size_t> kernel_size = {3, 3};
    size_t stride = 1;
    size_t padding = 1;
    
    ConvLayer<float> layer(in_channels, out_channels, kernel_size, stride, padding);
    
    // Create input tensor
    std::vector<size_t> input_shape = {1, in_channels, 4, 4};
    tensor<float> input(input_shape);
    
    // Fill input with test data
    std::vector<float> input_data(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i) / input_data.size();
    }
    input.upload(input_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Create gradient tensor
    tensor<float> grad_output(output.shape());
    std::vector<float> grad_data(output.size());
    for (size_t i = 0; i < grad_data.size(); ++i) {
        grad_data[i] = static_cast<float>(i) / grad_data.size();
    }
    grad_output.upload(grad_data.data());
    
    // Backward pass
    tensor<float> grad_input = layer.backward(grad_output, input);
    
    // Verify gradient shapes
    EXPECT_EQ(grad_input.shape(), input_shape);
    
    // Check if gradients are computed for parameters
    auto grads = layer.gradients();
    EXPECT_EQ(grads.size(), 2); // weights and bias gradients
    
    // Verify weight gradients shape
    EXPECT_EQ(grads[0]->shape()[0], out_channels);
    EXPECT_EQ(grads[0]->shape()[1], in_channels);
    EXPECT_EQ(grads[0]->shape()[2], kernel_size[0]);
    EXPECT_EQ(grads[0]->shape()[3], kernel_size[1]);
    
    // Verify bias gradients shape
    EXPECT_EQ(grads[1]->shape()[0], out_channels);
}

TEST_F(ConvLayerTest, DifferentStrideAndPadding) {
    size_t in_channels = 2;
    size_t out_channels = 3;
    std::vector<size_t> kernel_size = {3, 3};
    size_t stride = 2;
    size_t padding = 0;
    
    ConvLayer<float> layer(in_channels, out_channels, kernel_size, stride, padding);
    
    // Create input tensor (batch_size=1, channels=2, height=5, width=5)
    std::vector<size_t> input_shape = {1, in_channels, 5, 5};
    tensor<float> input(input_shape);
    
    // Fill input with test data
    std::vector<float> input_data(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i) / input_data.size();
    }
    input.upload(input_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    std::vector<size_t> expected_shape = compute_output_shape(input_shape, kernel_size, stride, padding, out_channels);
    EXPECT_EQ(output.shape(), expected_shape);
    
    // Verify output dimensions
    EXPECT_EQ(output.shape()[2], 2); // (5 - 3) / 2 + 1 = 2
    EXPECT_EQ(output.shape()[3], 2); // (5 - 3) / 2 + 1 = 2
}

} // namespace test
} // namespace lenet5 
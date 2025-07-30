#include <gtest/gtest.h>

#include <dnn/core/cuda.cuh>
#include <dnn/layers/conv_layer.cuh>

#include <vector>
#include <cmath>
#include <random>

namespace dnn {
namespace test {

class ConvLayerTest : public ::testing::Test {
protected:
    std::unique_ptr<Cuda> cuda_;

    void SetUp() override {
        // Always create a new context on this thread
        cuda_ = std::make_unique<Cuda>(0);
        cudaDeviceSynchronize();
    }

    void TearDown() override {
        cudaDeviceSynchronize();
        cuda_.reset();
    }

    // Helper function to compute expected output size
    std::vector<int> compute_output_shape(
        const std::vector<int>& input_shape,
        const std::vector<int>& kernel_size,
        int stride,
        int padding,
        int out_channels)
    {
        std::vector<int> output_shape(4);
        output_shape[0] = input_shape[0];              // batch size
        output_shape[1] = out_channels;

        output_shape[2] = (input_shape[2] + 2 * padding - kernel_size[0]) / stride + 1;
        output_shape[3] = (input_shape[3] + 2 * padding - kernel_size[1]) / stride + 1;

        return output_shape;
    }
};

TEST_F(ConvLayerTest, ConstructorAndInitialization) {
    int in_channels = 3;
    int out_channels = 4;
    std::vector<int> kernel_size = {3, 3};
    int stride = 1;
    int padding = 1;
    
    ConvLayer<float> layer(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        {},
        true);
    
    // Check if weights and bias are properly initialized
    auto params = layer.parameters();
    EXPECT_EQ(params.size(), 2); // weights and bias
    
    // Check weights shape
    EXPECT_EQ(params[0]->shape()[0], out_channels);
    EXPECT_EQ(params[0]->shape()[1], in_channels);
    EXPECT_EQ(params[0]->shape()[2], kernel_size[0]);
    EXPECT_EQ(params[0]->shape()[3], kernel_size[1]);
    
    // Check bias shape
    EXPECT_EQ(params[1]->shape()[1], out_channels);
}

TEST_F(ConvLayerTest, ForwardPass) {
    int in_channels = 2;
    int out_channels = 3;
    std::vector<int> kernel_size = {3, 3};
    int stride = 1;
    int padding = 1;
    
    ConvLayer<float> layer(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        {},
        true);
    
    // Create input tensor (batch_size=1, channels=2, height=4, width=4)
    std::vector<int> input_shape = {1, in_channels, 4, 4};
    tensor<float> input(input_shape);
    
    // Fill input with test data
    std::vector<float> input_data(static_cast<int>(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]));
    for (int i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i) / input_data.size();
    }
    input.upload(input_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    std::vector<int> expected_shape = compute_output_shape(input_shape, kernel_size, stride, padding, out_channels);
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
    int in_channels = 2;
    int out_channels = 3;
    std::vector<int> kernel_size = {3, 3};
    int stride = 1;
    int padding = 1;
    
    ConvLayer<float> layer(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        {},
        true);
    
    // Create input tensor
    std::vector<int> input_shape = {1, in_channels, 4, 4};
    tensor<float> input(input_shape);
    
    // Fill input with test data
    std::vector<float> input_data(static_cast<int>(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]));
    for (int i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i) / input_data.size();
    }
    input.upload(input_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Create gradient tensor
    tensor<float> grad_output(output.shape());
    std::vector<float> grad_data(output.size());
    for (int i = 0; i < grad_data.size(); ++i) {
        grad_data[i] = static_cast<float>(i) / grad_data.size();
    }
    grad_output.upload(grad_data.data());
    
    // Backward pass
    tensor<float> grad_input = layer.backward(grad_output);
    
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
    EXPECT_EQ(grads[1]->shape()[1], out_channels);
}

TEST_F(ConvLayerTest, DifferentStrideAndPadding) {
    int in_channels = 2;
    int out_channels = 3;
    std::vector<int> kernel_size = {3, 3};
    int stride = 2;
    int padding = 0;
    
    ConvLayer<float> layer(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        {},
        true);
    
    // Create input tensor (batch_size=1, channels=2, height=5, width=5)
    std::vector<int> input_shape = {1, in_channels, 5, 5};
    tensor<float> input(input_shape);
    
    // Fill input with test data
    std::vector<float> input_data(static_cast<int>(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]));
    for (int i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i) / input_data.size();
    }
    input.upload(input_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    std::vector<int> expected_shape = compute_output_shape(input_shape, kernel_size, stride, padding, out_channels);
    EXPECT_EQ(output.shape(), expected_shape);
    
    // Verify output dimensions
    EXPECT_EQ(output.shape()[2], 2); // (5 - 3) / 2 + 1 = 2
    EXPECT_EQ(output.shape()[3], 2); // (5 - 3) / 2 + 1 = 2
}

} // namespace test
} // namespace dnn 
#include <gtest/gtest.h>
#include "dnn/layers/activation/tanh_layer.cuh"
#include <vector>
#include <cmath>

namespace dnn {
namespace test {

class TanhLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }

    // Helper function to compute tanh
    float tanh_func(float x) {
        return std::tanh(x);
    }

    // Helper function to compute tanh derivative
    float tanh_derivative(float x) {
        float t = std::tanh(x);
        return 1.0f - t * t;
    }
};

TEST_F(TanhLayerTest, ForwardPass) {
    TanhLayer<float> layer;
    
    // Create input tensor
    std::vector<int> shape = {2, 2};
    tensor<float> input(shape);
    
    // Test data
    std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f};
    input.upload(input_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Download and verify results
    std::vector<float> output_data(4);
    output.download(output_data.data());
    
    for (int i = 0; i < input_data.size(); ++i) {
        float expected = tanh_func(input_data[i]);
        EXPECT_NEAR(output_data[i], expected, 1e-5);
    }
}

TEST_F(TanhLayerTest, BackwardPass) {
    TanhLayer<float> layer;
    
    // Create input and gradient tensors
    std::vector<int> shape = {2, 2};
    tensor<float> input(shape);
    tensor<float> grad_output(shape);
    
    // Test data
    std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f};
    std::vector<float> grad_data = {0.1f, 0.2f, 0.3f, 0.4f};
    
    input.upload(input_data.data());
    grad_output.upload(grad_data.data());
    
    // Backward pass
    tensor<float> grad_input = layer.backward(grad_output, input);
    
    // Download and verify results
    std::vector<float> grad_input_data(4);
    grad_input.download(grad_input_data.data());
    
    for (int i = 0; i < input_data.size(); ++i) {
        float expected = grad_data[i] * tanh_derivative(input_data[i]);
        EXPECT_NEAR(grad_input_data[i], expected, 1e-5);
    }
}

TEST_F(TanhLayerTest, EdgeCases) {
    TanhLayer<float> layer;
    
    // Create input tensor
    std::vector<int> shape = {1, 1};
    tensor<float> input(shape);
    
    // Test very large and very small values
    std::vector<float> input_data = {-100.0f, 100.0f};
    std::vector<float> output_data(1);
    
    // Test very negative value
    input.upload(&input_data[0]);
    tensor<float> output = layer.forward(input);
    output.download(output_data.data());
    EXPECT_NEAR(output_data[0], -1.0f, 1e-5);
    
    // Test very positive value
    input.upload(&input_data[1]);
    output = layer.forward(input);
    output.download(output_data.data());
    EXPECT_NEAR(output_data[0], 1.0f, 1e-5);
}

TEST_F(TanhLayerTest, ZeroInput) {
    TanhLayer<float> layer;
    
    // Create input tensor
    std::vector<int> shape = {1, 1};
    tensor<float> input(shape);
    
    // Test zero input
    std::vector<float> input_data = {0.0f};
    input.upload(input_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Download and verify results
    std::vector<float> output_data(1);
    output.download(output_data.data());
    
    EXPECT_NEAR(output_data[0], 0.0f, 1e-5);
    
    // Test backward pass with zero input
    tensor<float> grad_output(shape);
    std::vector<float> grad_data = {1.0f};
    grad_output.upload(grad_data.data());
    
    tensor<float> grad_input = layer.backward(grad_output, input);
    grad_input.download(output_data.data());
    
    // Derivative of tanh at x=0 is 1
    EXPECT_NEAR(output_data[0], 1.0f, 1e-5);
}

} // namespace test
} // namespace dnn 
#include <gtest/gtest.h>

#include <dnn/core/cuda.cuh>
#include <dnn/layers/fully_connected_layer.cuh>

#include <vector>
#include <cmath>
#include <random>

namespace dnn {
namespace test {

class FullyConnectedLayerTest : public ::testing::Test {
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

    // Helper function to compute expected output
    std::vector<float> compute_expected_output(
        const std::vector<float>& input,
        const std::vector<float>& weights,
        const std::vector<float>& bias,
        int in_features,
        int out_features) {
        
        std::vector<float> output(static_cast<size_t>(out_features));
        
        for (int i = 0; i < out_features; ++i) {
            output[i] = bias[i];
            for (int j = 0; j < in_features; ++j) {
                output[i] += input[j] * weights[i * in_features + j];
            }
        }
        
        return output;
    }

    // Helper function to compute expected gradients
    std::vector<float> compute_expected_gradients(
        const std::vector<float>& grad_output,
        const std::vector<float>& input,
        const std::vector<float>& weights,
        int in_features,
        int out_features) {
        
        std::vector<float> grad_input(static_cast<size_t>(in_features));
        std::vector<float> grad_weights(weights.size());
        std::vector<float> grad_bias(static_cast<size_t>(out_features));
        
        // Compute input gradients
        for (int i = 0; i < in_features; ++i) {
            grad_input[i] = 0.0f;
            for (int j = 0; j < out_features; ++j) {
                grad_input[i] += grad_output[j] * weights[j * in_features + i];
            }
        }
        
        // Compute weight gradients
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                grad_weights[i * in_features + j] = grad_output[i] * input[j];
            }
        }
        
        // Compute bias gradients
        for (int i = 0; i < out_features; ++i) {
            grad_bias[i] = grad_output[i];
        }
        
        return grad_input;
    }
};

TEST_F(FullyConnectedLayerTest, ConstructorAndInitialization) {
    int in_features = 4;
    int out_features = 3;
    
    const bool training_enabled = true;
    FullyConnectedLayer<float> layer(
        in_features,
        out_features,
        training_enabled);
    
    // Check if weights and bias are properly initialized
    auto params = layer.parameters();
    EXPECT_EQ(params.size(), 2); // weights and bias
    
    // Check weights shape
    EXPECT_EQ(params[0]->shape()[0], out_features);
    EXPECT_EQ(params[0]->shape()[1], in_features);
    
    // Check bias shape
    EXPECT_EQ(params[1]->shape()[1], out_features);
}

TEST_F(FullyConnectedLayerTest, ForwardPass) {
    int in_features = 4;
    int out_features = 3;
    
    const bool training_enabled = true;
    FullyConnectedLayer<float> layer(
        in_features,
        out_features,
        training_enabled);
    
    // Create input tensor
    std::vector<int> input_shape = {1, in_features};
    tensor<float> input(input_shape);
    
    // Fill input with test data
    std::vector<float> input_data(static_cast<size_t>(in_features));
    for (int i = 0; i < in_features; ++i) {
        input_data[i] = static_cast<float>(i) / in_features;
    }
    input.upload(input_data.data());
    
    // Get weights and bias
    auto params = layer.parameters();
    std::vector<float> weights_data(params[0]->size());
    std::vector<float> bias_data(params[1]->size());
    params[0]->download(weights_data.data());
    params[1]->download(bias_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Download and verify results
    std::vector<float> output_data(static_cast<size_t>(out_features));
    output.download(output_data.data());
    
    // Compute expected output
    std::vector<float> expected_output = compute_expected_output(
        input_data, weights_data, bias_data, in_features, out_features);
    
    for (int i = 0; i < out_features; ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-5);
    }
}

TEST_F(FullyConnectedLayerTest, BackwardPass) {
    int in_features = 4;
    int out_features = 3;
    
    const bool training_enabled = true;
    FullyConnectedLayer<float> layer(
        in_features,
        out_features,
        training_enabled);
    
    // Create input tensor
    std::vector<int> input_shape = {1, in_features};
    tensor<float> input(input_shape);
    
    // Fill input with test data
    std::vector<float> input_data(static_cast<size_t>(in_features));
    for (int i = 0; i < in_features; ++i) {
        input_data[i] = static_cast<float>(i) / in_features;
    }
    input.upload(input_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Create gradient tensor
    tensor<float> grad_output(output.shape());
    std::vector<float> grad_data(static_cast<size_t>(out_features));
    for (int i = 0; i < out_features; ++i) {
        grad_data[i] = static_cast<float>(i) / out_features;
    }
    grad_output.upload(grad_data.data());
    
    // Get weights
    auto params = layer.parameters();
    std::vector<float> weights_data(params[0]->size());
    params[0]->download(weights_data.data());
    
    // Backward pass
    tensor<float> grad_input = layer.backward(grad_output);
    
    // Download and verify results
    std::vector<float> grad_input_data(static_cast<size_t>(in_features));
    grad_input.download(grad_input_data.data());
    
    // Compute expected gradients
    std::vector<float> expected_gradients = compute_expected_gradients(
        grad_data, input_data, weights_data, in_features, out_features);
    
    for (int i = 0; i < in_features; ++i) {
        EXPECT_NEAR(grad_input_data[i], expected_gradients[i], 1e-5);
    }
    
    // Check if gradients are computed for parameters
    auto grads = layer.gradients();
    EXPECT_EQ(grads.size(), 2); // weights and bias gradients
    
    // Verify weight gradients shape
    EXPECT_EQ(grads[0]->shape()[0], out_features);
    EXPECT_EQ(grads[0]->shape()[1], in_features);
    
    // Verify bias gradients shape
    EXPECT_EQ(grads[1]->shape()[1], out_features);
}

TEST_F(FullyConnectedLayerTest, DifferentDimensions) {
    int in_features = 5;
    int out_features = 2;
    
    const bool training_enabled = true;
    FullyConnectedLayer<float> layer(
        in_features,
        out_features,
        training_enabled);
    
    // Create input tensor
    std::vector<int> input_shape = {1, in_features};
    tensor<float> input(input_shape);
    
    // Fill input with test data
    std::vector<float> input_data(static_cast<size_t>(in_features));
    for (int i = 0; i < in_features; ++i) {
        input_data[i] = static_cast<float>(i) / in_features;
    }
    input.upload(input_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], out_features);
    
    // Create gradient tensor
    tensor<float> grad_output(output.shape());
    std::vector<float> grad_data(static_cast<size_t>(out_features));
    for (int i = 0; i < out_features; ++i) {
        grad_data[i] = static_cast<float>(i) / out_features;
    }
    grad_output.upload(grad_data.data());
    
    // Backward pass
    tensor<float> grad_input = layer.backward(grad_output);
    
    // Verify gradient shapes
    EXPECT_EQ(grad_input.shape()[0], 1);
    EXPECT_EQ(grad_input.shape()[1], in_features);
}

TEST_F(FullyConnectedLayerTest, NonzeroInputGivesNonzeroOutput) {
    int in_features = 2, out_features = 2;
    const bool training_enabled = true;
    FullyConnectedLayer<float> layer(
        in_features,
        out_features,
        training_enabled);

    // Set weights and bias to known values
    auto params = layer.parameters();
    std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2
    std::vector<float> bias = {0.5f, -0.5f};
    params[0]->upload(weights.data());
    params[1]->upload(bias.data());

    // Input
    tensor<float> input({1, in_features});
    std::vector<float> input_data = {1.0f, 2.0f};
    input.upload(input_data.data());

    tensor<float> output = layer.forward(input);
    std::vector<float> output_data(out_features);
    output.download(output_data.data());

    // Output should not be all zeros
    for (float v : output_data)
        EXPECT_NE(v, 0.0f);
}

} // namespace test
} // namespace dnn 
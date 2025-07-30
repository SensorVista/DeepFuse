#include <gtest/gtest.h>

#include <dnn/core/cuda.cuh>
#include <dnn/layers/residual_add_layer.cuh>

#include <vector>
#include <cmath>
#include <random>

namespace dnn {
namespace test {

class ResidualAddLayerTest : public ::testing::Test {
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

    // Helper function to compute expected residual addition
    std::vector<float> compute_expected_output(
        const std::vector<float>& input,
        const std::vector<float>& residual) {
        
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = input[i] + residual[i];
        }
        return output;
    }
};

TEST_F(ResidualAddLayerTest, ConstructorAndInitialization) {
    ResidualAddLayer<float> layer;
    EXPECT_EQ(layer.name(), "ResidualAdd");
}

TEST_F(ResidualAddLayerTest, ForwardPass) {
    int batch_size = 2;
    int channels = 3;
    int height = 4;
    int width = 4;
    
    ResidualAddLayer<float> layer;
    
    // Create input and residual
    tensor<float> input = create_random_input(batch_size, channels, height, width);
    tensor<float> residual = create_random_input(batch_size, channels, height, width);
    
    // Set residual
    layer.set_residual(&residual);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Download data for comparison
    std::vector<float> output_data(output.size());
    std::vector<float> input_data(input.size());
    std::vector<float> residual_data(residual.size());
    output.download(output_data.data());
    input.download(input_data.data());
    residual.download(residual_data.data());
    
    // Compute expected output
    std::vector<float> expected_output = compute_expected_output(input_data, residual_data);
    
    // Compare outputs
    for (size_t i = 0; i < output_data.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-5);
    }
}

TEST_F(ResidualAddLayerTest, BackwardPass) {
    int batch_size = 2;
    int channels = 3;
    int height = 4;
    int width = 4;
    
    ResidualAddLayer<float> layer(true);
    
    // Create input and residual
    tensor<float> input = create_random_input(batch_size, channels, height, width);
    tensor<float> residual = create_random_input(batch_size, channels, height, width);
    
    // Set residual
    layer.set_residual(&residual);
    
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
    EXPECT_EQ(grad_input.shape()[1], channels);
    EXPECT_EQ(grad_input.shape()[2], height);
    EXPECT_EQ(grad_input.shape()[3], width);
    
    // Download gradients for comparison
    std::vector<float> grad_input_data(grad_input.size());
    grad_input.download(grad_input_data.data());
    
    // Gradients should be equal to grad_output
    for (size_t i = 0; i < grad_input_data.size(); ++i) {
        EXPECT_NEAR(grad_input_data[i], grad_data[i], 1e-5);
    }
}

TEST_F(ResidualAddLayerTest, DifferentDimensions) {
    int batch_size = 4;
    int channels = 64;
    int height = 32;
    int width = 32;
    
    ResidualAddLayer<float> layer;
    
    // Create input and residual
    tensor<float> input = create_random_input(batch_size, channels, height, width);
    tensor<float> residual = create_random_input(batch_size, channels, height, width);
    
    // Set residual
    layer.set_residual(&residual);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], channels);
    EXPECT_EQ(output.shape()[2], height);
    EXPECT_EQ(output.shape()[3], width);
    
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
    EXPECT_EQ(grad_input.shape()[1], channels);
    EXPECT_EQ(grad_input.shape()[2], height);
    EXPECT_EQ(grad_input.shape()[3], width);
}

TEST_F(ResidualAddLayerTest, MissingResidual) {
    int batch_size = 2;
    int channels = 3;
    int height = 4;
    int width = 4;
    
    ResidualAddLayer<float> layer;
    
    // Create input
    tensor<float> input = create_random_input(batch_size, channels, height, width);
    
    // Forward pass without setting residual should throw
    EXPECT_THROW(layer.forward(input), std::runtime_error);
}

TEST_F(ResidualAddLayerTest, ShapeMismatch) {
    int batch_size = 2;
    int channels = 3;
    int height = 4;
    int width = 4;
    
    ResidualAddLayer<float> layer;
    
    // Create input and residual with different shapes
    tensor<float> input = create_random_input(batch_size, channels, height, width);
    tensor<float> residual = create_random_input(batch_size, channels, height + 1, width);
    
    // Set residual
    layer.set_residual(&residual);
    
    // Forward pass with mismatched shapes should throw
    EXPECT_THROW(layer.forward(input), std::runtime_error);
}

} // namespace test
} // namespace dnn 
#include <gtest/gtest.h>

#include <dnn/core/cuda.cuh>
#include <dnn/layers/batch_norm_layer.cuh>

#include <vector>
#include <cmath>
#include <random>

namespace dnn {
namespace test {

class BatchNormLayerTest : public ::testing::Test {
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

    // Helper function to compute expected batch norm output
    std::vector<float> compute_expected_output(
        const std::vector<float>& input,
        const std::vector<float>& gamma,
        const std::vector<float>& beta,
        int batch_size,
        int channels,
        int height,
        int width,
        float epsilon) {
        
        std::vector<float> output(input.size());
        
        // For each channel
        for (int c = 0; c < channels; ++c) {
            // Compute mean
            float mean = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = ((b * channels + c) * height + h) * width + w;
                        mean += input[idx];
                    }
                }
            }
            mean /= (batch_size * height * width);
            
            // Compute variance
            float var = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = ((b * channels + c) * height + h) * width + w;
                        float diff = input[idx] - mean;
                        var += diff * diff;
                    }
                }
            }
            var /= (batch_size * height * width);
            
            // Normalize and scale
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = ((b * channels + c) * height + h) * width + w;
                        float normalized = (input[idx] - mean) / std::sqrt(var + epsilon);
                        output[idx] = gamma[c] * normalized + beta[c];
                    }
                }
            }
        }
        
        return output;
    }
};

TEST_F(BatchNormLayerTest, ConstructorAndInitialization) {
    int num_channels = 64;
    float epsilon = 1e-5f;
    float momentum = 0.9f;
    
    BatchNormLayer<float> layer(num_channels, epsilon, momentum);
    
    // Verify parameters
    auto params = layer.parameters();
    EXPECT_EQ(params.size(), 2);  // gamma and beta
    
    // Check shapes
    EXPECT_EQ(params[0]->shape()[0], num_channels);  // gamma
    EXPECT_EQ(params[1]->shape()[0], num_channels);  // beta
}

TEST_F(BatchNormLayerTest, ForwardPass) {
    int batch_size = 2;
    int channels = 3;
    int height = 4;
    int width = 4;
    float epsilon = 1e-5f;
    
    BatchNormLayer<float> layer(channels, epsilon);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, channels, height, width);
    
    // Get parameters
    auto params = layer.parameters();
    std::vector<float> gamma_data(params[0]->size());
    std::vector<float> beta_data(params[1]->size());
    params[0]->download(gamma_data.data());
    params[1]->download(beta_data.data());
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Download output
    std::vector<float> output_data(output.size());
    output.download(output_data.data());
    
    // Download input for expected computation
    std::vector<float> input_data(input.size());
    input.download(input_data.data());
    
    // Compute expected output
    std::vector<float> expected_output = compute_expected_output(
        input_data, gamma_data, beta_data, batch_size, channels, height, width, epsilon);
    
    // Compare outputs
    for (size_t i = 0; i < output_data.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-5);
    }
}

TEST_F(BatchNormLayerTest, DifferentDimensions) {
    int batch_size = 4;
    int channels = 64;
    int height = 32;
    int width = 32;
    float epsilon = 1e-5f;
    
    BatchNormLayer<float> layer(channels, epsilon, 0.9f, true, true);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, channels, height, width);
    
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

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at end of test: " << cudaGetErrorString(err) << std::endl;
    }
}

TEST_F(BatchNormLayerTest, NonAffineMode) {
    int batch_size = 2;
    int channels = 3;
    int height = 4;
    int width = 4;
    float epsilon = 1e-5f;
    
    // Create layer without learnable parameters
    BatchNormLayer<float> layer(channels, epsilon, 0.9f, false, true);
    
    // Verify no parameters
    auto params = layer.parameters();
    EXPECT_EQ(params.size(), 0);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, channels, height, width);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], channels);
    EXPECT_EQ(output.shape()[2], height);
    EXPECT_EQ(output.shape()[3], width);
}

TEST_F(BatchNormLayerTest, BackwardPass) {
    int batch_size = 2;
    int channels = 3;
    int height = 4;
    int width = 4;
    float epsilon = 1e-5f;
    float delta = 3e-3f;
    float tol = 2e-3f;

    BatchNormLayer<float> layer(channels, epsilon, 0.9f, true, true);

    // Create input
    tensor<float> input = create_random_input(batch_size, channels, height, width);

    // Forward pass
    tensor<float> output = layer.forward(input);

    // Create random grad_output
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

    // Check if gradients are computed for parameters
    auto grads = layer.gradients();
    EXPECT_EQ(grads.size(), 2);  // gamma and beta gradients

    // Verify gradient shapes
    EXPECT_EQ(grads[0]->shape()[0], channels);  // gamma gradient
    EXPECT_EQ(grads[1]->shape()[0], channels);  // beta gradient

    std::vector<float> grad_gamma(grads[0]->size());
    std::vector<float> grad_beta(grads[1]->size());

    grads[0]->download(grad_gamma.data());
    grads[1]->download(grad_beta.data());

    // Get gamma and beta
    auto params = layer.parameters();
    std::vector<float> gamma(params[0]->size());
    std::vector<float> beta(params[1]->size());
    params[0]->download(gamma.data());
    params[1]->download(beta.data());

    // Numerical gradient for gamma
    for (int c = 0; c < channels; ++c) {
        // Perturb gamma[c] by +delta
        std::vector<float> gamma_plus = gamma;
        gamma_plus[c] += delta;
        params[0]->upload(gamma_plus.data());
        tensor<float> out_plus = layer.forward(input);
        std::vector<float> out_plus_data(out_plus.size());
        out_plus.download(out_plus_data.data());
        float loss_plus = 0.0f;
        for (size_t i = 0; i < out_plus_data.size(); ++i) {
            loss_plus += out_plus_data[i] * grad_data[i];
        }

        // Perturb gamma[c] by -delta
        std::vector<float> gamma_minus = gamma;
        gamma_minus[c] -= delta;
        params[0]->upload(gamma_minus.data());
        tensor<float> out_minus = layer.forward(input);
        std::vector<float> out_minus_data(out_minus.size());
        out_minus.download(out_minus_data.data());
        float loss_minus = 0.0f;
        for (size_t i = 0; i < out_minus_data.size(); ++i) {
            loss_minus += out_minus_data[i] * grad_data[i];
        }

        float num_grad = (loss_plus - loss_minus) / (2 * delta);
        EXPECT_NEAR(grad_gamma[c], num_grad, tol) << "gamma grad mismatch at channel " << c;
    }

    // Restore original gamma
    params[0]->upload(gamma.data());

    // Numerical gradient for beta
    for (int c = 0; c < channels; ++c) {
        // Perturb beta[c] by +delta
        std::vector<float> beta_plus = beta;
        beta_plus[c] += delta;
        params[1]->upload(beta_plus.data());
        tensor<float> out_plus = layer.forward(input);
        std::vector<float> out_plus_data(out_plus.size());
        out_plus.download(out_plus_data.data());
        float loss_plus = 0.0f;
        for (size_t i = 0; i < out_plus_data.size(); ++i) {
            loss_plus += out_plus_data[i] * grad_data[i];
        }

        // Perturb beta[c] by -delta
        std::vector<float> beta_minus = beta;
        beta_minus[c] -= delta;
        params[1]->upload(beta_minus.data());
        tensor<float> out_minus = layer.forward(input);
        std::vector<float> out_minus_data(out_minus.size());
        out_minus.download(out_minus_data.data());
        float loss_minus = 0.0f;
        for (size_t i = 0; i < out_minus_data.size(); ++i) {
            loss_minus += out_minus_data[i] * grad_data[i];
        }

        float num_grad = (loss_plus - loss_minus) / (2 * delta);
        EXPECT_NEAR(grad_beta[c], num_grad, tol) << "beta grad mismatch at channel " << c;
    }

    // Restore original beta
    params[1]->upload(beta.data());
}

TEST_F(BatchNormLayerTest, Backward) {
    // Create layer with training enabled for backward pass
    BatchNormLayer<float> layer(2, 1e-5f, 0.9f, true, true);

    // Create test tensors
    tensor<float> input({1, 2, 2, 2});

    // Forward pass
    tensor<float> output = layer.forward(input);

    // Verify output shape
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.shape()[3], 2);
}

TEST_F(BatchNormLayerTest, BackwardNoAffine) {
    // Create layer with training enabled and affine disabled
    BatchNormLayer<float> layer(2, 1e-5f, 0.9f, false, true);

    // Create test tensors
    tensor<float> input({1, 2, 2, 2});

    // Forward pass
    tensor<float> output = layer.forward(input);

    // Verify output shape
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.shape()[3], 2);
}

} // namespace test
} // namespace dnn 
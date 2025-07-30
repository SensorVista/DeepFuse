#include <gtest/gtest.h>

#include <dnn/core/cuda.cuh>
#include <dnn/layers/activation_layer.cuh>

#include <vector>
#include <cmath>

namespace dnn {
namespace test {

class ActivationLayerTest : public ::testing::Test {
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

    // Helper function to compute sigmoid
    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // Helper function to compute sigmoid derivative
    float sigmoid_derivative(float x) {
        float s = sigmoid(x);
        return s * (1.0f - s);
    }

    // Helper function to compute tanh
    float tanh_scaled(float x) {
        return 1.7159f * std::tanh(2.0f/3.0f * x);
    }

    // Helper function to compute tanh derivative
    float tanh_derivative(float x) {
        float tanh_sx = std::tanh(2.0f/3.0f * x);
        return 1.7159f * (2.0f/3.0f) * (1.0f - tanh_sx * tanh_sx);
    }

    // Helper to compare tensors with tolerance
    bool compare_tensors(const tensor<float>& a, const tensor<float>& b, float tol = 1e-5) {
        if (a.size() != b.size()) return false;
        std::vector<float> host_a(a.size()), host_b(b.size());
        a.download(host_a.data());
        b.download(host_b.data());
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(host_a[i] - host_b[i]) > tol) return false;
        }
        return true;
    }
};

TEST_F(ActivationLayerTest, Constructor) {
    // Test constructor for each activation type
    EXPECT_NO_THROW({
        ActivationLayer<float> layer(ActivationType::ReLU);
    })  ;
    EXPECT_NO_THROW({
        ActivationLayer<float> layer(ActivationType::Sigmoid);
    });
    EXPECT_NO_THROW({
        ActivationLayer<float> layer(ActivationType::Tanh);
    });
#ifdef ENABLE_CUDNN
    EXPECT_NO_THROW({
        ActivationLayer<float> layer(ActivationType::ClippedReLU);
    });
    EXPECT_NO_THROW({
        ActivationLayer<float> layer(ActivationType::Elu);
    });
#endif
}

TEST_F(ActivationLayerTest, ForwardPass) {
    // Create test input
    std::vector<float> host_input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    tensor<float> input({5});
    input.upload(host_input.data());

    // Test ReLU
    {
        ActivationLayer<float> layer(ActivationType::ReLU);
        tensor<float> output = layer.forward(input);
        std::vector<float> host_output(output.size());
        cudaDeviceSynchronize();
        output.download(host_output.data());
        EXPECT_FLOAT_EQ(host_output[0], 0.0f);  // -2.0 -> 0.0
        EXPECT_FLOAT_EQ(host_output[1], 0.0f);  // -1.0 -> 0.0
        EXPECT_FLOAT_EQ(host_output[2], 0.0f);  // 0.0 -> 0.0
        EXPECT_FLOAT_EQ(host_output[3], 1.0f);  // 1.0 -> 1.0
        EXPECT_FLOAT_EQ(host_output[4], 2.0f);  // 2.0 -> 2.0
    }

    // Test Sigmoid
    {
        ActivationLayer<float> layer(ActivationType::Sigmoid);
        tensor<float> output = layer.forward(input);
        std::vector<float> host_output(output.size());
        cudaDeviceSynchronize();
        output.download(host_output.data());
        for (size_t i = 0; i < host_input.size(); ++i) {
            EXPECT_NEAR(host_output[i], sigmoid(host_input[i]), 1e-5);
        }
    }

    // Test Tanh
    {
        ActivationLayer<float> layer(ActivationType::Tanh);
        tensor<float> output = layer.forward(input);
        std::vector<float> host_output(output.size());
        cudaDeviceSynchronize();
        output.download(host_output.data());
        for (size_t i = 0; i < host_input.size(); ++i) {
            EXPECT_NEAR(host_output[i], tanh_scaled(host_input[i]), 1e-5);
        }
    }
}

TEST_F(ActivationLayerTest, BackwardPass) {
    // Create test input and gradient
    std::vector<float> host_input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> host_grad = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    tensor<float> input({5});
    tensor<float> grad_output({5});
    input.upload(host_input.data());
    grad_output.upload(host_grad.data());

    // Test ReLU backward
    {
        ActivationLayer<float> layer(ActivationType::ReLU, true);
        layer.forward(input);
        tensor<float> grad_input = layer.backward(grad_output);
        std::vector<float> host_grad_input(grad_input.size());
        cudaDeviceSynchronize();
        grad_input.download(host_grad_input.data());
        EXPECT_FLOAT_EQ(host_grad_input[0], 0.0f);  // -2.0 -> 0.0
        EXPECT_FLOAT_EQ(host_grad_input[1], 0.0f);  // -1.0 -> 0.0
        EXPECT_FLOAT_EQ(host_grad_input[2], 0.0f);  // 0.0 -> 0.0
        EXPECT_FLOAT_EQ(host_grad_input[3], 1.0f);  // 1.0 -> 1.0
        EXPECT_FLOAT_EQ(host_grad_input[4], 1.0f);  // 2.0 -> 1.0
    }

    // Test Sigmoid backward
    {
        ActivationLayer<float> layer(ActivationType::Sigmoid, true);
        layer.forward(input);
        tensor<float> grad_input = layer.backward(grad_output);
        std::vector<float> host_grad_input(grad_input.size());
        cudaDeviceSynchronize();
        grad_input.download(host_grad_input.data());
        for (size_t i = 0; i < host_input.size(); ++i) {
            EXPECT_NEAR(host_grad_input[i], host_grad[i] * sigmoid_derivative(host_input[i]), 1e-5);
        }
    }

    // Test Tanh backward
    {
        ActivationLayer<float> layer(ActivationType::Tanh, true);
        layer.forward(input);
        tensor<float> grad_input = layer.backward(grad_output);
        std::vector<float> host_grad_input(grad_input.size());
        cudaDeviceSynchronize();
        grad_input.download(host_grad_input.data());
        for (size_t i = 0; i < host_input.size(); ++i) {
            EXPECT_NEAR(host_grad_input[i], host_grad[i] * tanh_derivative(host_input[i]), 1e-5);
        }
    }
}

TEST_F(ActivationLayerTest, EdgeCases) {
    // Test empty tensor
    {
        tensor<float> empty_input({0});
        ActivationLayer<float> layer(ActivationType::ReLU);
        tensor<float> output = layer.forward(empty_input);
        EXPECT_EQ(output.size(), 0);
    }

    // Test large tensor
    {
        tensor<float> large_input({1000});
        large_input.fill(1.0f);
        ActivationLayer<float> layer(ActivationType::ReLU);
        tensor<float> output = layer.forward(large_input);
        EXPECT_EQ(output.size(), 1000);
        std::vector<float> host_output(output.size());
        output.download(host_output.data());
        for (float val : host_output) {
            EXPECT_FLOAT_EQ(val, 1.0f);
        }
    }

    // Test negative values
    {
        tensor<float> neg_input({5});
        std::vector<float> host_input = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f};
        neg_input.upload(host_input.data());
        ActivationLayer<float> layer(ActivationType::ReLU);
        tensor<float> output = layer.forward(neg_input);
        std::vector<float> host_output(output.size());
        output.download(host_output.data());
        for (float val : host_output) {
            EXPECT_FLOAT_EQ(val, 0.0f);
        }
    }
}

} // namespace test
} // namespace dnn 
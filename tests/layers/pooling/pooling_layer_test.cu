#include <gtest/gtest.h>
#include "dnn/layers/pooling/pooling_layer.cuh"
#include <vector>
#include <cmath>
#include <random>

namespace dnn {
namespace test {

class PoolingLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }

    // Helper function to create a test tensor with sequential values
    tensor<float> create_sequential_tensor(const std::vector<int>& shape) {
        tensor<float> t(shape);
        std::vector<float> host_data(t.size());
        for (size_t i = 0; i < host_data.size(); ++i) {
            host_data[i] = static_cast<float>(i);
        }
        t.upload(host_data.data());
        return t;
    }

    // Helper function to create a test tensor with random values
    tensor<float> create_random_tensor(const std::vector<int>& shape) {
        tensor<float> t(shape);
        std::vector<float> host_data(t.size());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (size_t i = 0; i < host_data.size(); ++i) {
            host_data[i] = dis(gen);
        }
        t.upload(host_data.data());
        return t;
    }

    // Helper function to compare tensors with tolerance
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

TEST_F(PoolingLayerTest, Constructor) {
    // Test constructor with valid parameters
    EXPECT_NO_THROW(PoolingLayer<float>(PoolingType::Max, 2, 2));
    EXPECT_NO_THROW(PoolingLayer<float>(PoolingType::Average, 3, 1));
#ifdef ENABLE_CUDNN
    EXPECT_NO_THROW(PoolingLayer<float>(PoolingType::MaxDeterministic, 2, 2));
#endif

    // Test constructor with invalid parameters
    EXPECT_THROW(PoolingLayer<float>(PoolingType::Max, 0, 2), std::invalid_argument);
    EXPECT_THROW(PoolingLayer<float>(PoolingType::Max, 2, 0), std::invalid_argument);
}

TEST_F(PoolingLayerTest, MaxPoolingForward) {
    // Create a 2x2x4x4 input tensor
    tensor<float> input = create_sequential_tensor({2, 2, 4, 4});
    
    // Create max pooling layer with 2x2 kernel and stride 2
    PoolingLayer<float> layer(PoolingType::Max, 2, 2);
    
    // Perform forward pass
    tensor<float> output = layer.forward(input);
    
    // Expected output shape: 2x2x2x2
    EXPECT_EQ(output.shape(), std::vector<int>({2, 2, 2, 2}));
    
    // Verify max pooling results
    std::vector<float> host_output(output.size());
    output.download(host_output.data());
    
    // Check first channel of first batch
    EXPECT_FLOAT_EQ(host_output[0], 5.0f);  // max of [0,1,4,5]
    EXPECT_FLOAT_EQ(host_output[1], 7.0f);  // max of [2,3,6,7]
    EXPECT_FLOAT_EQ(host_output[2], 13.0f); // max of [8,9,12,13]
    EXPECT_FLOAT_EQ(host_output[3], 15.0f); // max of [10,11,14,15]
}

TEST_F(PoolingLayerTest, AveragePoolingForward) {
    // Create a 2x2x4x4 input tensor
    tensor<float> input = create_sequential_tensor({2, 2, 4, 4});
    
    // Create average pooling layer with 2x2 kernel and stride 2
    PoolingLayer<float> layer(PoolingType::Average, 2, 2);
    
    // Perform forward pass
    tensor<float> output = layer.forward(input);
    
    // Expected output shape: 2x2x2x2
    EXPECT_EQ(output.shape(), std::vector<int>({2, 2, 2, 2}));
    
    // Verify average pooling results
    std::vector<float> host_output(output.size());
    output.download(host_output.data());
    
    // Check first channel of first batch
    EXPECT_FLOAT_EQ(host_output[0], 2.5f);  // avg of [0,1,4,5]
    EXPECT_FLOAT_EQ(host_output[1], 4.5f);  // avg of [2,3,6,7]
    EXPECT_FLOAT_EQ(host_output[2], 10.5f); // avg of [8,9,12,13]
    EXPECT_FLOAT_EQ(host_output[3], 12.5f); // avg of [10,11,14,15]
}

TEST_F(PoolingLayerTest, MaxPoolingBackward) {
    // Create input and gradient tensors
    tensor<float> input = create_sequential_tensor({2, 2, 4, 4});
    tensor<float> grad_output = create_sequential_tensor({2, 2, 2, 2});
    
    // Create max pooling layer
    PoolingLayer<float> layer(PoolingType::Max, 2, 2);
    
    // Perform backward pass
    tensor<float> grad_input = layer.backward(grad_output, input);
    
    // Verify gradient shape matches input shape
    EXPECT_EQ(grad_input.shape(), input.shape());
    
    // Verify gradients are properly distributed
    std::vector<float> host_grad_input(grad_input.size());
    grad_input.download(host_grad_input.data());
    
    // For max pooling, gradients should only flow to the maximum elements
    // in each pooling window
    EXPECT_FLOAT_EQ(host_grad_input[5], 0.0f); // Because grad_output[0] == 0.0f
    EXPECT_FLOAT_EQ(host_grad_input[7], 1.0f); // grad_output[1]
    EXPECT_FLOAT_EQ(host_grad_input[13], 2.0f); // grad_output[2]
    EXPECT_FLOAT_EQ(host_grad_input[15], 3.0f); // grad_output[3]
}

TEST_F(PoolingLayerTest, AveragePoolingBackward) {
    // Create input and gradient tensors
    tensor<float> input = create_sequential_tensor({2, 2, 4, 4});
    tensor<float> grad_output = create_sequential_tensor({2, 2, 2, 2});
    
    // Create average pooling layer
    PoolingLayer<float> layer(PoolingType::Average, 2, 2);
    
    // Perform backward pass
    tensor<float> grad_input = layer.backward(grad_output, input);
    
    // Verify gradient shape matches input shape
    EXPECT_EQ(grad_input.shape(), input.shape());
    
    // Verify gradients are evenly distributed
    std::vector<float> host_grad_input(grad_input.size());
    grad_input.download(host_grad_input.data());
    
    // For average pooling, gradients should be evenly distributed
    // across each pooling window
    float expected_grad = host_grad_input[0] / 4.0f; // 2x2 window
    EXPECT_FLOAT_EQ(host_grad_input[0], expected_grad);
    EXPECT_FLOAT_EQ(host_grad_input[1], expected_grad);
    EXPECT_FLOAT_EQ(host_grad_input[4], expected_grad);
    EXPECT_FLOAT_EQ(host_grad_input[5], expected_grad);
}

TEST_F(PoolingLayerTest, EdgeCases) {
    // Test with minimum valid dimensions
    {
        tensor<float> input = create_sequential_tensor({1, 1, 2, 2});
        PoolingLayer<float> layer(PoolingType::Max, 2, 2);
        tensor<float> output = layer.forward(input);
        EXPECT_EQ(output.shape(), std::vector<int>({1, 1, 1, 1}));
    }

    // Test with large dimensions
    {
        tensor<float> input = create_sequential_tensor({2, 3, 32, 32});
        PoolingLayer<float> layer(PoolingType::Max, 2, 2);
        tensor<float> output = layer.forward(input);
        EXPECT_EQ(output.shape(), std::vector<int>({2, 3, 16, 16}));
    }

    // Test with non-square kernel
    {
        tensor<float> input = create_sequential_tensor({1, 1, 4, 4});
        PoolingLayer<float> layer(PoolingType::Max, 2, 2);
        tensor<float> output = layer.forward(input);
        EXPECT_EQ(output.shape(), std::vector<int>({1, 1, 2, 2}));
    }
}

TEST_F(PoolingLayerTest, RandomInputs) {
    // Test with random inputs to ensure numerical stability
    for (int i = 0; i < 10; ++i) {
        tensor<float> input = create_random_tensor({2, 2, 8, 8});
        PoolingLayer<float> layer(PoolingType::Max, 2, 2);
        
        // Forward pass
        tensor<float> output = layer.forward(input);
        EXPECT_EQ(output.shape(), std::vector<int>({2, 2, 4, 4}));
        
        // Backward pass
        tensor<float> grad_output = create_random_tensor({2, 2, 4, 4});
        tensor<float> grad_input = layer.backward(grad_output, input);
        EXPECT_EQ(grad_input.shape(), input.shape());
    }
}

#ifdef ENABLE_CUDNN
TEST_F(PoolingLayerTest, MaxDeterministicPooling) {
    // Test deterministic max pooling
    tensor<float> input = create_sequential_tensor({2, 2, 4, 4});
    PoolingLayer<float> layer(PoolingType::MaxDeterministic, 2, 2);
    
    // Perform forward pass
    tensor<float> output = layer.forward(input);
    EXPECT_EQ(output.shape(), std::vector<int>({2, 2, 2, 2}));
    
    // Verify deterministic results
    tensor<float> output2 = layer.forward(input);
    EXPECT_TRUE(compare_tensors(output, output2));
}
#endif

} // namespace test
} // namespace dnn
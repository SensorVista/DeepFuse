#include <gtest/gtest.h>
#include "dnn/losses/cross_entropy.cuh"
#include <vector>
#include <cmath>
#include <numeric>
#include <random>

namespace dnn {
namespace test {

class CrossEntropyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }

    // Helper function to compute softmax
    std::vector<float> softmax(const std::vector<float>& x) {
        std::vector<float> exp_x(x.size());
        float max_x = *std::max_element(x.begin(), x.end());
        
        // Compute exp(x - max_x) for numerical stability
        for (size_t i = 0; i < x.size(); ++i) {
            exp_x[i] = std::exp(x[i] - max_x);
        }
        
        float sum = std::accumulate(exp_x.begin(), exp_x.end(), 0.0f);
        for (float& val : exp_x) {
            val /= sum;
        }
        
        return exp_x;
    }

    // Helper function to compute cross entropy loss
    float compute_cross_entropy(const std::vector<float>& predictions, const std::vector<float>& targets) {
        std::vector<float> softmax_pred = softmax(predictions);
        float loss = 0.0f;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (targets[i] > 0.0f) {
                loss -= targets[i] * std::log(std::max(softmax_pred[i], 1e-7f));
            }
        }
        
        return loss;
    }

    // Helper function to compute cross entropy gradient
    std::vector<float> compute_cross_entropy_gradient(const std::vector<float>& predictions, const std::vector<float>& targets) {
        std::vector<float> softmax_pred = softmax(predictions);
        std::vector<float> gradients(predictions.size());
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            gradients[i] = softmax_pred[i] - targets[i];
        }
        
        return gradients;
    }
};

TEST_F(CrossEntropyTest, ForwardPass) {
    CrossEntropyLoss<float> loss;
    
    // Create prediction and target tensors
    std::vector<int> shape = {1, 4};
    tensor<float> predictions(shape);
    tensor<float> targets(shape);
    
    // Test data
    std::vector<float> pred_data = {2.0f, 1.0f, 0.1f, 0.5f};
    std::vector<float> target_data = {1.0f, 0.0f, 0.0f, 0.0f}; // One-hot encoded
    
    predictions.upload(pred_data.data());
    targets.upload(target_data.data());
    
    // Compute loss
    float computed_loss = loss.compute(predictions, targets);
    
    // Compute expected loss
    float expected_loss = compute_cross_entropy(pred_data, target_data);
    
    EXPECT_NEAR(computed_loss, expected_loss, 1e-5);
}

TEST_F(CrossEntropyTest, BackwardPass) {
    CrossEntropyLoss<float> loss;
    
    // Create prediction and target tensors
    std::vector<int> shape = {1, 4};
    tensor<float> predictions(shape);
    tensor<float> targets(shape);
    
    // Test data
    std::vector<float> pred_data = {2.0f, 1.0f, 0.1f, 0.5f};
    std::vector<float> target_data = {1.0f, 0.0f, 0.0f, 0.0f}; // One-hot encoded
    
    predictions.upload(pred_data.data());
    targets.upload(target_data.data());
    
    // Compute gradients
    tensor<float> gradients = loss.compute_gradient(predictions, targets);
    
    // Download and verify results
    std::vector<float> gradient_data(4);
    gradients.download(gradient_data.data());
    
    // Compute expected gradients
    std::vector<float> expected_gradients = compute_cross_entropy_gradient(pred_data, target_data);
    
    for (size_t i = 0; i < gradient_data.size(); ++i) {
        EXPECT_NEAR(gradient_data[i], expected_gradients[i], 1e-5);
    }
}

TEST_F(CrossEntropyTest, EdgeCases) {
    CrossEntropyLoss<float> loss;
    
    // Create prediction and target tensors
    std::vector<int> shape = {1, 4};
    tensor<float> predictions(shape);
    tensor<float> targets(shape);
    
    // Test edge cases
    std::vector<float> pred_data = {100.0f, -100.0f, 0.0f, 0.0f};
    std::vector<float> target_data = {1.0f, 0.0f, 0.0f, 0.0f}; // One-hot encoded
    
    predictions.upload(pred_data.data());
    targets.upload(target_data.data());
    
    // Compute loss
    float computed_loss = loss.compute(predictions, targets);
    
    // Verify loss is finite
    EXPECT_FALSE(std::isnan(computed_loss));
    EXPECT_FALSE(std::isinf(computed_loss));
    
    // Compute gradients
    tensor<float> gradients = loss.compute_gradient(predictions, targets);
    
    // Download and verify results
    std::vector<float> gradient_data(4);
    gradients.download(gradient_data.data());
    
    // Verify gradients are finite
    for (float grad : gradient_data) {
        EXPECT_FALSE(std::isnan(grad));
        EXPECT_FALSE(std::isinf(grad));
    }
}

TEST_F(CrossEntropyTest, MultipleClasses) {
    CrossEntropyLoss<float> loss;
    
    // Create prediction and target tensors
    std::vector<int> shape = {1, 10}; // 10 classes
    tensor<float> predictions(shape);
    tensor<float> targets(shape);
    
    // Test data
    std::vector<float> pred_data(10);
    std::vector<float> target_data(10, 0.0f);
    
    // Generate random predictions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < 10; ++i) {
        pred_data[i] = dis(gen);
    }
    
    // Set target class (one-hot encoded)
    target_data[5] = 1.0f;
    
    predictions.upload(pred_data.data());
    targets.upload(target_data.data());
    
    // Compute loss
    float computed_loss = loss.compute(predictions, targets);
    
    // Compute expected loss
    float expected_loss = compute_cross_entropy(pred_data, target_data);
    
    EXPECT_NEAR(computed_loss, expected_loss, 1e-5);
    
    // Compute gradients
    tensor<float> gradients = loss.compute_gradient(predictions, targets);
    
    // Download and verify results
    std::vector<float> gradient_data(10);
    gradients.download(gradient_data.data());
    
    // Compute expected gradients
    std::vector<float> expected_gradients = compute_cross_entropy_gradient(pred_data, target_data);
    
    for (size_t i = 0; i < gradient_data.size(); ++i) {
        EXPECT_NEAR(gradient_data[i], expected_gradients[i], 1e-5);
    }
}

} // namespace test
} // namespace dnn 
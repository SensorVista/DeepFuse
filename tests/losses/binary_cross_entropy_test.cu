#include <gtest/gtest.h>
#include "dnn/losses/binary_cross_entropy.cuh"
#include <vector>
#include <cmath>

namespace dnn {
namespace test {

class BinaryCrossEntropyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }

    // Helper function to compute binary cross entropy
    float compute_bce(const std::vector<float>& predictions, const std::vector<float>& targets) {
        float loss = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float p = std::max(std::min(predictions[i], 1.0f - 1e-7f), 1e-7f);
            loss -= targets[i] * std::log(p) + (1.0f - targets[i]) * std::log(1.0f - p);
        }
        return loss / predictions.size();
    }

    // Helper function to compute binary cross entropy gradient
    std::vector<float> compute_bce_gradient(const std::vector<float>& predictions, const std::vector<float>& targets) {
        std::vector<float> gradients(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            float p = std::max(std::min(predictions[i], 1.0f - 1e-7f), 1e-7f);
            gradients[i] = (p - targets[i]) / (p * (1.0f - p));
        }
        return gradients;
    }
};

TEST_F(BinaryCrossEntropyTest, ForwardPass) {
    BinaryCrossEntropyLoss<float> loss;
    
    // Create prediction and target tensors
    std::vector<int> shape = {1, 4};
    tensor<float> predictions(shape);
    tensor<float> targets(shape);
    
    // Test data
    std::vector<float> pred_data = {0.1f, 0.5f, 0.8f, 0.9f};
    std::vector<float> target_data = {0.0f, 0.5f, 1.0f, 1.0f};
    
    predictions.upload(pred_data.data());
    targets.upload(target_data.data());
    
    // Compute loss
    float computed_loss = loss.compute(predictions, targets);
    
    // Compute expected loss
    float expected_loss = compute_bce(pred_data, target_data);
    
    EXPECT_NEAR(computed_loss, expected_loss, 1e-5);
}

TEST_F(BinaryCrossEntropyTest, BackwardPass) {
    BinaryCrossEntropyLoss<float> loss;
    
    // Create prediction and target tensors
    std::vector<int> shape = {1, 4};
    tensor<float> predictions(shape);
    tensor<float> targets(shape);
    
    // Test data
    std::vector<float> pred_data = {0.1f, 0.5f, 0.8f, 0.9f};
    std::vector<float> target_data = {0.0f, 0.5f, 1.0f, 1.0f};
    
    predictions.upload(pred_data.data());
    targets.upload(target_data.data());
    
    // Compute gradients
    tensor<float> gradients = loss.compute_gradient(predictions, targets);
    
    // Download and verify results
    std::vector<float> gradient_data(4);
    gradients.download(gradient_data.data());
    
    // Compute expected gradients
    std::vector<float> expected_gradients = compute_bce_gradient(pred_data, target_data);
    
    for (size_t i = 0; i < gradient_data.size(); ++i) {
        EXPECT_NEAR(gradient_data[i], expected_gradients[i], 1e-5);
    }
}

TEST_F(BinaryCrossEntropyTest, EdgeCases) {
    BinaryCrossEntropyLoss<float> loss;
    
    // Create prediction and target tensors
    std::vector<int> shape = {1, 4};
    tensor<float> predictions(shape);
    tensor<float> targets(shape);
    
    // Test edge cases
    std::vector<float> pred_data = {0.0f, 1.0f, 0.5f, 0.5f};
    std::vector<float> target_data = {0.0f, 1.0f, 0.0f, 1.0f};
    
    predictions.upload(pred_data.data());
    targets.upload(target_data.data());
    
    // Compute loss
    float computed_loss = loss.compute(predictions, targets);
    
    // Compute expected loss
    float expected_loss = compute_bce(pred_data, target_data);
    
    EXPECT_NEAR(computed_loss, expected_loss, 1e-5);
    
    // Compute gradients
    tensor<float> gradients = loss.compute_gradient(predictions, targets);
    
    // Download and verify results
    std::vector<float> gradient_data(4);
    gradients.download(gradient_data.data());
    
    // Compute expected gradients
    std::vector<float> expected_gradients = compute_bce_gradient(pred_data, target_data);
    
    for (size_t i = 0; i < gradient_data.size(); ++i) {
        EXPECT_NEAR(gradient_data[i], expected_gradients[i], 1e-5);
    }
}

TEST_F(BinaryCrossEntropyTest, NumericalStability) {
    BinaryCrossEntropyLoss<float> loss;
    
    // Create prediction and target tensors
    std::vector<int> shape = {1, 4};
    tensor<float> predictions(shape);
    tensor<float> targets(shape);
    
    // Test with values very close to 0 and 1
    std::vector<float> pred_data = {1e-7f, 1.0f - 1e-7f, 0.5f, 0.5f};
    std::vector<float> target_data = {0.0f, 1.0f, 0.0f, 1.0f};
    
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

} // namespace test
} // namespace dnn 
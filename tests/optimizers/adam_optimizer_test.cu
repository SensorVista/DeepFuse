#include "dnn/optimizers/adam_optimizer.cuh"
#include "dnn/core/tensor.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace dnn {
namespace test {

class AdamOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }
};

TEST_F(AdamOptimizerTest, BasicUpdate) {
    // Create test parameters and gradients
    tensor<float> param1({ 2, 2 });
    tensor<float> param2({ 2, 2 });
    tensor<float> grad1({ 2, 2 });
    tensor<float> grad2({ 2, 2 });

    // Initialize with test values
    param1.fill(1.0f);
    param2.fill(2.0f);
    grad1.fill(0.1f);
    grad2.fill(0.2f);

    // Create optimizer with default parameters
    AdamOptimizer<float> optimizer(0.001f); // learning rate = 0.001

    // Store initial values
    std::vector<float> initial_params1(4), initial_params2(4);
    param1.download(initial_params1.data());
    param2.download(initial_params2.data());

    // Update parameters
    optimizer.update_parameters({ &param1, &param2 }, { &grad1, &grad2 });
    optimizer.step();

    // Download updated values
    std::vector<float> updated_params1(4), updated_params2(4);
    param1.download(updated_params1.data());
    param2.download(updated_params2.data());

    // Check that parameters were updated (direction of update)
    EXPECT_LT(updated_params1[0], initial_params1[0]);
    EXPECT_LT(updated_params2[0], initial_params2[0]);
}

TEST_F(AdamOptimizerTest, MomentumAndRMS) {
    // Create test parameters and gradients
    tensor<float> param({ 2, 2 });
    tensor<float> grad({ 2, 2 });

    // Initialize with test values
    param.fill(1.0f);
    grad.fill(0.1f);

    // Create optimizer with specific beta values
    AdamOptimizer<float> optimizer(0.001f, 0.9f, 0.999f);

    // First step
    optimizer.update_parameters({ &param }, { &grad });
    optimizer.step();

    std::vector<float> first_step_param(4);
    param.download(first_step_param.data());

    // Second step with same gradient
    optimizer.step();

    std::vector<float> second_step_param(4);
    param.download(second_step_param.data());

    // Check that the second update is different from the first
    // due to momentum and RMS accumulation
    EXPECT_NE(first_step_param[0], second_step_param[0]);
}

TEST_F(AdamOptimizerTest, WeightDecay) {
    // Create test parameters and gradients
    tensor<float> param({ 2, 2 });
    tensor<float> grad({ 2, 2 });

    // Initialize with test values
    param.fill(1.0f);
    grad.fill(0.1f);

    // Create optimizer with weight decay
    AdamOptimizer<float> optimizer(0.001f, 0.9f, 0.999f, 1e-8f, 0.1f);

    std::vector<float> initial_param(4);
    param.download(initial_param.data());

    // Update parameters
    optimizer.update_parameters({ &param }, { &grad });
    optimizer.step();

    std::vector<float> updated_param(4);
    param.download(updated_param.data());

    // Check that weight decay had an effect
    // The update should be larger than without weight decay
    float update_with_decay = initial_param[0] - updated_param[0];
    EXPECT_GT(update_with_decay, 0.0f);
}

TEST_F(AdamOptimizerTest, ZeroGrad) {
    // Create test parameters and gradients
    tensor<float> param({ 2, 2 });
    tensor<float> grad({ 2, 2 });

    // Initialize with test values
    param.fill(1.0f);
    grad.fill(0.1f);

    // Create optimizer
    AdamOptimizer<float> optimizer(0.001f);

    // Zero gradients
    optimizer.update_parameters({ &param }, { &grad });
    optimizer.zero_grad();

    // Check all gradients are zero
    std::vector<float> grad_data(4);
    grad.download(grad_data.data());
    for (int i = 0; i < grad.size(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 0.0f);
    }
}

TEST_F(AdamOptimizerTest, LearningRateChange) {
    // Create test parameters and gradients
    tensor<float> param({ 2, 2 });
    tensor<float> grad({ 2, 2 });

    // Initialize with test values
    param.fill(1.0f);
    grad.fill(0.1f);

    // Create optimizer
    AdamOptimizer<float> optimizer(0.001f);

    // First step with learning rate 0.001
    optimizer.update_parameters({ &param }, { &grad });
    optimizer.step();

    std::vector<float> first_step_param(4);
    param.download(first_step_param.data());

    // Change learning rate to 0.002
    optimizer.set_learning_rate(0.002f);

    // Second step with learning rate 0.002
    optimizer.step();

    std::vector<float> second_step_param(4);
    param.download(second_step_param.data());

    // Check that the second update is larger than the first
    float first_update = first_step_param[0] - 1.0f;
    float second_update = second_step_param[0] - first_step_param[0];
    EXPECT_GT(std::abs(second_update), std::abs(first_update));
}

TEST_F(AdamOptimizerTest, ParameterUpdate) {
    // Create test parameters and gradients
    tensor<float> param1({ 2, 2 });
    tensor<float> param2({ 2, 2 });
    tensor<float> grad1({ 2, 2 });
    tensor<float> grad2({ 2, 2 });

    // Initialize with test values
    param1.fill(1.0f);
    param2.fill(2.0f);
    grad1.fill(0.1f);
    grad2.fill(0.2f);

    // Create optimizer
    AdamOptimizer<float> optimizer(0.001f);

    // Initial update
    optimizer.update_parameters({ &param1 }, { &grad1 });
    optimizer.step();

    std::vector<float> first_step_param1(4);
    param1.download(first_step_param1.data());

    // Create new parameters and gradients
    tensor<float> new_param({ 2, 2 });
    tensor<float> new_grad({ 2, 2 });

    new_param.fill(3.0f);
    new_grad.fill(0.3f);

    // Update with new parameters
    optimizer.update_parameters({ &new_param }, { &new_grad });
    optimizer.step();

    // Check that old parameters weren't affected
    std::vector<float> old_param1(4);
    param1.download(old_param1.data());
    EXPECT_FLOAT_EQ(old_param1[0], first_step_param1[0]);

    // Check that new parameters were updated
    std::vector<float> new_param_data(4);
    new_param.download(new_param_data.data());
    EXPECT_NE(new_param_data[0], 3.0f);
}

} // namespace test
} // namespace dnn 
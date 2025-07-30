#include "dnn/optimizers/sgd_optimizer.cuh"
#include "dnn/core/tensor.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace dnn {
namespace test {

class SGDOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }
};

TEST_F(SGDOptimizerTest, BasicUpdate) {
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

    // Create optimizer and update parameters
    SGDOptimizer<float> optimizer(0.1f); // learning rate = 0.1

    // Store initial values
    std::vector<float> initial_params1(4), initial_params2(4);
    param1.download(initial_params1.data());
    param2.download(initial_params2.data());

    // Update both parameters at once
    optimizer.update_parameters({ &param1, &param2 }, { &grad1, &grad2 });
    optimizer.step();

    // Download updated values
    std::vector<float> updated_params1(4), updated_params2(4);
    param1.download(updated_params1.data());
    param2.download(updated_params2.data());

    // Check parameter updates
    // param = param - lr * grad
    EXPECT_NEAR(updated_params1[0], initial_params1[0] - 0.1f * 0.1f, 1e-5);
    EXPECT_NEAR(updated_params2[0], initial_params2[0] - 0.1f * 0.2f, 1e-5);
}


TEST_F(SGDOptimizerTest, Momentum) {
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

    // Create optimizer with momentum
    SGDOptimizer<float> optimizer(0.1f, 0.9f); // learning rate = 0.1, momentum = 0.9

    // First step: update both parameters at once
    optimizer.update_parameters({ &param1, &param2 }, { &grad1, &grad2 });
    optimizer.step();

    std::vector<float> first_step_params1(4), first_step_params2(4);
    param1.download(first_step_params1.data());
    param2.download(first_step_params2.data());

    // Second step: update both again
    optimizer.update_parameters({ &param1, &param2 }, { &grad1, &grad2 });
    optimizer.step();

    // Download final values
    std::vector<float> final_params1(4), final_params2(4);
    param1.download(final_params1.data());
    param2.download(final_params2.data());

    // Manually compute expected values
    float v1 = -0.1f * 0.1f;
    float v1_next = 0.9f * v1 - 0.1f * 0.1f;
    float expected1 = 1.0f + v1 + v1_next;

    float v2 = -0.1f * 0.2f;
    float v2_next = 0.9f * v2 - 0.1f * 0.2f;
    float expected2 = 2.0f + v2 + v2_next;

    EXPECT_NEAR(final_params1[0], expected1, 1e-5);
    EXPECT_NEAR(final_params2[0], expected2, 1e-5);
}

TEST_F(SGDOptimizerTest, WeightDecay) {
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

    // Create optimizer with weight decay
    SGDOptimizer<float> optimizer(0.1f, 0.0f, 0.1f); // learning rate = 0.1, weight decay = 0.1

    std::vector<float> initial_params1(4), initial_params2(4);
    param1.download(initial_params1.data());
    param2.download(initial_params2.data());

    // Update parameters one at a time
    optimizer.update_parameters({ &param1 }, { &grad1 });
    optimizer.step();
    optimizer.update_parameters({ &param2 }, { &grad2 });
    optimizer.step();

    std::vector<float> final_params1(4), final_params2(4);
    param1.download(final_params1.data());
    param2.download(final_params2.data());

    // Check weight decay effect on first element
    float expected_grad1 = 0.1f + 0.1f * initial_params1[0];
    float expected_grad2 = 0.2f + 0.1f * initial_params2[0];

    EXPECT_NEAR(final_params1[0], initial_params1[0] - 0.1f * expected_grad1, 1e-5);
    EXPECT_NEAR(final_params2[0], initial_params2[0] - 0.1f * expected_grad2, 1e-5);
}

TEST_F(SGDOptimizerTest, ZeroGrad) {
    // Create test parameters and gradients
    tensor<float> param1({2, 2});
    tensor<float> param2({2, 2});
    tensor<float> grad1({2, 2});
    tensor<float> grad2({2, 2});

    // Initialize with test values
    param1.fill(1.0f);
    param2.fill(2.0f);
    grad1.fill(0.1f);
    grad2.fill(0.2f);

    // Create optimizer
    SGDOptimizer<float> optimizer(0.1f);

    // Zero gradients one at a time
    optimizer.update_parameters({&param1}, {&grad1});
    optimizer.zero_grad();
    optimizer.update_parameters({&param2}, {&grad2});
    optimizer.zero_grad();

    // Check all gradients are zero
    std::vector<float> grad_data(4);
    grad1.download(grad_data.data());
    for (int i = 0; i < grad1.size(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 0.0f);
    }
    grad2.download(grad_data.data());
    for (int i = 0; i < grad2.size(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 0.0f);
    }
}

TEST_F(SGDOptimizerTest, LearningRateChange) {
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
    SGDOptimizer<float> optimizer(0.1f);

    std::vector<float> initial_params1(4), initial_params2(4);
    param1.download(initial_params1.data());
    param2.download(initial_params2.data());

    // First step with learning rate 0.1
    optimizer.update_parameters({ &param1 }, { &grad1 });
    optimizer.step();
    optimizer.update_parameters({ &param2 }, { &grad2 });
    optimizer.step();

    std::vector<float> first_step_params1(4), first_step_params2(4);
    param1.download(first_step_params1.data());
    param2.download(first_step_params2.data());

    // Change learning rate to 0.2
    optimizer.set_learning_rate(0.2f);

    // Second step with learning rate 0.2
    optimizer.update_parameters({ &param1 }, { &grad1 });
    optimizer.step();
    optimizer.update_parameters({ &param2 }, { &grad2 });
    optimizer.step();

    std::vector<float> final_params1(4), final_params2(4);
    param1.download(final_params1.data());
    param2.download(final_params2.data());

    // Check that the second update is approximately twice the first
    float update1 = first_step_params1[0] - initial_params1[0];
    float update2 = first_step_params2[0] - initial_params2[0];

    EXPECT_NEAR(final_params1[0] - first_step_params1[0], 2.0f * update1, 1e-5);
    EXPECT_NEAR(final_params2[0] - first_step_params2[0], 2.0f * update2, 1e-5);
}

TEST_F(SGDOptimizerTest, ParameterUpdate) {
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
    SGDOptimizer<float> optimizer(0.1f);

    // Initial update
    optimizer.update_parameters({ &param1 }, { &grad1 });
    optimizer.step();
    optimizer.update_parameters({ &param2 }, { &grad2 });
    optimizer.step();

    std::vector<float> first_step_params1(4), first_step_params2(4);
    param1.download(first_step_params1.data());
    param2.download(first_step_params2.data());

    // Create new parameters and gradients
    tensor<float> new_param1({ 2, 2 });
    tensor<float> new_param2({ 2, 2 });
    tensor<float> new_grad1({ 2, 2 });
    tensor<float> new_grad2({ 2, 2 });

    new_param1.fill(3.0f);
    new_param2.fill(4.0f);
    new_grad1.fill(0.3f);
    new_grad2.fill(0.4f);

    // Update with new parameters
    optimizer.update_parameters({ &new_param1 }, { &new_grad1 });
    optimizer.step();
    optimizer.update_parameters({ &new_param2 }, { &new_grad2 });
    optimizer.step();

    // Check that old parameters weren't affected
    std::vector<float> old_params1(4), old_params2(4);
    param1.download(old_params1.data());
    param2.download(old_params2.data());
    EXPECT_FLOAT_EQ(old_params1[0], first_step_params1[0]);
    EXPECT_FLOAT_EQ(old_params2[0], first_step_params2[0]);

    // Check that new parameters were updated
    std::vector<float> new_params1(4), new_params2(4);
    new_param1.download(new_params1.data());
    new_param2.download(new_params2.data());
    EXPECT_NEAR(new_params1[0], 3.0f - 0.1f * 0.3f, 1e-5);
    EXPECT_NEAR(new_params2[0], 4.0f - 0.1f * 0.4f, 1e-5);
}

} // namespace test
} // namespace dnn

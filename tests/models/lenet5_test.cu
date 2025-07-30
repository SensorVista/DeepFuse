#include <gtest/gtest.h>

#include <dnn/models/lenet5.cuh>

#include <vector>
#include <random>
#include <fstream>
#include <cstdio>
#include <filesystem>

namespace dnn {
namespace test {

class LeNet5Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }

    // Helper function to generate random input data
    std::vector<float> generate_random_input(size_t size) {
        std::vector<float> data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        for (size_t i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
        return data;
    }

    // Helper function to generate random target data
    std::vector<float> generate_random_target(size_t size) {
        std::vector<float> data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 9);
        
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<float>(dis(gen));
        }
        return data;
    }
};

TEST_F(LeNet5Test, Constructor) {
    LeNet5<float> model;
    
    // Verify model architecture
    auto layers = model.parameters();
    EXPECT_GT(layers.size(), 0);
}

TEST_F(LeNet5Test, ForwardPass) {
    LeNet5<float> model;
    
    // Create input tensor (batch_size=1, channels=1, height=32, width=32)
    std::vector<int> input_shape = {1, 1, 32, 32};
    tensor<float> input(input_shape);
    
    // Fill input with random data
    std::vector<float> input_data = generate_random_input(static_cast<size_t>(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]));
    input.upload(input_data.data());
    
    // Forward pass
    tensor<float> output = model.forward(input);
    
    // Verify output shape (should be [1, 10] for 10 classes)
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 10);
    
    // Download and verify output values
    std::vector<float> output_data(output.size());
    output.download(output_data.data());
    
    // Check if output values are reasonable (not NaN or Inf)
    for (float val : output_data) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_F(LeNet5Test, TrainingStep) {
    LeNet5<float> model(0.01, 0.9, true);
    
    // Create input tensor
    std::vector<int> input_shape = {1, 1, 32, 32};
    tensor<float> input(input_shape);
    
    // Create target tensor
    std::vector<int> target_shape = {1, 10};
    tensor<float> target(target_shape);
    
    // Fill input with random data
    std::vector<float> input_data = generate_random_input(static_cast<size_t>(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]));
    input.upload(input_data.data());
    
    // Fill target with random one-hot encoded data
    std::vector<float> target_data = generate_random_target(static_cast<size_t>(target_shape[1]));
    target.upload(target_data.data());
    
    // Record initial loss
    float initial_loss = model.loss();
    
    // Perform training step
    model.train_step(input, target);
    
    // Record new loss
    float new_loss = model.loss();
    
    // Verify that loss has changed
    EXPECT_NE(initial_loss, new_loss);
}

TEST_F(LeNet5Test, MultipleTrainingSteps) {
    LeNet5<float> model(0.01, 0.9, true);
    
    // Create input tensor
    std::vector<int> input_shape = {1, 1, 32, 32};
    tensor<float> input(input_shape);
    
    // Create target tensor
    std::vector<int> target_shape = {1, 10};
    tensor<float> target(target_shape);
    
    // Perform multiple training steps
    std::vector<float> losses;
    for (int i = 0; i < 5; ++i) {
        // Fill input with random data
        std::vector<float> input_data = generate_random_input(static_cast<size_t>(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]));
        input.upload(input_data.data());
        
        // Fill target with random one-hot encoded data
        std::vector<float> target_data = generate_random_target(static_cast<size_t>(target_shape[1]));
        target.upload(target_data.data());
        
        // Perform training step
        model.train_step(input, target);
        
        // Record loss
        losses.push_back(model.loss());
    }
    
    // Verify that loss is generally decreasing
    bool loss_decreased = false;
    for (size_t i = 1; i < losses.size(); ++i) {
        if (losses[i] < losses[i-1]) {
            loss_decreased = true;
            break;
        }
    }
    EXPECT_TRUE(loss_decreased);
}

TEST_F(LeNet5Test, SaveLoadRoundtrip) {
    LeNet5<float> model(0.01, 0.9, true);
    std::vector<int> input_shape = {1, 1, 32, 32};
    tensor<float> input(input_shape);
    std::vector<float> input_data = generate_random_input(static_cast<size_t>(input.size()));
    input.upload(input_data.data());
    std::vector<int> target_shape = {1, 10};
    tensor<float> target(target_shape);
    std::vector<float> target_data = generate_random_target(static_cast<size_t>(target.size()));
    target.upload(target_data.data());
    // Mutate model parameters by training
    model.train_step(input, target);
    // Save to a temp file
    std::string path = "lenet5_test_model.bin";
    model.save(path);
    // Load from file
    auto loaded = LeNet5<float>::load(path, true);
    // Compare parameters
    auto params1 = model.parameters();
    auto params2 = loaded->parameters();
    ASSERT_EQ(params1.size(), params2.size());
    for (size_t i = 0; i < params1.size(); ++i) {
        // Use approx_equal for floating point tolerance
        EXPECT_TRUE(params1[i]->approx_equal(*params2[i], 1e-5f)) << "Parameter " << i << " mismatch after load/save.";
    }
    // Clean up
    std::filesystem::remove(path);
}

TEST_F(LeNet5Test, LoadCorruptFileThrows) {
    std::string bad_path = "nonexistent_file.bin";
    EXPECT_THROW({
        auto m = LeNet5<float>::load(bad_path, true);
    }, std::runtime_error);
}

} // namespace test
} // namespace dnn 
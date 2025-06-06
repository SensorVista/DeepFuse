#include <gtest/gtest.h>

#include <dnn/core/tensor.cuh>

#include <vector>
#include <random>

namespace dnn {
namespace test {

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }
};

TEST_F(TensorTest, ConstructorAndShape) {
    std::vector<int> shape = {2, 3, 4};
    tensor<float> t(shape);
    
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 24); // 2 * 3 * 4
    EXPECT_EQ(t.size(0), 2);
    EXPECT_EQ(t.size(1), 3);
    EXPECT_EQ(t.size(2), 4);
}

TEST_F(TensorTest, UploadDownload) {
    std::vector<int> shape = {2, 2};
    tensor<float> t(shape);
    
    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
    t.upload(host_data.data());
    
    std::vector<float> downloaded_data(4);
    t.download(downloaded_data.data());
    
    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_data[i], downloaded_data[i]);
    }
}

TEST_F(TensorTest, FillAndZero) {
    std::vector<int> shape = {2, 2};
    tensor<float> t(shape);
    
    t.fill(1.0f);
    std::vector<float> data(4);
    t.download(data.data());
    
    for (float val : data) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
    
    t.zero();
    t.download(data.data());
    
    for (float val : data) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }
}

TEST_F(TensorTest, Reshape) {
    std::vector<int> initial_shape = {2, 2};
    tensor<float> t(initial_shape);
      
    std::vector<int> new_shape = {4, 1};
    t.reshape(new_shape);
    
    EXPECT_EQ(t.shape(), new_shape);
    EXPECT_EQ(t.size(), 4);    
}

TEST_F(TensorTest, MoveSemantics) {
    std::vector<int> shape = {2, 2};
    tensor<float> t1(shape);
    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
    t1.upload(host_data.data());
    
    // Test move constructor
    tensor<float> t2(std::move(t1));
    EXPECT_EQ(t1.data(), nullptr); // Original tensor should be empty
    EXPECT_NE(t2.data(), nullptr); // New tensor should have the data
    
    std::vector<float> downloaded_data(4);
    t2.download(downloaded_data.data());
    
    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_data[i], downloaded_data[i]);
    }
    
    // Test move assignment
    tensor<float> t3(shape);
    t3 = std::move(t2);
    EXPECT_EQ(t2.data(), nullptr);
    EXPECT_NE(t3.data(), nullptr);
}

} // namespace test
} // namespace dnn 
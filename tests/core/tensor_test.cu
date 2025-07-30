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

TEST_F(TensorTest, Addition) {
    std::vector<int> shape = {2, 2};
    tensor<float> t1(shape);
    tensor<float> t2(shape);
    
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
    t1.upload(data1.data());
    t2.upload(data2.data());
    
    // Test operator+
    tensor<float> t3 = t1 + t2;
    std::vector<float> result(4);
    t3.download(result.data());
    
    EXPECT_FLOAT_EQ(result[0], 6.0f);  // 1.0 + 5.0
    EXPECT_FLOAT_EQ(result[1], 8.0f);  // 2.0 + 6.0
    EXPECT_FLOAT_EQ(result[2], 10.0f); // 3.0 + 7.0
    EXPECT_FLOAT_EQ(result[3], 12.0f); // 4.0 + 8.0
}

TEST_F(TensorTest, InPlaceAddition) {
    std::vector<int> shape = {2, 2};
    tensor<float> t1(shape);
    tensor<float> t2(shape);
    
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
    t1.upload(data1.data());
    t2.upload(data2.data());
    
    // Test operator+=
    t1 += t2;
    std::vector<float> result(4);
    t1.download(result.data());
    
    EXPECT_FLOAT_EQ(result[0], 6.0f);  // 1.0 + 5.0
    EXPECT_FLOAT_EQ(result[1], 8.0f);  // 2.0 + 6.0
    EXPECT_FLOAT_EQ(result[2], 10.0f); // 3.0 + 7.0
    EXPECT_FLOAT_EQ(result[3], 12.0f); // 4.0 + 8.0
}

TEST_F(TensorTest, Transpose) {
    std::vector<int> shape = {2, 3};
    tensor<float> t(shape);
    
    // Initialize with a 2x3 matrix:
    // [1 2 3]
    // [4 5 6]
    std::vector<float> data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    t.upload(data.data());

    // Transpose it: should become 3x2
    // [1 4]
    // [2 5]
    // [3 6]
    t.transpose();

    // Check new shape
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 2);

    // Check row-major flattened result: [1, 4, 2, 5, 3, 6]
    std::vector<float> result(6);
    t.download(result.data());

    EXPECT_FLOAT_EQ(result[0], 1.0f);  // (0,0)
    EXPECT_FLOAT_EQ(result[1], 4.0f);  // (0,1)
    EXPECT_FLOAT_EQ(result[2], 2.0f);  // (1,0)
    EXPECT_FLOAT_EQ(result[3], 5.0f);  // (1,1)
    EXPECT_FLOAT_EQ(result[4], 3.0f);  // (2,0)
    EXPECT_FLOAT_EQ(result[5], 6.0f);  // (2,1)
}

TEST_F(TensorTest, ShapeMismatchAddition) {
    std::vector<int> shape1 = {2, 2};
    std::vector<int> shape2 = {2, 3};
    tensor<float> t1(shape1);
    tensor<float> t2(shape2);
    
    // Test operator+
    EXPECT_THROW(t1 + t2, std::runtime_error);
    
    // Test operator+=
    EXPECT_THROW(t1 += t2, std::runtime_error);
}

TEST_F(TensorTest, InvalidTranspose) {
    std::vector<int> shape = {2, 3, 4}; // 3D tensor
    tensor<float> t(shape);
    
    // Transpose should not throw for non-2D tensors as it handles higher dimensions
    // Test the behavior for 3D tensor
    std::vector<int> shape3D = {2, 3, 4};
    tensor<float> t3D(shape3D);
    t3D.transpose();
    std::vector<int> expectedShape = {2, 4, 3};
    EXPECT_EQ(t3D.shape(), expectedShape);
}

TEST_F(TensorTest, Clone) {
    std::vector<int> shape = {2, 2};
    tensor<float> t1(shape);
    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
    t1.upload(host_data.data());
    
    tensor<float> t2 = t1.clone();
    EXPECT_EQ(t2.shape(), shape);
    EXPECT_NE(t2.data(), t1.data()); // Ensure deep copy
    
    std::vector<float> downloaded_data(4);
    t2.download(downloaded_data.data());
    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_data[i], downloaded_data[i]);
    }
}

TEST_F(TensorTest, Equals) {
    std::vector<int> shape = {2, 2};
    tensor<float> t1(shape);
    tensor<float> t2(shape);
    tensor<float> t3(shape);
    
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {1.0f, 2.0f, 3.0f, 5.0f};
    t1.upload(data1.data());
    t2.upload(data1.data());
    t3.upload(data2.data());
    
    EXPECT_TRUE(t1.equals(t2));
    EXPECT_FALSE(t1.equals(t3));
}

TEST_F(TensorTest, ApproxEqual) {
    std::vector<int> shape = {2, 2};
    tensor<float> t1(shape);
    tensor<float> t2(shape);
    
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {1.001f, 2.001f, 3.001f, 4.001f};
    t1.upload(data1.data());
    t2.upload(data2.data());
    
    EXPECT_TRUE(t1.approx_equal(t2, 0.01f));
    EXPECT_FALSE(t1.approx_equal(t2, 0.0001f));
}

TEST_F(TensorTest, NarrowAndSlice) {
    std::vector<int> shape = {3, 3};
    tensor<float> t(shape);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    t.upload(data.data());
    
    // Narrow along dimension 0
    t.narrow(0, 1, 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    
    std::vector<float> narrowed_data(6);
    t.download(narrowed_data.data());
    EXPECT_FLOAT_EQ(narrowed_data[0], 4.0f);
    EXPECT_FLOAT_EQ(narrowed_data[1], 5.0f);
    EXPECT_FLOAT_EQ(narrowed_data[2], 6.0f);
    EXPECT_FLOAT_EQ(narrowed_data[3], 7.0f);
    EXPECT_FLOAT_EQ(narrowed_data[4], 8.0f);
    EXPECT_FLOAT_EQ(narrowed_data[5], 9.0f);
    
    // Reset tensor for slice test
    tensor<float> t2(shape);
    t2.upload(data.data());
    t2.slice(1, 1, 3);
    EXPECT_EQ(t2.shape()[0], 3);
    EXPECT_EQ(t2.shape()[1], 2);
    
    std::vector<float> sliced_data(6);
    t2.download(sliced_data.data());
    EXPECT_FLOAT_EQ(sliced_data[0], 2.0f);
    EXPECT_FLOAT_EQ(sliced_data[1], 3.0f);
    EXPECT_FLOAT_EQ(sliced_data[2], 5.0f);
    EXPECT_FLOAT_EQ(sliced_data[3], 6.0f);
    EXPECT_FLOAT_EQ(sliced_data[4], 8.0f);
    EXPECT_FLOAT_EQ(sliced_data[5], 9.0f);
}

TEST_F(TensorTest, Permute) {
    std::vector<int> shape = { 2, 3, 4 };
    tensor<float> t(shape);
    std::vector<float> data(24);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i);
    t.upload(data.data());

    std::vector<int> perm = { 1, 0, 2 };
    t.permute(perm);
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 2);
    EXPECT_EQ(t.shape()[2], 4);

    std::vector<float> result(24);
    t.download(result.data());

    // (0,0,0) -> (0,0,0)   value = 0
    // (1,0,0) -> (0,1,0)   value = 12
    // (0,1,0) -> (1,0,0)   value = 4
    EXPECT_FLOAT_EQ(result[0], 0.0f);   // (0,0,0)
    EXPECT_FLOAT_EQ(result[4], 12.0f);  // (0,1,0)
    EXPECT_FLOAT_EQ(result[8], 4.0f);   // (1,0,0)
}

TEST_F(TensorTest, Transpose2D) {
    std::vector<int> shape = {2, 3};
    tensor<float> t(shape);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    t.upload(data.data());
    t.transpose();
    std::vector<int> expectedShape = {3, 2};
    EXPECT_EQ(t.shape(), expectedShape);
    std::vector<float> result(6);
    t.download(result.data());
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 4.0f);
    EXPECT_FLOAT_EQ(result[2], 2.0f);
    EXPECT_FLOAT_EQ(result[3], 5.0f);
    EXPECT_FLOAT_EQ(result[4], 3.0f);
    EXPECT_FLOAT_EQ(result[5], 6.0f);
}

TEST_F(TensorTest, Transpose3D_DHW_DWH) {
    std::vector<int> shape = { 2, 3, 4 };
    tensor<float> t(shape);
    std::vector<float> data(24);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i);
    t.upload(data.data());

    std::vector<int> perm = { 0, 2, 1 };
    t.permute(perm);

    std::vector<int> expectedShape = { 2, 4, 3 };
    EXPECT_EQ(t.shape(), expectedShape);

    std::vector<float> result(24);
    t.download(result.data());

    EXPECT_FLOAT_EQ(result[0], 0.0f);   // (0,0,0)
    EXPECT_FLOAT_EQ(result[1], 4.0f);   // (0,0,1)
    EXPECT_FLOAT_EQ(result[12], 12.0f); // (1,0,0)
}

TEST_F(TensorTest, Transpose3D_HWD_WDH) {
    std::vector<int> shape = {3, 4, 2};  // [H, W, D]
    tensor<float> t(shape);
    std::vector<float> data(24);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i);
    t.upload(data.data());

    std::vector<int> perm = {2, 1, 0};  // HWD -> WHD
    t.permute(perm);

    std::vector<int> expectedShape = {2, 4, 3};  // [W, D, H]
    EXPECT_EQ(t.shape(), expectedShape);

    std::vector<float> result(24);
    t.download(result.data());

    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 8.0f);
    EXPECT_FLOAT_EQ(result[2], 16.0f);
}


TEST_F(TensorTest, Transpose4D_NHWC_NCHW) {
    std::vector<int> shape = {2, 3, 4, 5};
    tensor<float> t(shape);
    std::vector<float> data(120);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i);
    t.upload(data.data());
    std::vector<int> perm = {0, 3, 1, 2};
    t.permute(perm);
    std::vector<int> expectedShape = {2, 5, 3, 4};
    EXPECT_EQ(t.shape(), expectedShape);
    std::vector<float> result(120);
    t.download(result.data());
    // Check specific values: original (0,0,0,0)=0 -> new (0,0,0,0)=0
    EXPECT_FLOAT_EQ(result[0], 0.0f);
    // original (0,0,0,1)=1 -> new (0,1,0,0)=1
    EXPECT_FLOAT_EQ(result[12], 1.0f);
    // original (1,0,0,0)=60 -> new (1,0,0,0)=60
    EXPECT_FLOAT_EQ(result[60], 60.0f);
}

TEST_F(TensorTest, Transpose4D_NCHW_NHWC) {
    std::vector<int> shape = {2, 5, 3, 4};
    tensor<float> t(shape);
    std::vector<float> data(120);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i);
    t.upload(data.data());
    std::vector<int> perm = {0, 2, 3, 1};
    t.permute(perm);
    std::vector<int> expectedShape = {2, 3, 4, 5};
    EXPECT_EQ(t.shape(), expectedShape);
    std::vector<float> result(120);
    t.download(result.data());
    // Check specific values: original (0,0,0,0)=0 -> new (0,0,0,0)=0
    EXPECT_FLOAT_EQ(result[0], 0.0f);
    // original (0,1,0,0)=12 -> new (0,0,0,1)=12
    EXPECT_FLOAT_EQ(result[1], 12.0f);
    // original (1,0,0,0)=60 -> new (1,0,0,0)=60
    EXPECT_FLOAT_EQ(result[60], 60.0f);
}

} // namespace test
} // namespace dnn 
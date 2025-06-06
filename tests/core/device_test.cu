#include <gtest/gtest.h>
#include "dnn/core/device.cuh"
#include <sstream>

namespace dnn {
namespace test {

class DeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test resources
    }

    void TearDown() override {
        // Clean up any common test resources
    }
};

TEST_F(DeviceTest, Constructor) {
    // Test constructor with valid device ID
    EXPECT_NO_THROW({
        Device device(0);
        EXPECT_EQ(device.id(), 0);
    });

    // Test constructor with invalid device ID
    int device_count;
    cudaGetDeviceCount(&device_count);
    EXPECT_THROW(Device device(device_count), std::runtime_error);
}

TEST_F(DeviceTest, DeviceProperties) {
    Device device(0);
    
    // Test device name
    EXPECT_FALSE(device.name().empty());
    
    // Test compute capability
    EXPECT_GE(device.compute_capability_major(), 3);
    EXPECT_GE(device.compute_capability_minor(), 0);
    
    // Test architecture name
    EXPECT_FALSE(device.architecture_name().empty());
    
    // Test memory properties
    EXPECT_GT(device.total_memory_bytes(), 0);
    EXPECT_GT(device.available_memory_bytes(), 0);
    EXPECT_LE(device.available_memory_bytes(), device.total_memory_bytes());
    
    // Test SM properties
    EXPECT_GT(device.num_sm(), 0);
    EXPECT_GT(device.cuda_cores_per_sm(), 0);
    EXPECT_GT(device.total_cuda_cores(), 0);
    
    // Test tensor cores
    EXPECT_GE(device.tensor_cores_per_sm(), 0);
    EXPECT_GE(device.total_tensor_cores(), 0);
    
    // Test clock speed
    EXPECT_GT(device.clock_ghz(), 0.0f);
}

TEST_F(DeviceTest, NVLinkSupport) {
    Device device(0);
    // NVLink support is hardware dependent, so we just test the function doesn't crash
    device.nvlink_supported();
}

TEST_F(DeviceTest, DumpInfo) {
    Device device(0);
    std::stringstream ss;
    device.dump_info(ss);
    std::string info = ss.str();
    
    // Verify that the output contains expected information
    EXPECT_TRUE(info.find("CUDA Device") != std::string::npos);
    EXPECT_TRUE(info.find("Architecture") != std::string::npos);
    EXPECT_TRUE(info.find("VRAM") != std::string::npos);
    EXPECT_TRUE(info.find("SMs") != std::string::npos);
    EXPECT_TRUE(info.find("CUDA Cores") != std::string::npos);
    EXPECT_TRUE(info.find("Clock") != std::string::npos);
}

TEST_F(DeviceTest, MultiDeviceSupport) {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count > 1) {
        // Test multiple devices
        Device device0(0);
        Device device1(1);
        
        EXPECT_NE(device0.id(), device1.id());
        // Devices should have different properties
        EXPECT_NE(device0.total_memory_bytes(), device1.total_memory_bytes());
    }
}

} // namespace test
} // namespace dnn 
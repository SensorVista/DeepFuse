#include <gtest/gtest.h>

#include <dnn/core/cuda.cuh>

#include <sstream>
#include <thread>

namespace dnn {
namespace test {

class CudaTest : public ::testing::Test {
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
};

TEST_F(CudaTest, Constructor) {
    // Test default constructor
    EXPECT_NO_THROW({
        Cuda cuda;
        EXPECT_EQ(cuda.id(), 0);
    });

    // Test constructor with specific device ID
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count > 1) {
        EXPECT_NO_THROW({
            Cuda cuda(1);
            EXPECT_EQ(cuda.id(), 1);
        });
    }

    // Test constructor with invalid device ID
    EXPECT_THROW(Cuda cuda(device_count), std::runtime_error);

    // Reset device state for rest of test suite
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    Cuda::set_stream(nullptr);    
}

TEST_F(CudaTest, CurrentInstance) {
    // Test current instance management
    {
        Cuda cuda1(0);
        EXPECT_EQ(&Cuda::current(), &cuda1);

        {
            Cuda cuda2(0);
            EXPECT_EQ(&Cuda::current(), &cuda2);  // Now inside cuda2 scope
        }
    }

    // Test that current() throws when no instance exists
    EXPECT_THROW(Cuda::current(), std::runtime_error);
}

TEST_F(CudaTest, GetDevices) {
    const auto& devices = Cuda::get_devices();
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    EXPECT_EQ(devices.size(), device_count);
    
    // Verify each device
    for (size_t i = 0; i < devices.size(); ++i) {
        EXPECT_EQ(devices[i].id(), i);
        EXPECT_FALSE(devices[i].name().empty());
    }
}

TEST_F(CudaTest, DumpInfo) {
    Cuda cuda;
    std::stringstream ss;
    cuda.dump_info(ss);
    std::string info = ss.str();
    
    // Verify that the output contains expected information
    EXPECT_TRUE(info.find("[CUDA System Summary]") != std::string::npos);
    EXPECT_TRUE(info.find("CUDA Device") != std::string::npos);
}

#ifdef ENABLE_CUDNN
TEST_F(CudaTest, CudnnHandle) {
    Cuda cuda;
    EXPECT_NE(cuda.cudnn(), nullptr);
}

TEST_F(CudaTest, CublasHandle) {
    Cuda cuda;
    EXPECT_NE(cuda.cublas(), nullptr);
}
#endif

TEST_F(CudaTest, ThreadSafety) {
    // Test that each thread gets its own Cuda instance
    std::thread t1([]() {
        Cuda cuda1(0);
        EXPECT_EQ(cuda1.id(), 0);
    });
    
    std::thread t2([]() {
        Cuda cuda2(0);
        EXPECT_EQ(cuda2.id(), 0);
    });
    
    t1.join();
    t2.join();
}

TEST_F(CudaTest, DeviceSwitching) {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count > 1) {
        // Test switching between devices
        Cuda cuda1(0);
        EXPECT_EQ(cuda1.id(), 0);
        
        Cuda cuda2(1);
        EXPECT_EQ(cuda2.id(), 1);
        
        // Verify that current() returns the most recently created instance
        EXPECT_EQ(&Cuda::current(), &cuda2);
    }
}

TEST_F(CudaTest, DefaultStreamIsLegacy) {
    EXPECT_EQ(Cuda::stream(), cudaStreamLegacy);
}

TEST_F(CudaTest, SetAndGetStream) {
    cudaStream_t s;
    cudaStreamCreate(&s);

    Cuda::set_stream(s);
    EXPECT_EQ(Cuda::stream(), s);

    Cuda::set_stream(nullptr);  // Reset
    EXPECT_EQ(Cuda::stream(), cudaStreamLegacy);

    cudaStreamDestroy(s);
}

TEST_F(CudaTest, StreamGuardRestoresPreviousStream) {
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    Cuda::set_stream(s1);
    EXPECT_EQ(Cuda::stream(), s1);

    {
        StreamGuard guard(s2);
        EXPECT_EQ(Cuda::stream(), s2);
    }

    EXPECT_EQ(Cuda::stream(), s1);  // Restored
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
}

} // namespace test
} // namespace dnn 
#pragma once

#include "device.cuh"
#include "../utils/common.cuh"

#include <cuda_runtime.h>

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#include <cublas_v2.h>
#endif

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <iomanip>
#include <thread>

namespace dnn {

// Cuda: Sets context and manages cudnnHandle_t
class Cuda {
private:
    int id_;

#ifdef ENABLE_CUDNN
    cudnnHandle_t handle_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
#endif

    // TLS for current Cuda instance
    static Cuda*& current_instance_ref();

    // TLS for current CUDA stream (multiplexing)
    static thread_local cudaStream_t current_stream_;

public:
    Cuda(int device_id = 0);
    ~Cuda();

#ifdef ENABLE_CUDNN
    cudnnHandle_t cudnn() const;
    cublasHandle_t cublas() const;
#endif

    int id() const;

    static Cuda& current();

    static const std::vector<Device>& get_devices();

    void dump_info(std::ostream& os = std::cout) const;


    // Stream multiplexing API
    static void set_stream(cudaStream_t stream);

    static cudaStream_t stream();

};

// StreamGuard: Scoped stream RAII helper
class StreamGuard {
private:
    cudaStream_t previous_;

public:
    StreamGuard(cudaStream_t new_stream);
    ~StreamGuard();
};

} // namespace dnn

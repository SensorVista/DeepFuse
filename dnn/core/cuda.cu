#pragma once

#include "cuda.cuh"
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

// TLS for current Cuda instance
/* static */ Cuda*& Cuda::current_instance_ref() {
    static thread_local Cuda* instance = nullptr;
    return instance;
}

Cuda::Cuda(int device_id /* = 0 */) : id_(device_id) {
    utils::CHECK_CUDA_EX(cudaSetDevice(id_));

#ifdef ENABLE_CUDNN
    utils::CHECK_CUDNN_EX(cudnnCreate(&handle_));
    utils::CHECK_CUBLAS_EX(cublasCreate(&cublas_handle_));
#endif

    Cuda::current_instance_ref() = this;
}

Cuda::~Cuda() {
#ifdef ENABLE_CUDNN
    if (handle_) cudnnDestroy(handle_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);
#endif
    if (current_instance_ref() == this)
        current_instance_ref() = nullptr;
}

#ifdef ENABLE_CUDNN
cudnnHandle_t Cuda::cudnn() const { return handle_; }
cublasHandle_t Cuda::cublas() const { return cublas_handle_; }
#endif

int Cuda::id() const { return id_; }

Cuda& Cuda::current() {
    Cuda* inst = current_instance_ref();
    if (!inst) throw std::runtime_error("No current Cuda instance set.");
    return *inst;
}

const std::vector<Device>& Cuda::get_devices() {
    static std::vector<Device> cache = [] {
        int count = 0;
        utils::CHECK_CUDA_EX(cudaGetDeviceCount(&count));
        std::vector<Device> result;
        for (int i = 0; i < count; ++i)
            result.emplace_back(i);
        return result;
    }();
    return cache;
}

void Cuda::dump_info(std::ostream& os /* = std::cout */) const {
    os << "[CUDA System Summary]\n";
    for (const auto& device : get_devices()) {
        device.dump_info(os);
        os << std::endl;
    }
}

// Stream multiplexing API
/* static */ void Cuda::set_stream(cudaStream_t stream) {
    current_stream_ = stream;
}

/* static */ cudaStream_t Cuda::stream() {
    return current_stream_ ? current_stream_ : cudaStreamLegacy;
}

thread_local cudaStream_t Cuda::current_stream_ = nullptr;

// StreamGuard: Scoped stream RAII helper
StreamGuard::StreamGuard(cudaStream_t new_stream) {
    previous_ = Cuda::stream();
    Cuda::set_stream(new_stream);
}

StreamGuard::~StreamGuard() {
    Cuda::set_stream(previous_);
}

} // namespace dnn

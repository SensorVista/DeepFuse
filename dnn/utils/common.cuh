#pragma once

#include "../core/tensor.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace dnn::utils {

inline void check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            ": " + cudaGetErrorString(error)
        );
    }
}

inline void check_cuda_error(const char* file, int line) {
    cudaError_t error = cudaGetLastError();
    check_cuda_error(error, file, line);
}

#define CHECK_CUDA_EX(call) check_cuda_error(call, __FILE__, __LINE__)
#define THROW_CUDA_EX() check_cuda_error(__FILE__, __LINE__)

#define CHECK_CUDNN_EX(call) do { \
    cudnnStatus_t err = call; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuDNN error"); \
    } \
} while (0)

inline void pad2d(
    const float* input_dev,  // [N, C, H_in, W_in]
    float* output_dev,       // [N, C, H_out, W_out]
    int N, int C,
    int H_in, int W_in,
    int H_out, int W_out,
    int pad_y, int pad_x);

void clip_grad_norm(
    std::vector<tensor<float>*>& gradients, 
    float max_norm);

} // namespace dnn::utils


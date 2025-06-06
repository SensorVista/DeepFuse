#pragma once

#include "../core/tensor.cuh"

#include <cuda_runtime.h>

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#include <cublas_v2.h>
#endif

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

#ifdef ENABLE_CUDNN
[[nodiscard]] inline void check_cudnn_error(cudnnStatus_t err, const char* file, int line) noexcept(false) {
    if (err != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuDNN error at ") + file + ":" + std::to_string(line) +
            " -> " + cudnnGetErrorString(err)
        );
    }
}

#define CHECK_CUDNN_EX(call) check_cudnn_error((call), __FILE__, __LINE__)

[[nodiscard]] inline void check_cublas_error(cublasStatus_t err, const char* file, int line) noexcept(false) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::string msg;
        switch (err) {
        case CUBLAS_STATUS_NOT_INITIALIZED:  msg = "cuBLAS not initialized"; break;
        case CUBLAS_STATUS_ALLOC_FAILED:     msg = "cuBLAS memory allocation failed"; break;
        case CUBLAS_STATUS_INVALID_VALUE:    msg = "cuBLAS received an invalid value"; break;
        case CUBLAS_STATUS_ARCH_MISMATCH:    msg = "cuBLAS architecture mismatch"; break;
        case CUBLAS_STATUS_MAPPING_ERROR:    msg = "cuBLAS memory mapping error"; break;
        case CUBLAS_STATUS_EXECUTION_FAILED: msg = "cuBLAS execution failed"; break;
        case CUBLAS_STATUS_INTERNAL_ERROR:   msg = "cuBLAS internal error"; break;
        default:                              msg = "Unknown cuBLAS error"; break;
        }
        throw std::runtime_error(
            std::string("cuBLAS error at ") + file + ":" + std::to_string(line) + " -> " + msg
        );
    }
}

#define CHECK_CUBLAS_EX(call) check_cublas_error((call), __FILE__, __LINE__)

#endif

// Return pointer to scalar 1 of type T (host-side constant)
template<typename T>
const void* one();

// Return pointer to scalar 0 of type T (host-side constant)
template<typename T>
const void* zero();

#ifdef ENABLE_CUDNN
// cuDNN data type
template<typename T>
inline const cudnnDataType_t dnn_type() {
    if constexpr (std::is_same_v<T, float>)
        return CUDNN_DATA_FLOAT;
    else if constexpr (std::is_same_v<T, __half>)
        return CUDNN_DATA_HALF;
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return CUDNN_DATA_BFLOAT16;
#ifdef __CUDA_FP8_TYPES_EXIST__
    else if constexpr (std::is_same_v<T, __nv_fp8x4_e5m2>)
        return CUDNN_DATA_FP8_E5M2;
    else if constexpr (std::is_same_v<T, __nv_fp8x4_e4m3>)
        return CUDNN_DATA_FP8_E4M3;
#endif
    else if constexpr (std::is_same_v<T, int8_t>)
        return CUDNN_DATA_INT8;
    else if constexpr (std::is_same_v<T, uint8_t>)
        return CUDNN_DATA_UINT8;
    else
        throw std::runtime_error("Unsupported data type for cuDNN");
}

// cuBLAS data type
template<typename T>
const cudaDataType_t blas_type() {
    if constexpr (std::is_same_v<T, float>)
        return CUDA_R_32F;
    else if constexpr (std::is_same_v<T, __half>)
        return CUDA_R_16F;
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return CUDA_R_16BF;
#ifdef __CUDA_FP8_TYPES_EXIST__
    else if constexpr (std::is_same_v<T, __nv_fp8x4_e5m2>)
        return CUDA_R_8F_E5M2;
    else if constexpr (std::is_same_v<T, __nv_fp8x4_e4m3>)
        return CUDA_R_8F_E4M3;
#endif
    else if constexpr (std::is_same_v<T, int8_t>)
        return CUDA_R_8I;
    else if constexpr (std::is_same_v<T, uint8_t>)
        return CUDA_R_8U;
    else
        throw std::runtime_error("Unsupported data type for cuBLAS");
}

// Compute type for cuBLAS (e.g., CUBLAS_COMPUTE_32F_FAST_TF32, etc.)
template<typename T>
const cublasComputeType_t compute_type();
#endif

// Infinity support
template<typename T>
__device__ __host__ inline T neg_infinity();
template<typename T>
__device__ __host__ inline T infinity();

// Pad 2D input tensor to output tensor
inline void pad2d(
    const float* input_dev,  // [N, C, H_in, W_in]
    float* output_dev,       // [N, C, H_out, W_out]
    int N, int C,
    int H_in, int W_in,
    int H_out, int W_out,
    int pad_y, int pad_x);

// Clip gradients by norm
void clip_grad_norm(
    std::vector<tensor<float>*>& gradients, 
    float max_norm);

} // namespace dnn::utils


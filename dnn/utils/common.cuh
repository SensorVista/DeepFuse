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
#include <iomanip>

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
inline const cudnnDataType_t dnn_type() ;

// cuBLAS data type
template<typename T>
const cudaDataType_t blas_type();

// Compute type for cuBLAS (e.g., CUBLAS_COMPUTE_32F_FAST_TF32, etc.)
template<typename T>
const cublasComputeType_t compute_type();
#endif

// Infinity support
template<typename T>
__device__ __host__ inline T neg_infinity();
template<typename T>
__device__ __host__ inline T infinity();

template<typename T>
__device__ __host__ inline float to_float(T);

template<typename T>
__device__ __host__ inline __half to_half(T);

// Pad 2D input tensor to output tensor
inline void pad2d(
    const float* input_dev,  // [N, C, H_in, W_in]
    float* output_dev,       // [N, C, H_out, W_out]
    int N, int C,
    int H_in, int W_in,
    int H_out, int W_out,
    int pad_y, int pad_x);

// Clip gradients by norm
template<typename T>
void clip_grad_norm(
    std::vector<tensor<T>*>& gradients, 
    T max_norm);

// Utility: Convert tensor<int> to one-hot tensor<T>
template<typename T>
tensor<T> to_one_hot(const tensor<int>& indices, int num_classes) {
    std::vector<int> shape = indices.shape();
    shape.push_back(num_classes); // e.g., [B, T, C]
    tensor<T> one_hot(shape);
    one_hot.zero();
    std::vector<int> idx_data(indices.size());
    indices.download(idx_data.data());
    std::vector<T> one_hot_data(one_hot.size(), static_cast<T>(0));
    for (int i = 0; i < idx_data.size(); ++i) {
        int class_idx = idx_data[i];
        if (class_idx >= 0 && class_idx < num_classes) {
            one_hot_data[i * num_classes + class_idx] = static_cast<T>(1);
        }
    }
    one_hot.upload(one_hot_data.data());
    return one_hot;
}

} // namespace dnn::utils

#pragma once

#include "dnn/core/tensor.cuh"

#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace dnn::utils;

template<typename T>
__global__ void fill_kernel(T* data, T value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride)
        data[i] = value;
}

namespace dnn {

template<typename T>
tensor<T>::tensor(const std::vector<int>& shape)
: shape_(shape)
, data_(nullptr)
#ifdef ENABLE_CUDNN
, desc_(nullptr) 
#endif
{
#ifdef ENABLE_CUDNN
    if (shape.size() < 1 || shape.size() > CUDNN_DIM_MAX) {
        throw std::invalid_argument("cuDNN only supports 1 to 8 dimensions");
    }
#endif

    total_size_ = 1;
    for (int d : shape) total_size_ *= d;
    cudaMalloc(&data_, total_size_ * sizeof(T));

#ifdef ENABLE_CUDNN
    cudnnCreateTensorDescriptor(&desc_);

    // Use Nd descriptor for general case
    std::vector<int> dims = shape_;
    std::vector<int> strides(dims.size(), 1);
    for (int i = dims.size() - 2; i >= 0; --i)
        strides[i] = strides[i+1] * dims[i+1];

    // Type-to-cuDNN mapping (inline)
    cudnnDataType_t dtype;
    if (std::is_same<T, float>::value)        dtype = CUDNN_DATA_FLOAT;
    else if (std::is_same<T, __half>::value)       dtype = CUDNN_DATA_HALF;
    else if (std::is_same<T, __nv_bfloat16>::value)dtype = CUDNN_DATA_BFLOAT16;
    else if (std::is_same<T, int8_t>::value)       dtype = CUDNN_DATA_INT8;
    else if (std::is_same<T, uint8_t>::value)      dtype = CUDNN_DATA_UINT8;
    else
        throw std::runtime_error("Unsupported data type for cuDNN");

    // Set the tensor descriptor
    cudnnSetTensorNdDescriptor(
        desc_,
        dtype,
        static_cast<int>(dims.size()),
        dims.data(),
        strides.data()
    );
#endif
}

#ifdef ENABLE_CUDNN
template<typename T>
cudnnTensorDescriptor_t tensor<T>::desc() const { 
    return desc_; 
}

template<typename T>
cudnnDataType_t tensor<T>::dnn_type() const { 
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

template<typename T>
cudaDataType_t tensor<T>::blas_type() const {
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
#endif

template<typename T>
tensor<T>::~tensor() {
    if (data_) {
        CHECK_CUDA_EX(cudaFree(data_));
    }
#ifdef ENABLE_CUDNN
    if (desc_) {
        CHECK_CUDNN_EX(cudnnDestroyTensorDescriptor(desc_));
        desc_ = nullptr;
    }
#endif
}

template<typename T>
void tensor<T>::upload(const T* host_data) {
    CHECK_CUDA_EX(cudaMemcpy(data_, host_data, total_size_ * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void tensor<T>::download(T* host_data) const {
    CHECK_CUDA_EX(cudaMemcpy(host_data, data_, total_size_ * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
T* tensor<T>::data() { return data_; }

template<typename T>
const T* tensor<T>::data() const { return data_; }

template<typename T>
const std::vector<int>& tensor<T>::shape() const { return shape_; }

template<typename T>
int tensor<T>::size() const { return total_size_; }

template<typename T>
int tensor<T>::size(int dim) const {
    if (dim >= size()) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape_[dim];
}

template<typename T>
void tensor<T>::reshape(const std::vector<int>& new_shape) {
    if (data_) cudaFree(data_);
    shape_ = new_shape;
    total_size_ = 1;
    for (int d : shape_) total_size_ *= d;
    CHECK_CUDA_EX(cudaMalloc(&data_, total_size_ * sizeof(T)));
}

template<typename T>
void tensor<T>::fill(T value) {
    int threads = 256;
    int blocks = (total_size_ + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(data_, value, total_size_);
    CHECK_CUDA_EX(cudaDeviceSynchronize());
}

template<typename T>
void tensor<T>::zero() {
    fill(static_cast<T>(0));
}

template<typename T>
tensor<T>::tensor(tensor&& other) noexcept 
    : shape_(std::move(other.shape_))
    , data_(other.data_)
    , total_size_(other.total_size_)
#ifdef ENABLE_CUDNN
    , desc_(other.desc_) 
#endif
{
    other.data_ = nullptr; // Nullify the source pointer
#ifdef ENABLE_CUDNN
    other.desc_ = nullptr; // Nullify the source descriptor
#endif
}

template<typename T>
tensor<T>& tensor<T>::operator=(tensor&& other) noexcept {
    if (this != &other) {
        // Free existing resources
        if (data_) {
            CHECK_CUDA_EX(cudaFree(data_));
        }
#ifdef ENABLE_CUDNN
        if (desc_) { // Destroy existing descriptor
            CHECK_CUDNN_EX(cudnnDestroyTensorDescriptor(desc_));
            desc_ = nullptr;
        }
#endif
        // Transfer ownership
        shape_ = std::move(other.shape_);
        data_ = other.data_;
        total_size_ = other.total_size_;
        other.data_ = nullptr; // Nullify the source pointer

#ifdef ENABLE_CUDNN
        desc_ = other.desc_; // Transfer descriptor
        other.desc_ = nullptr; // Nullify the source descriptor
#endif
    }
    return *this;
}

template<typename T>
void tensor<T>::copy_from(const tensor<T>& src) {
    if (total_size_ != src.total_size_)
        throw std::invalid_argument("Mismatched sizes in copy_from");
    CHECK_CUDA_EX(cudaMemcpy(data_, src.data_, total_size_ * sizeof(T), cudaMemcpyDeviceToDevice));
}

template<typename T>
std::string tensor<T>::to_string() const {
    std::ostringstream oss;
    oss << "tensor: [";
    for (int i = 0; i < shape_.size(); ++i)
        oss << shape_[i] << (i < shape_.size() - 1 ? ", " : "]");
    return oss.str();
}

// Forward declarations for supported template types
template class tensor<float>;  // FP32
template class tensor<__half>; // FP16
template class tensor<__nv_bfloat16>; // BF16

#ifdef __CUDA_FP8_TYPES_EXIST__
// Use FP8 Tensor Core Hopper+ (SM 9.0+)
template class tensor<__nv_fp8x4_e5m2>;
template class tensor<__nv_fp8x4_e4m3>;
#endif

// general purpose (or int8 emulation if needed)
template class tensor<int8_t>;
template class tensor<uint8_t>;

} // namespace dnn

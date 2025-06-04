#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef __CUDA_FP8_TYPES_EXIST__
#include <cuda_fp8.h>
#endif

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

#include <vector>
#include <string>

namespace dnn {

// Define the tensor struct
template<typename T>
struct tensor {
protected:
    std::vector<int> shape_;
    T* data_;
    int total_size_;
#ifdef ENABLE_CUDNN
    cudnnTensorDescriptor_t desc_;
#endif

public:
    tensor(const std::vector<int>& shape);
    virtual ~tensor();

    // Upload data from host to device
    virtual void upload(const T* host_data);

    // Download data from device to host
    virtual void download(T* host_data) const;

    // Get device pointer
    T* data();
    const T* data() const;

    // Get shape
    const std::vector<int>& shape() const;

    // Get total size
    int size() const;

    // Get size of a specific dimension
    int size(int dim) const;

    // Resize and allocate new memory
    void reshape(const std::vector<int>& new_shape);

    // Fill tensor with a value
    void fill(T value);

    // Fill tensor with zero
    void zero();

    // Delete copy constructor and copy assignment operator
    tensor(const tensor&) = delete;
    tensor& operator=(const tensor&) = delete;

    // Implement move constructor
    tensor(tensor&& other) noexcept;

    // Implement move assignment operator
    tensor& operator=(tensor&& other) noexcept;

    // Copy tensor data from another tensor
    void copy_from(const tensor<T>& src);

    // Output tensor to string
    std::string to_string() const;

#ifdef ENABLE_CUDNN
    // cuDNN descriptor
    cudnnTensorDescriptor_t desc() const;

    // cuDNN data type
    cudnnDataType_t dnn_type() const;

    // cuBLAS data type
    cudaDataType_t blas_type() const;
#endif
};

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

// Concrete aliases for commonly used tensor types
using tensorf   = tensor<float>;
using tensorh   = tensor<__half>;
using tensorbf  = tensor<__nv_bfloat16>;
using tensori8  = tensor<int8_t>;
using tensoru8  = tensor<uint8_t>;

} // namespace dnn 
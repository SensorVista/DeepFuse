#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <vector>
#include <string>

namespace lenet5 {

// Define the tensor struct
template<typename T>
struct tensor {
protected:
    std::vector<size_t> shape_;
    T* device_ptr_;
    size_t total_size_;

public:
    tensor(const std::vector<size_t>& shape);
    virtual ~tensor();

    // Upload data from host to device
    virtual void upload(const T* host_data);

    // Download data from device to host
    virtual void download(T* host_data) const;

    // Get device pointer
    T* data();
    const T* data() const;

    // Get shape
    const std::vector<size_t>& shape() const;

    // Get total size
    size_t size() const;

    // Get size of a specific dimension
    size_t size(size_t dim) const;

    // Resize and allocate new memory
    void reshape(const std::vector<size_t>& new_shape);

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

    // Copy tensor data
    void copy_from(const tensor<T>& src);

    // Output tensor to string
    std::string to_string() const;
};

// Forward declarations for supported template types
template class tensor<float>;  // FP32
template class tensor<__half>; // FP16
template class tensor<__nv_bfloat16>; // BF16
template class tensor<int8_t>;
template class tensor<unsigned char>; // Explicitly instantiate for unsigned char
template class tensor<bool>;

// Concrete aliases for commonly used tensor types
using tensorf   = tensor<float>;    // 32-bit float tensor
using tensorh   = tensor<__half>;   // 16-bit half tensor
using tensorbf  = tensor<__nv_bfloat16>;   // 16-bit half tensor
using tensori8  = tensor<int8_t>;   // 8-bit integer tensor (for quantized models)

} // namespace lenet5 
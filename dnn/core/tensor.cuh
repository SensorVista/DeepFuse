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
#include <iostream>
#include <cstdint>

namespace dnn {

// Define the tensor struct
template<typename TT>
struct tensor {
protected:
    std::vector<int> shape_;
    TT* data_;
    int total_size_;
#ifdef ENABLE_CUDNN
    cudnnTensorDescriptor_t desc_;
#endif

public:
    tensor(const std::vector<int>& shape);
    virtual ~tensor();

    // Upload data from host to device
    virtual void upload(const TT* host_data);

    // Download data from device to host
    virtual void download(TT* host_data) const;

    // Get device pointer
    TT* data();
    const TT* data() const;

    // Get shape
    const std::vector<int>& shape() const;

    // Get total size
    int size() const;

    // Get size of a specific dimension
    int size(int dim) const;

    // Resize and allocate new memory
    tensor<TT>& reshape(const std::vector<int>& new_shape);

    // Fill tensor with a value
    tensor<TT>& fill(TT value);

    // Fill tensor with zero
    tensor<TT>& zero();

    // Delete copy constructor and copy assignment operator
    tensor(const tensor&) = delete;
    tensor& operator=(const tensor&) = delete;

    // Implement move constructor and move assignment operator
    tensor(tensor&& other) noexcept;
    tensor& operator=(tensor&& other) noexcept;

    // Create a deep copy of the tensor
    tensor<TT> clone() const;

    // In-place addition: a += b
    tensor<TT>& operator+=(const tensor<TT>& other);

    // Element-wise equality: returns a binary tensor mask (uint8_t)
    tensor<uint8_t> operator==(const tensor<TT>& other) const;

    // Exact tensor equality: shape and all values must match bitwise
    bool equals(const tensor<TT>& other) const;

    // Approximate equality for floats (e.g., for __half, float, bfloat16)
    bool approx_equal(const tensor<TT>& other, float atol = 1e-5f) const;

    // Narrow tensor along `dim`, starting at `start`, taking `length` entries
    tensor<TT>& narrow(int dim, int start, int length);

    // Slice tensor along `dim` from `start` (inclusive) to `end` (exclusive)
    tensor<TT>& slice(int dim, int start, int end);    

    // Transpose tensor in-place, preseving N in NHWC <-> NCHW
    tensor<TT>& transpose();

    // Permute tensor in-place, transpose dimensions according to perm
    tensor<TT>& permute(const std::vector<int>& perm);

    // Output tensor to string
    std::string to_string() const;

    // Serialize tensor to stream
    void save(std::ostream& out) const {
        int ndim = static_cast<int>(shape_.size());
        out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        out.write(reinterpret_cast<const char*>(shape_.data()), ndim * sizeof(int));
        int total = size();
        std::vector<TT> host_data(total);
        download(host_data.data());
        out.write(reinterpret_cast<const char*>(host_data.data()), total * sizeof(TT));
    }

    // Deserialize tensor from stream
    void load(std::istream& in) {
        int ndim;
        in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<int> shape(ndim);
        in.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int));
        *this = tensor<TT>(shape);
        int total = size();
        std::vector<TT> host_data(total);
        in.read(reinterpret_cast<char*>(host_data.data()), total * sizeof(TT));
        upload(host_data.data());
    }

#ifdef ENABLE_CUDNN
    // cuDNN descriptor
    cudnnTensorDescriptor_t desc() const;

    // cuDNN data type
    cudnnDataType_t dnn_type() const;

    // cuBLAS data type
    cudaDataType_t blas_type() const;
#endif
};

// Free functions
template<typename TT>
tensor<TT> operator+(const tensor<TT>& a, const tensor<TT>& b);

// Concrete aliases for commonly used tensor types
using tensorf   = tensor<float>;
using tensorh   = tensor<__half>;
using tensorbf  = tensor<__nv_bfloat16>;
using tensori  = tensor<int>;
using tensori8  = tensor<int8_t>;
using tensoru8  = tensor<uint8_t>;

} // namespace dnn 
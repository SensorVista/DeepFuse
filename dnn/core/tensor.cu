#include "dnn/core/tensor.cuh"

#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace dnn::utils;

template<typename TT>
__global__ void fill_kernel(TT* data, TT value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride)
        data[i] = value;
}

template <typename TT>
__global__ void add_kernel(const TT* a, const TT* b, TT* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}

template <typename TT>
__global__ void add_inplace_kernel(TT* a, const TT* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] += b[idx];
}

template <typename TT>
__global__ void equal_kernel(const TT* a, const TT* b, uint8_t* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = (a[idx] == b[idx]) ? 1 : 0;
}

template <typename TT>
__global__ void approx_equal_kernel(const TT* a, const TT* b, uint8_t* out, int size, float atol) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = fabsf(static_cast<float>(a[idx]) - static_cast<float>(b[idx]));
        out[idx] = (diff <= atol) ? 1 : 0;
    }
}

template <typename TT>
__global__ void slice_kernel(const TT* __restrict__ input,
                             TT* __restrict__ output,
                             int outer_dim,
                             int slice_dim,
                             int inner_dim,
                             int start,
                             int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_dim * length * inner_dim;
    if (idx >= total) return;

    int inner = idx % inner_dim;
    int mid = (idx / inner_dim) % length;
    int outer = idx / (length * inner_dim);

    int in_idx = outer * slice_dim * inner_dim + (start + mid) * inner_dim + inner;
    output[idx] = input[in_idx];
}

namespace dnn {

template<typename TT>
tensor<TT>::tensor(const std::vector<int>& shape)
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
    cudaMalloc(&data_, total_size_ * sizeof(TT));

#ifdef ENABLE_CUDNN
    cudnnCreateTensorDescriptor(&desc_);

    // Use Nd descriptor for general case
    std::vector<int> dims = shape_;
    std::vector<int> strides(dims.size(), 1);
    for (int i = dims.size() - 2; i >= 0; --i)
        strides[i] = strides[i+1] * dims[i+1];

    // Type-to-cuDNN mapping (inline)
    cudnnDataType_t dtype;
    if (std::is_same<TT, float>::value)        dtype = CUDNN_DATA_FLOAT;
    else if (std::is_same<TT, __half>::value)       dtype = CUDNN_DATA_HALF;
    else if (std::is_same<TT, __nv_bfloat16>::value)dtype = CUDNN_DATA_BFLOAT16;
    else if (std::is_same<TT, int8_t>::value)       dtype = CUDNN_DATA_INT8;
    else if (std::is_same<TT, uint8_t>::value)      dtype = CUDNN_DATA_UINT8;
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
template<typename TT>
cudnnTensorDescriptor_t tensor<TT>::desc() const { 
    return desc_; 
}

template<typename TT>
cudnnDataType_t tensor<TT>::dnn_type() const { 
    if constexpr (std::is_same_v<TT, float>)
        return CUDNN_DATA_FLOAT;
    else if constexpr (std::is_same_v<TT, __half>)
        return CUDNN_DATA_HALF;
    else if constexpr (std::is_same_v<TT, __nv_bfloat16>)
        return CUDNN_DATA_BFLOAT16;
#ifdef __CUDA_FP8_TYPES_EXIST__
    else if constexpr (std::is_same_v<TT, __nv_fp8x4_e5m2>)
        return CUDNN_DATA_FP8_E5M2;
    else if constexpr (std::is_same_v<TT, __nv_fp8x4_e4m3>)
        return CUDNN_DATA_FP8_E4M3;
#endif
    else if constexpr (std::is_same_v<TT, int8_t>)
        return CUDNN_DATA_INT8;
    else if constexpr (std::is_same_v<TT, uint8_t>)
        return CUDNN_DATA_UINT8;
    else
        throw std::runtime_error("Unsupported data type for cuDNN");
}

template<typename TT>
cudaDataType_t tensor<TT>::blas_type() const {
    if constexpr (std::is_same_v<TT, float>)
        return CUDA_R_32F;
    else if constexpr (std::is_same_v<TT, __half>)
        return CUDA_R_16F;
    else if constexpr (std::is_same_v<TT, __nv_bfloat16>)
        return CUDA_R_16BF;
#ifdef __CUDA_FP8_TYPES_EXIST__
    else if constexpr (std::is_same_v<TT, __nv_fp8x4_e5m2>)
        return CUDA_R_8F_E5M2;
    else if constexpr (std::is_same_v<TT, __nv_fp8x4_e4m3>)
        return CUDA_R_8F_E4M3;
#endif
    else if constexpr (std::is_same_v<TT, int8_t>)
        return CUDA_R_8I;
    else if constexpr (std::is_same_v<TT, uint8_t>)
        return CUDA_R_8U;
    else
        throw std::runtime_error("Unsupported data type for cuBLAS");
}
#endif

template<typename TT>
tensor<TT>::~tensor() {
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

template<typename TT>
void tensor<TT>::upload(const TT* host_data) {
    CHECK_CUDA_EX(cudaMemcpy(data_, host_data, total_size_ * sizeof(TT), cudaMemcpyHostToDevice));
}

template<typename TT>
void tensor<TT>::download(TT* host_data) const {
    CHECK_CUDA_EX(cudaMemcpy(host_data, data_, total_size_ * sizeof(TT), cudaMemcpyDeviceToHost));
}

template<typename TT>
TT* tensor<TT>::data() { return data_; }

template<typename TT>
const TT* tensor<TT>::data() const { return data_; }

template<typename TT>
const std::vector<int>& tensor<TT>::shape() const { return shape_; }

template<typename TT>
int tensor<TT>::size() const { return total_size_; }

template<typename TT>
int tensor<TT>::size(int dim) const {
    if (dim >= size()) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape_[dim];
}

template<typename TT>
tensor<TT>& tensor<TT>::reshape(const std::vector<int>& new_shape) {
    int new_total = 1;
    for (int d : new_shape) new_total *= d;

    if (new_total != total_size_) {
        std::ostringstream oss;
        oss << "reshape: size mismatch, current=";
        oss << "[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            oss << shape_[i];
            if (i + 1 < shape_.size()) oss << ",";
        }
        oss << "] new=";
        oss << "[";
        for (size_t i = 0; i < new_shape.size(); ++i) {
            oss << new_shape[i];
            if (i + 1 < new_shape.size()) oss << ",";
        }
        oss << "]";
        throw std::runtime_error(oss.str());
    }

    shape_ = new_shape;
    return *this;
}

template<typename TT>
tensor<TT>& tensor<TT>::fill(TT value) {
    int threads = 256;
    int blocks = (total_size_ + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(data_, value, total_size_);
    CHECK_CUDA_EX(cudaDeviceSynchronize());
    return *this;
}

template<typename TT>
tensor<TT>& tensor<TT>::zero() {
    return fill(static_cast<TT>(0));
}

template<typename TT>
tensor<TT>::tensor(tensor&& other) noexcept 
    : shape_(std::move(other.shape_))
    , data_(other.data_)
    , total_size_(other.total_size_)
#ifdef ENABLE_CUDNN
    , desc_(other.desc_) 
#endif
{
    other.data_ = nullptr; // Nullify the source pointer
    other.total_size_ = 0; // Reset total_size_
    other.shape_.clear();  // Clear shape_
#ifdef ENABLE_CUDNN
    other.desc_ = nullptr; // Nullify the source descriptor
#endif
}

template<typename TT>
tensor<TT>& tensor<TT>::operator=(tensor&& other) noexcept {
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
        other.total_size_ = 0; // Reset total_size_
        other.shape_.clear();  // Clear shape_
#ifdef ENABLE_CUDNN
        desc_ = other.desc_; // Transfer descriptor
        other.desc_ = nullptr; // Nullify the source descriptor
#endif
    }
    return *this;
}

template <typename TT>
tensor<uint8_t> tensor<TT>::operator==(const tensor<TT>& other) const {
    if (shape_ != other.shape_)
        throw std::invalid_argument("Shape mismatch in operator==");

    tensor<uint8_t> out(shape_);
    int threads = 256;
    int blocks = (total_size_ + threads - 1) / threads;
    equal_kernel<<<blocks, threads>>>(data_, other.data_, out.data(), total_size_);
    CHECK_CUDA_EX(cudaDeviceSynchronize());
    return out;
}

template <typename TT>
bool tensor<TT>::equals(const tensor<TT>& other) const {
    if (shape_ != other.shape_ || total_size_ != other.total_size_)
        return false;

    std::vector<TT> host_a(total_size_);
    std::vector<TT> host_b(total_size_);
    download(host_a.data());
    other.download(host_b.data());

    for (int i = 0; i < total_size_; ++i) {
        // Handle half-precision types that don't have host-side comparison operators
        if constexpr (std::is_same_v<TT, __half>) {
            if (__half2float(host_a[i]) != __half2float(host_b[i]))
                return false;
        } else if constexpr (std::is_same_v<TT, __nv_bfloat16>) {
            if (__bfloat162float(host_a[i]) != __bfloat162float(host_b[i]))
                return false;
        } else {
            if (host_a[i] != host_b[i])
                return false;
        }
    }

    return true;
}

template <typename TT>
bool tensor<TT>::approx_equal(const tensor<TT>& other, float atol) const {
    if (shape_ != other.shape_ || total_size_ != other.total_size_)
        return false;

    tensor<uint8_t> mask(shape_);
    int threads = 256;
    int blocks = (total_size_ + threads - 1) / threads;
    approx_equal_kernel<<<blocks, threads>>>(data_, other.data_, mask.data(), total_size_, atol);
    CHECK_CUDA_EX(cudaDeviceSynchronize());

    std::vector<uint8_t> host_mask(total_size_);
    mask.download(host_mask.data());

    for (uint8_t val : host_mask)
        if (val == 0) return false;

    return true;
}

template<typename TT>
tensor<TT> tensor<TT>::clone() const {
    tensor<TT> out(shape_);
    CHECK_CUDA_EX(cudaMemcpy(out.data_, data_, total_size_ * sizeof(TT), cudaMemcpyDeviceToDevice));
    return out;
}

template<typename TT>
std::string tensor<TT>::to_string() const {
    std::ostringstream oss;
    oss << "tensor: [";
    for (int i = 0; i < shape_.size(); ++i)
        oss << shape_[i] << (i < shape_.size() - 1 ? ", " : "]");
    return oss.str();
}

// ------------------------------------------
// Operator +
// ------------------------------------------
template<typename TT>
tensor<TT> operator+(const tensor<TT>& a, const tensor<TT>& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shape mismatch in operator+");
    }

    tensor<TT> out(a.shape());
    int size = a.size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(a.data(), b.data(), out.data(), size);
    CHECK_CUDA_EX(cudaDeviceSynchronize());
    
    return out;
}

// Operator +=
template<typename TT>
tensor<TT>& tensor<TT>::operator+=(const tensor<TT>& other) {
    if (shape_  != other.shape()) {
        throw std::runtime_error("Shape mismatch in operator+=");
    }

    int size = total_size_;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    add_inplace_kernel<<<blocks, threads>>>(data_, other.data_, size);
    CHECK_CUDA_EX(cudaDeviceSynchronize());

    return *this;
}

// Narrow
template <typename TT>
tensor<TT>& tensor<TT>::narrow(int dim, int start, int length) {
    if (dim < 0 || dim >= shape_.size())
        throw std::invalid_argument("Invalid dimension in narrow()");
    if (start < 0 || start + length > shape_[dim])
        throw std::invalid_argument("Start/length out of bounds in narrow()");

    std::vector<int> new_shape = shape_;
    new_shape[dim] = length;

    int outer_dim = 1;
    for (int i = 0; i < dim; ++i) outer_dim *= shape_[i];

    int inner_dim = 1;
    for (int i = dim + 1; i < shape_.size(); ++i) inner_dim *= shape_[i];

    tensor<TT> temp(new_shape);
    int total = outer_dim * length * inner_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    slice_kernel<<<blocks, threads>>>(
        data_, temp.data(), outer_dim, shape_[dim], inner_dim, start, length
    );
    CHECK_CUDA_EX(cudaDeviceSynchronize());

    // Move new data into current tensor
    CHECK_CUDA_EX(cudaFree(data_));
    data_ = temp.data_;
    temp.data_ = nullptr;

    shape_ = std::move(new_shape);
    total_size_ = total;
    return *this;
}

// Slice
template <typename TT>
tensor<TT>& tensor<TT>::slice(int dim, int start, int end) {
    if (dim < 0 || dim >= shape_.size())
        throw std::invalid_argument("Invalid dimension in slice()");
    if (start < 0 || end > shape_[dim] || start >= end)
        throw std::invalid_argument("Invalid start/end in slice()");
    return this->narrow(dim, start, end - start);
}

// Forward declarations for supported template types
// FP32
template struct tensor<float>;
template tensor<float> operator+(const tensor<float>&, const tensor<float>&);

// FP16
template struct tensor<__half>;
template tensor<__half> operator+(const tensor<__half>&, const tensor<__half>&);

// BF16
template struct tensor<__nv_bfloat16>;
template tensor<__nv_bfloat16> operator+(const tensor<__nv_bfloat16>&, const tensor<__nv_bfloat16>&);

// Int8
template struct tensor<int8_t>;
template tensor<int8_t> operator+(const tensor<int8_t>&, const tensor<int8_t>&);

// UInt8
template struct tensor<uint8_t>;
template tensor<uint8_t> operator+(const tensor<uint8_t>&, const tensor<uint8_t>&);

// Int32
template struct tensor<int>;
template tensor<int> operator+(const tensor<int>&, const tensor<int>&);

#ifdef __CUDA_FP8_TYPES_EXIST__
// Use FP8 Tensor Core Hopper+ (SM 9.0+)
// FP8E5M2
template struct tensor<__nv_fp8x4_e5m2>;
template tensor<__nv_fp8x4_e5m2> operator+(const tensor<__nv_fp8x4_e5m2>&, const tensor<__nv_fp8x4_e5m2>&);

// FP8E4M3
template struct tensor<__nv_fp8x4_e4m3>;
template tensor<__nv_fp8x4_e4m3> operator+(const tensor<__nv_fp8x4_e4m3>&, const tensor<__nv_fp8x4_e4m3>&);
#endif
} // namespace dnn


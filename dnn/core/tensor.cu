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

template<typename T>
__global__ void fill_kernel(T* data, T value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride)
        data[i] = value;
}

namespace lenet5 {

template<typename T>
tensor<T>::tensor(const std::vector<size_t>& shape) : shape_(shape) {
    total_size_ = 1;
    for (size_t dim : shape) {
        total_size_ *= dim;
    }
    utils::CHECK_CUDA_EX(cudaMalloc(&device_ptr_, total_size_ * sizeof(T)));
}

template<typename T>
tensor<T>::~tensor() {
    if (device_ptr_) {
        utils::CHECK_CUDA_EX(cudaFree(device_ptr_));
    }
}

template<typename T>
void tensor<T>::upload(const T* host_data) {
    utils::CHECK_CUDA_EX(cudaMemcpy(device_ptr_, host_data, total_size_ * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void tensor<T>::download(T* host_data) const {
    utils::CHECK_CUDA_EX(cudaMemcpy(host_data, device_ptr_, total_size_ * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
T* tensor<T>::data() { return device_ptr_; }

template<typename T>
const T* tensor<T>::data() const { return device_ptr_; }

template<typename T>
const std::vector<size_t>& tensor<T>::shape() const { return shape_; }

template<typename T>
size_t tensor<T>::size() const { return total_size_; }

template<typename T>
size_t tensor<T>::size(size_t dim) const {
    if (dim >= shape_.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape_[dim];
}

template<typename T>
void tensor<T>::reshape(const std::vector<size_t>& new_shape) {
    if (device_ptr_) cudaFree(device_ptr_);
    shape_ = new_shape;
    total_size_ = 1;
    for (size_t d : shape_) total_size_ *= d;
    utils::CHECK_CUDA_EX(cudaMalloc(&device_ptr_, total_size_ * sizeof(T)));
}

template<typename T>
void tensor<T>::fill(T value) {
    int threads = 256;
    int blocks = (total_size_ + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(device_ptr_, value, total_size_);
    utils::CHECK_CUDA_EX(cudaDeviceSynchronize());
}

template<typename T>
void tensor<T>::zero() {
    fill(static_cast<T>(0));
}

template<typename T>
tensor<T>::tensor(tensor&& other) noexcept 
    : shape_(std::move(other.shape_)), device_ptr_(other.device_ptr_), total_size_(other.total_size_) {
    other.device_ptr_ = nullptr; // Nullify the source pointer
}

template<typename T>
tensor<T>& tensor<T>::operator=(tensor&& other) noexcept {
    if (this != &other) {
        // Free existing resources
        if (device_ptr_) {
            utils::CHECK_CUDA_EX(cudaFree(device_ptr_));
        }
        // Transfer ownership
        shape_ = std::move(other.shape_);
        device_ptr_ = other.device_ptr_;
        total_size_ = other.total_size_;
        other.device_ptr_ = nullptr; // Nullify the source pointer
    }
    return *this;
}

template<typename T>
void tensor<T>::copy_from(const tensor<T>& src) {
    if (total_size_ != src.total_size_)
        throw std::invalid_argument("Mismatched sizes in copy_from");
    utils::CHECK_CUDA_EX(cudaMemcpy(device_ptr_, src.device_ptr_, total_size_ * sizeof(T), cudaMemcpyDeviceToDevice));
}

template<typename T>
std::string tensor<T>::to_string() const {
    std::ostringstream oss;
    oss << "tensor: [";
    for (size_t i = 0; i < shape_.size(); ++i)
        oss << shape_[i] << (i < shape_.size() - 1 ? ", " : "]");
    return oss.str();
}

// Explicit template instantiations
template class tensor<float>;  // FP32
template class tensor<__half>; // FP16
template class tensor<__nv_bfloat16>; // BF16
template class tensor<int8_t>; // Quantized tensor

} // namespace lenet5 
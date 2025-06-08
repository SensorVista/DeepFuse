#include "activation_layer.cuh"

#include "../utils/common.cuh"
#include "../core/device.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

template<typename T>
__device__ T gelu(T x) {
    if constexpr (std::is_same_v<T, __half>) {
        float x_float = __half2float(x);
        const float k = 0.7978845608f; // sqrt(2/pi)
        float result = x_float * 0.5f * (1.0f + tanhf(k * (x_float + 0.044715f * x_float * x_float * x_float)));
        return __float2half(result);
    } else {
        const T k = T(0.7978845608); // sqrt(2/pi)
        return x * T(0.5) * (T(1) + tanh(k * (x + T(0.044715) * x * x * x)));
    }
}

template<typename T>
__global__ void gelu_forward_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu(input[idx]);
    }
}

template<typename T>
__global__ void gelu_backward_kernel(const T* grad_output, const T* input, T* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, __half>) {
            float x = __half2float(input[idx]);
            const float c0 = 0.044715f;
            const float sqrt_2_over_pi = 0.7978845608f;
            float x3 = x * x * x;
            float tanh_arg = sqrt_2_over_pi * (x + c0 * x3);
            float tanh_val = tanhf(tanh_arg);
            float sech2 = 1.0f - tanh_val * tanh_val;
            float dy_dx = 0.5f * (1.0f + tanh_val + x * sech2 * sqrt_2_over_pi * (1.0f + 3.0f * c0 * x * x));
            grad_input[idx] = __float2half(__half2float(grad_output[idx]) * dy_dx);
        } else {
            T x = input[idx];
            const T c0 = T(0.044715);
            const T sqrt_2_over_pi = T(0.7978845608);
            T x3 = x * x * x;
            T tanh_arg = sqrt_2_over_pi * (x + c0 * x3);
            T tanh_val = tanh(tanh_arg);
            T sech2 = T(1) - tanh_val * tanh_val;
            T dy_dx = T(0.5) * (T(1) + tanh_val + x * sech2 * sqrt_2_over_pi * (T(1) + T(3) * c0 * x * x));
            grad_input[idx] = grad_output[idx] * dy_dx;
        }
    }
}

#ifndef ENABLE_CUDNN

// CUDA kernels for different activation functions
template<typename T>
__global__ void relu_forward_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, __half>) {
            output[idx] = __hgt(input[idx], __float2half(0.0f)) ? input[idx] : __float2half(0.0f);
        } else {
            output[idx] = max(static_cast<T>(0.0f), input[idx]);
        }
    }
}

template<typename T>
__global__ void relu_backward_kernel(const T* grad_output, const T* input, T* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, __half>) {
            grad_input[idx] = __hgt(input[idx], __float2half(0.0f)) ? grad_output[idx] : __float2half(0.0f);
        } else {
            grad_input[idx] = (input[idx] > static_cast<T>(0.0f)) ? grad_output[idx] : static_cast<T>(0.0f);
        }
    }
}

template<typename T>
__global__ void sigmoid_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, __half>) {
            float x = __half2float(input[idx]);
            float result = 1.0f / (1.0f + expf(-x));
            output[idx] = __float2half(result);
        } else {
            output[idx] = T(1.0) / (T(1.0) + exp(-input[idx]));
        }
    }
}

template<typename T>
__global__ void sigmoid_derivative_kernel(const T* grad_output, const T* input, T* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, __half>) {
            float x = __half2float(input[idx]);
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            float result = __half2float(grad_output[idx]) * sigmoid_x * (1.0f - sigmoid_x);
            grad_input[idx] = __float2half(result);
        } else {
            T sigmoid_x = T(1.0) / (T(1.0) + exp(-input[idx]));
            grad_input[idx] = grad_output[idx] * sigmoid_x * (T(1.0) - sigmoid_x);
        }
    }
}

template<typename T>
__global__ void tanh_forward_kernel(T* output, const T* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, __half>) {
            float x = __half2float(input[idx]);
            float result = 1.7159f * tanhf(2.0f/3.0f * x);
            output[idx] = __float2half(result);
        } else {
            output[idx] = static_cast<T>(1.7159f * tanh(2.0f/3.0f * input[idx]));
        }
    }
}

template<typename T>
__global__ void tanh_backward_kernel(T* grad_input, const T* grad_output, const T* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, __half>) {
            float x = __half2float(input[idx]);
            float tanh_sx = tanhf(2.0f/3.0f * x);
            float activation_derivative = 1.7159f * (2.0f/3.0f) * (1.0f - tanh_sx * tanh_sx);
            float result = __half2float(grad_output[idx]) * activation_derivative;
            grad_input[idx] = __float2half(result);
        } else {
            T tanh_sx = tanh(2.0f/3.0f * static_cast<float>(input[idx]));
            T activation_derivative = static_cast<T>(1.7159f * (2.0f/3.0f) * (1.0f - static_cast<float>(tanh_sx * tanh_sx)));
            grad_input[idx] = grad_output[idx] * activation_derivative;
        }
    }
}
#endif

namespace dnn {

template<typename T>
ActivationLayer<T>::ActivationLayer(ActivationType type) : type_(type) {
#ifdef ENABLE_CUDNN
    utils::CHECK_CUDNN_EX(cudnnCreateActivationDescriptor(&act_desc_));
    cudnnActivationMode_t mode;
    switch (type) {
        case ActivationType::ReLU:
            mode = CUDNN_ACTIVATION_RELU;
            break;
        case ActivationType::Sigmoid:
            mode = CUDNN_ACTIVATION_SIGMOID;
            break;
        case ActivationType::Tanh:
            mode = CUDNN_ACTIVATION_TANH;
            break;
        case ActivationType::ClippedReLU:
            mode = CUDNN_ACTIVATION_CLIPPED_RELU;
            break;
        case ActivationType::Elu:
            mode = CUDNN_ACTIVATION_ELU;
            break;
        default:
            throw std::runtime_error("Unsupported activation type");
    }
    utils::CHECK_CUDNN_EX(cudnnSetActivationDescriptor(act_desc_, mode, CUDNN_NOT_PROPAGATE_NAN, 0.0));
#endif
}

// Helper constexpr mapping
constexpr const char* activation_type_to_string(ActivationType type) {
    switch (type) {
        case ActivationType::ReLU: return "ReLU";
        case ActivationType::Sigmoid: return "Sigmoid";
        case ActivationType::Tanh: return "Tanh";
        case ActivationType::ClippedReLU: return "ClippedReLU";
        case ActivationType::Elu: return "Elu";
        case ActivationType::GELU: return "GELU";
        default: return "Unknown";
    }
}

template<typename T>
std::string ActivationLayer<T>::name() const { return "Activation-" + std::string(activation_type_to_string(type_)); }

template<typename T>
ActivationLayer<T>::~ActivationLayer() {
#ifdef ENABLE_CUDNN
    utils::CHECK_CUDNN_EX(cudnnDestroyActivationDescriptor(act_desc_));
#endif
}

template<typename T>
tensor<T> ActivationLayer<T>::forward(const tensor<T>& input) {
    tensor<T> output(input.shape());
    int size = input.size();
    
    if (size == 0) return output;

#ifdef ENABLE_CUDNN
    const float alpha = 1.0f, beta = 0.0f;
    auto handle = Cuda::current().cudnn();
    utils::CHECK_CUDNN_EX(cudnnActivationForward(handle, act_desc_, &alpha, 
        input.desc(), input.data(), &beta, output.desc(), output.data()));
#else
    const int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (type_) {
        case ActivationType::ReLU:
            relu_forward_kernel<<<num_blocks, BLOCK_SIZE>>>(input.data(), output.data(), size);
            break;
        case ActivationType::Sigmoid:
            sigmoid_kernel<<<num_blocks, BLOCK_SIZE>>>(input.data(), output.data(), size);
            break;
        case ActivationType::Tanh:
            tanh_forward_kernel<<<num_blocks, BLOCK_SIZE>>>(output.data(), input.data(), size);
            break;
        case ActivationType::GELU:
            gelu_forward_kernel<<<num_blocks, BLOCK_SIZE>>>(input.data(), output.data(), size);
            break;            
        default:
            throw std::runtime_error("Unsupported activation type");
    }
    utils::THROW_CUDA_EX();
#endif

    return output;
}

template<typename T>
tensor<T> ActivationLayer<T>::backward(const tensor<T>& grad_output, const tensor<T>& input) {
    tensor<T> grad_input(grad_output.shape());
    int size = grad_output.size();

    if (size == 0) return grad_input;

#ifdef ENABLE_CUDNN
    const float alpha = 1.0f, beta = 0.0f;
    auto handle = Cuda::current().cudnn();
    utils::CHECK_CUDNN_EX(cudnnActivationBackward(handle, act_desc_, &alpha,
        grad_output.desc(), grad_output.data(), grad_output.desc(), grad_output.data(),
        input.desc(), input.data(), &beta, grad_input.desc(), grad_input.data()));
#else
    const int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (type_) {
        case ActivationType::ReLU:
            relu_backward_kernel<<<num_blocks, BLOCK_SIZE>>>(grad_output.data(), input.data(), grad_input.data(), size);
            break;
        case ActivationType::Sigmoid:
            sigmoid_derivative_kernel<<<num_blocks, BLOCK_SIZE>>>(grad_output.data(), input.data(), grad_input.data(), size);
            break;
        case ActivationType::Tanh:
            tanh_backward_kernel<<<num_blocks, BLOCK_SIZE>>>(grad_input.data(), grad_output.data(), input.data(), size);
            break;
        case ActivationType::GELU:
            gelu_backward_kernel<<<num_blocks, BLOCK_SIZE>>>(grad_output.data(), input.data(), grad_input.data(), size);
            break;
        default:
            throw std::runtime_error("Unsupported activation type");
    }
    utils::THROW_CUDA_EX();
#endif

    return grad_input;
}

// Explicit template instantiations
template class ActivationLayer<float>;
template class ActivationLayer<__half>;

} // namespace dnn 
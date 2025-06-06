#include "common.cuh"

namespace dnn::utils {

// Positive infinity
template<typename T>
__device__ __host__ inline T infinity() {
    if constexpr (std::is_same_v<T, float>)
        return INFINITY;
    else if constexpr (std::is_same_v<T, double>)
        return INFINITY;
    else if constexpr (std::is_same_v<T, __half>)
        return __half(INFINITY);
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return __nv_bfloat16(INFINITY);
    else if constexpr (std::is_same_v<T, int8_t>)
        return 127;
    else if constexpr (std::is_same_v<T, uint8_t>)
        return 255;
    else {
        static_assert(sizeof(T) == 0, "infinity<T>() not implemented.");
        return T();
    }
}

// Negative infinity
template<typename T>
__device__ __host__ inline T neg_infinity() {
    if constexpr (std::is_same_v<T, float>)
        return -INFINITY;
    else if constexpr (std::is_same_v<T, double>)
        return -INFINITY;
    else if constexpr (std::is_same_v<T, __half>)
        return __half(-INFINITY);
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return __nv_bfloat16(-INFINITY);
    else if constexpr (std::is_same_v<T, int8_t>)
        return -128;
    else if constexpr (std::is_same_v<T, uint8_t>)
        return 0;
    else {
        static_assert(sizeof(T) == 0, "neg_infinity<T>() not implemented.");
        return T();
    }
}

template __device__ __host__ float dnn::utils::neg_infinity<float>();
template __device__ __host__ double dnn::utils::neg_infinity<double>();
template __device__ __host__ __half dnn::utils::neg_infinity<__half>();
template __device__ __host__ __nv_bfloat16 dnn::utils::neg_infinity<__nv_bfloat16>();
template __device__ __host__ int8_t dnn::utils::neg_infinity<int8_t>();
template __device__ __host__ uint8_t dnn::utils::neg_infinity<uint8_t>();

template __device__ __host__ float dnn::utils::infinity<float>();
template __device__ __host__ double dnn::utils::infinity<double>();
template __device__ __host__ __half dnn::utils::infinity<__half>();
template __device__ __host__ __nv_bfloat16 dnn::utils::infinity<__nv_bfloat16>();
template __device__ __host__ int8_t dnn::utils::infinity<int8_t>();
template __device__ __host__ uint8_t dnn::utils::infinity<uint8_t>();

}
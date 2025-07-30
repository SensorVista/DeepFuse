#include "common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>

namespace dnn::utils {

// Positive infinity
template<typename T>
__device__ __host__ inline T infinity() {
    if constexpr (std::is_same_v<T, float>)
        return __builtin_huge_valf();
    else if constexpr (std::is_same_v<T, double>)
        return __builtin_huge_val();
    else if constexpr (std::is_same_v<T, __half>)
        return __half_raw{ 0x7C00 };  // +inf in IEEE 754 half
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return __nv_bfloat16_raw{ 0x7F80 };  // +inf in bfloat16
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
        return -__builtin_huge_valf();
    else if constexpr (std::is_same_v<T, double>)
        return -__builtin_huge_val();
    else if constexpr (std::is_same_v<T, __half>)
        return __half_raw{ 0xFC00 };  // -inf in IEEE 754 half
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return __nv_bfloat16_raw{ 0xFF80 };  // -inf in bfloat16
    else if constexpr (std::is_same_v<T, int8_t>)
        return -128;
    else if constexpr (std::is_same_v<T, uint8_t>)
        return 0;
    else {
        static_assert(sizeof(T) == 0, "neg_infinity<T>() not implemented.");
        return T();
    }
}

template __device__ __host__ float neg_infinity<float>();
template __device__ __host__ double neg_infinity<double>();
template __device__ __host__ __half neg_infinity<__half>();
template __device__ __host__ __nv_bfloat16 neg_infinity<__nv_bfloat16>();
template __device__ __host__ int8_t neg_infinity<int8_t>();
template __device__ __host__ uint8_t neg_infinity<uint8_t>();

template __device__ __host__ float infinity<float>();
template __device__ __host__ double infinity<double>();
template __device__ __host__ __half infinity<__half>();
template __device__ __host__ __nv_bfloat16 infinity<__nv_bfloat16>();
template __device__ __host__ int8_t infinity<int8_t>();
template __device__ __host__ uint8_t infinity<uint8_t>();

} // namespace dnn::utils
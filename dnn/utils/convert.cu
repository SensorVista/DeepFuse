#include "common.cuh"

namespace dnn::utils {

template<typename T>
__device__ __host__ inline float to_float(T v) { 
    if constexpr (std::is_same_v<T, float>)
        return v;
    else if constexpr (std::is_same_v<T, __half>)
        return __half2float(v);
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return __nv_bfloat162float(v);
    else if constexpr (std::is_same_v<T, int8_t>)
        return v;
}

template<typename T>
__device__ __host__ inline __half to_half(T v) { 
    if constexpr (std::is_same_v<T, __half>)
        return v;
    else if constexpr (std::is_same_v<T, float>)
        return __float2half(v);
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return __nv_bfloat162half(v);
    else if constexpr (std::is_same_v<T, int8_t>)
        return v;
}

template __device__ __host__ float to_float<__half>(__half);
template __device__ __host__ float to_float<float>(float);

template __device__ __host__ __half to_half<__half>(__half);
template __device__ __host__ __half to_half<float>(float);

} // namespace dnn::utils
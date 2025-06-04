#include "common.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef __CUDA_FP8_TYPES_EXIST__
#include <cuda_fp8.h>
#endif

#include <cstdint>

// Host-side constants
static const float one_f = 1.0f;
static const float zero_f = 0.0f;

static const __half one_h = __float2half(1.0f);
static const __half zero_h = __float2half(0.0f);

static const __nv_bfloat16 one_bf = __float2bfloat16(1.0f);
static const __nv_bfloat16 zero_bf = __float2bfloat16(0.0f);

static const int8_t one_i8 = 1;
static const int8_t zero_i8 = 0;

static const uint8_t one_u8 = 1;
static const uint8_t zero_u8 = 0;


namespace dnn::utils {
// Specializations
template<> const void* one<float>()         { return &one_f; }
template<> const void* zero<float>()        { return &zero_f; }

template<> const void* one<__half>()        { return &one_h; }
template<> const void* zero<__half>()       { return &zero_h; }

template<> const void* one<__nv_bfloat16>() { return &one_bf; }
template<> const void* zero<__nv_bfloat16>(){ return &zero_bf; }

template<> const void* one<int8_t>()        { return &one_i8; }
template<> const void* zero<int8_t>()       { return &zero_i8; }

template<> const void* one<uint8_t>()       { return &one_u8; }
template<> const void* zero<uint8_t>()      { return &zero_u8; }

// Default fallback: disallowed
template<typename T>
const cublasComputeType_t compute_type() {
    static_assert(sizeof(T) == 0, "Unsupported type for cuBLAS compute_type()");
    return CUBLAS_COMPUTE_32F; // Unreachable
}

// FP32 input → FP32 compute
template<>
const cublasComputeType_t compute_type<float>() {
    return CUBLAS_COMPUTE_32F;
}

// FP16 or BF16 input → FP32 compute
template<>
const cublasComputeType_t compute_type<__half>() {
    return CUBLAS_COMPUTE_32F;
}

template<>
const cublasComputeType_t compute_type<__nv_bfloat16>() {
    return CUBLAS_COMPUTE_32F;
}

// INT8 → INT32 accumulation (typical for quantized ops)
template<>
const cublasComputeType_t compute_type<int8_t>() {
    return CUBLAS_COMPUTE_32I;
}

template<>
const cublasComputeType_t compute_type<uint8_t>() {
    return CUBLAS_COMPUTE_32I;
}


} // namespace dnn::utils

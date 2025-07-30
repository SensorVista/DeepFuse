#include "common.cuh"

namespace dnn::utils {

#ifdef ENABLE_CUDNN
// cuDNN data type
template<typename T>
inline const cudnnDataType_t dnn_type() {
    if constexpr (std::is_same_v<T, float>)
        return CUDNN_DATA_FLOAT;
    else if constexpr (std::is_same_v<T, __half>)
        return CUDNN_DATA_HALF;
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return CUDNN_DATA_BFLOAT16;
#ifdef __CUDA_FP8_TYPES_EXIST__
    else if constexpr (std::is_same_v<T, __nv_fp8x4_e5m2>)
        return CUDNN_DATA_FP8_E5M2;
    else if constexpr (std::is_same_v<T, __nv_fp8x4_e4m3>)
        return CUDNN_DATA_FP8_E4M3;
#endif
    else if constexpr (std::is_same_v<T, int8_t>)
        return CUDNN_DATA_INT8;
    else if constexpr (std::is_same_v<T, uint8_t>)
        return CUDNN_DATA_UINT8;
    else
        throw std::runtime_error("Unsupported data type for cuDNN");
}

// cuBLAS data type
template<typename T>
const cudaDataType_t blas_type() {
    if constexpr (std::is_same_v<T, float>)
        return CUDA_R_32F;
    else if constexpr (std::is_same_v<T, __half>)
        return CUDA_R_16F;
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return CUDA_R_16BF;
#ifdef __CUDA_FP8_TYPES_EXIST__
    else if constexpr (std::is_same_v<T, __nv_fp8x4_e5m2>)
        return CUDA_R_8F_E5M2;
    else if constexpr (std::is_same_v<T, __nv_fp8x4_e4m3>)
        return CUDA_R_8F_E4M3;
#endif
    else if constexpr (std::is_same_v<T, int8_t>)
        return CUDA_R_8I;
    else if constexpr (std::is_same_v<T, uint8_t>)
        return CUDA_R_8U;
    else
        throw std::runtime_error("Unsupported data type for cuBLAS");
}
#endif

} // namespace dnn::utils
#include "dnn/core/tensor.cuh"

#include "dnn/utils/common.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

using namespace dnn::utils;

namespace dnn {

// 2D row-major transpose kernel
// Input shape: [M, N], Output shape: [N, M]
template<typename T>
__global__ void transpose_2d(const T* __restrict__ in, T* __restrict__ out, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < M)
        out[x * M + y] = in[y * N + x];
}

// 3D row-major permutation kernel
// Input shape: [D, H, W], Output shape: [D, W, H]
template<typename T>
__global__ void transpose_3d_dhw_dwh(const T* __restrict__ in, T* __restrict__ out, int D, int H, int W) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    if (d < D && h < H && w < W)
        out[d * W * H + w * H + h] = in[d * H * W + h * W + w];
}

// 3D row-major permutation kernel
// Input shape: [H, W, D], Output shape: [W, H, D] (HWD to WHD)
template<typename T>
__global__ void transpose_3d_hwd_whd(const T* __restrict__ in, T* __restrict__ out, int H, int W, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z * blockDim.z + threadIdx.z;
    if (h < H && w < W && d < D)
        out[w * H * D + h * D + d] = in[h * W * D + w * D + d];
}

// 4D row-major permutation kernel
// Input shape: [N, H, W, C], Output shape: [N, C, H, W] (NHWC to NCHW)
template<typename T>
__global__ void transpose_4d_nhwc_nchw(const T* __restrict__ in, T* __restrict__ out, int N, int C, int H, int W) {
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int n = blockIdx.z;
    if (n < N && h < H && w < W) {
        for (int c = 0; c < C; ++c) {
            int in_idx  = n * H * W * C + h * W * C + w * C + c;
            int out_idx = n * C * H * W + c * H * W + h * W + w;
            out[out_idx] = in[in_idx];
        }
    }
}

// 4D row-major permutation kernel
// Input shape: [N, C, H, W], Output shape: [N, H, W, C] (NCHW to NHWC)
template<typename T>
__global__ void transpose_4d_nchw_nhwc(const T* __restrict__ in, T* __restrict__ out, int N, int C, int H, int W) {
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int n = blockIdx.z;
    if (n < N && h < H && w < W) {
        for (int c = 0; c < C; ++c) {
            int in_idx  = n * C * H * W + c * H * W + h * W + w;
            int out_idx = n * H * W * C + h * W * C + w * C + c;
            out[out_idx] = in[in_idx];
        }
    }
}

// 4D row-major permutation kernel
// Input shape: [N, C, H, W], Output shape: [N, W, H, C] (NCHW to NWHC)
template<typename T>
__global__ void transpose_4d_nchw_nwhc(const T* __restrict__ in, T* __restrict__ out, int N, int C, int H, int W) {
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int n = blockIdx.z;
    if (n < N && h < H && w < W) {
        for (int c = 0; c < C; ++c) {
            int in_idx  = n * C * H * W + c * H * W + h * W + w;
            int out_idx = n * W * H * C + w * H * C + h * C + c;
            out[out_idx] = in[in_idx];
        }
    }
}

// 4D row-major permutation kernel
// Input shape: [N, H, W, C], Output shape: [W, H, C, N] (NHWC to WHCN)
template<typename T>
__global__ void transpose_4d_nhwc_whcn(const T* __restrict__ in, T* __restrict__ out, int N, int H, int W, int C) {
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int w = threadIdx.z + blockIdx.z * blockDim.z;
    if (c < C && h < H && w < W) {
        for (int n = 0; n < N; ++n) {
            int in_idx  = n * H * W * C + h * W * C + w * C + c;
            int out_idx = w * H * C * N + h * C * N + c * N + n;
            out[out_idx] = in[in_idx];
        }
    }
}

// 5D row-major permutation kernel
// Input shape: [B, C, D, H, W], Output shape: [B, C, H, D, W] (BCDHW to BCHDW)
template<typename T>
__global__ void transpose_5d_bcdhw_bchdw(const T* __restrict__ in, T* __restrict__ out, int B, int C, int D, int H, int W) {
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int d = threadIdx.z + blockIdx.z * blockDim.z;
    if (w < W && h < H && d < D) {
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                int in_idx  = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                int out_idx = b * C * H * D * W + c * H * D * W + h * D * W + d * W + w;
                out[out_idx] = in[in_idx];
            }
        }
    }
}

// 5D row-major permutation kernel
// Input shape: [B, D, C, H, W], Output shape: [B, C, D, H, W] (BDCHW to BCDHW)
template<typename T>
__global__ void transpose_5d_bdchw_bcdhw(const T* __restrict__ in, T* __restrict__ out, int B, int D, int C, int H, int W) {
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int d = threadIdx.z + blockIdx.z * blockDim.z;
    if (w < W && h < H && d < D) {
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                int in_idx  = b * D * C * H * W + d * C * H * W + c * H * W + h * W + w;
                int out_idx = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                out[out_idx] = in[in_idx];
            }
        }
    }
}

// 5D row-major permutation kernel
// Input shape: [B, C, H, D, W], Output shape: [B, C, D, H, W] (BCHDW to BCDHW)
template<typename T>
__global__ void transpose_5d_bchdw_bcdhw(const T* __restrict__ in, T* __restrict__ out, int B, int C, int D, int H, int W) {
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int d = threadIdx.y + blockIdx.y * blockDim.y;
    int h = threadIdx.z + blockIdx.z * blockDim.z;
    if (w < W && d < D && h < H) {
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                int in_idx  = b * C * H * D * W + c * H * D * W + h * D * W + d * W + w;
                int out_idx = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                out[out_idx] = in[in_idx];
            }
        }
    }
}

// CUDA kernel for N-D axis permutation
template<typename T>
__global__ void transpose_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int* __restrict__ in_strides,
    const int* __restrict__ out_strides,
    const int* __restrict__ perm,
    const int ndim,
    const int total_elems,
    const int* __restrict__ out_shape)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    int coord[8] = { 0 };
    int tmp = idx;

    for (int i = ndim - 1; i >= 0; --i) {
        coord[i] = tmp % out_shape[i];
        tmp /= out_shape[i];
    }

    int src_offset = 0;
    for (int i = 0; i < ndim; ++i) {
        src_offset += coord[i] * in_strides[perm[i]];
    }

    output[idx] = input[src_offset];
}

// Transpose
template<typename T>
tensor<T>& tensor<T>::transpose() {
    int ndim = static_cast<int>(shape_.size());
    if (ndim < 2) return *this;

    std::vector<int> perm(ndim);

    if (ndim == 2) {
        perm[0] = 1;
        perm[1] = 0;
    } else {
        perm[0] = 0;  // Preserve batch dimension
        for (int i = 1, j = ndim - 1; i < ndim; ++i, --j)
            perm[i] = j;
    }

    return this->permute(perm); 
}

// Permute
template<typename T>
tensor<T>& tensor<T>::permute(const std::vector<int>& perm) {
    if (perm.size() != shape_.size())
        throw std::invalid_argument("Permutation dims do not match tensor rank");

    int ndim = static_cast<int>(shape_.size());
    std::vector<int> new_shape(ndim);
    for (int i = 0; i < ndim; ++i)
        new_shape[i] = shape_[perm[i]];

    int total = 1;
    for (int d : shape_) total *= d;

    T* new_data;
    CHECK_CUDA_EX(cudaMalloc(&new_data, total * sizeof(T)));

    dim3 threads(16, 16, 1);
    dim3 blocks;

    if (ndim == 2 && perm[0] == 1 && perm[1] == 0) {
        int M = shape_[0];
        int N = shape_[1];
        blocks = dim3((N + 15) / 16, (M + 15) / 16);
        transpose_2d<T><<<blocks, threads>>>(data_, new_data, M, N);
    }
    else if (ndim == 3 && perm == std::vector<int>({0, 2, 1})) {
        int D = shape_[0], H = shape_[1], W = shape_[2];
        blocks = dim3((W + 7) / 8, (H + 7) / 8, D);
        threads = dim3(8, 8, 1);
        transpose_3d_dhw_dwh<T><<<blocks, threads>>>(data_, new_data, D, H, W);
    }
    else if (ndim == 3 && perm == std::vector<int>({1, 2, 0})) {
        int H = shape_[0], W = shape_[1], D = shape_[2];
        blocks = dim3((D + 7) / 8, (W + 7) / 8, H);
        threads = dim3(8, 8, 1);
        transpose_3d_hwd_whd<T><<<blocks, threads>>>(data_, new_data, H, W, D);
    }
    else if (ndim == 4 && perm == std::vector<int>({0, 3, 1, 2})) {
        int N = shape_[0], H = shape_[1], W = shape_[2], C = shape_[3];
        blocks = dim3((W + 15) / 16, (H + 15) / 16, N);
        transpose_4d_nhwc_nchw<T><<<blocks, threads>>>(data_, new_data, N, C, H, W);
    }
    else if (ndim == 4 && perm == std::vector<int>({0, 2, 3, 1})) {
        int N = shape_[0], C = shape_[1], H = shape_[2], W = shape_[3];
        blocks = dim3((W + 15) / 16, (H + 15) / 16, N);
        transpose_4d_nchw_nhwc<T><<<blocks, threads>>>(data_, new_data, N, C, H, W);
    }
    else {
        std::vector<int> in_strides(ndim, 1);
        for (int i = ndim - 2; i >= 0; --i)
            in_strides[i] = in_strides[i + 1] * shape_[i + 1];

        std::vector<int> out_strides(ndim, 1);
        for (int i = ndim - 2; i >= 0; --i)
            out_strides[i] = out_strides[i + 1] * new_shape[i + 1];

        int* d_in_strides;
        int* d_out_strides;
        int* d_perm;
        int* d_out_shape;

        CHECK_CUDA_EX(cudaMalloc(&d_in_strides, ndim * sizeof(int)));
        CHECK_CUDA_EX(cudaMalloc(&d_out_strides, ndim * sizeof(int)));
        CHECK_CUDA_EX(cudaMalloc(&d_perm, ndim * sizeof(int)));
        CHECK_CUDA_EX(cudaMalloc(&d_out_shape, ndim * sizeof(int)));

        CHECK_CUDA_EX(cudaMemcpy(d_in_strides, in_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_EX(cudaMemcpy(d_out_strides, out_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_EX(cudaMemcpy(d_perm, perm.data(), ndim * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_EX(cudaMemcpy(d_out_shape, new_shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice));

        int threads_flat = 256;
        int blocks_flat = (total + threads_flat - 1) / threads_flat;
        transpose_kernel<T><<<blocks_flat, threads_flat>>>(data_, new_data, d_in_strides, d_out_strides, d_perm, ndim, total, d_out_shape);

        CHECK_CUDA_EX(cudaFree(d_in_strides));
        CHECK_CUDA_EX(cudaFree(d_out_strides));
        CHECK_CUDA_EX(cudaFree(d_perm));
        CHECK_CUDA_EX(cudaFree(d_out_shape));
    }

    CHECK_CUDA_EX(cudaFree(data_));
    data_ = new_data;
    total_size_ = total;
    shape_ = new_shape;

    return *this;
}

// Explicit template instantiations
template class tensor<float>;
template class tensor<__half>;
template class tensor<__nv_bfloat16>;
template class tensor<int8_t>;
template class tensor<uint8_t>;
template class tensor<int>;

} // namespace dnn

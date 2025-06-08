#include "device.cuh"
#include "../utils/common.cuh"

#include <cuda_runtime.h>

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#include <cublas_v2.h>
#endif

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <iomanip>
#include <thread>

namespace dnn {

// Device: Holds hardware capability info
Device::Device(int device_id) : id_(device_id) {
    utils::CHECK_CUDA_EX(cudaGetDeviceProperties(&props_, id_));
}

int Device::id() const { return id_; }
std::string Device::name() const { return props_.name; }
int Device::compute_capability_major() const { return props_.major; }
int Device::compute_capability_minor() const { return props_.minor; }

std::string Device::architecture_name() const {
    int sm = compute_capability_major() * 10 + compute_capability_minor();
    static const std::unordered_map<int, std::string> arch_map = {
        {30, "Kepler"}, {32, "Kepler"}, {35, "Kepler"}, {37, "Kepler"},
        {50, "Maxwell"},{52, "Maxwell"},{53, "Maxwell"},
        {60, "Pascal"}, {61, "Pascal"}, {62, "Pascal"},
        {70, "Volta"},  {72, "Xavier"},
        {75, "Turing"},
        {80, "Ampere"}, {86, "Ampere"}, {87, "Ampere"},
        {89, "Ada"},    {90, "Ada"},
        {100,"Hopper"}, {101,"Hopper"},{102,"Hopper"},
        {105,"Blackwell"},{110,"Blackwell"}
    };
    auto it = arch_map.find(sm);
    return it != arch_map.end() ? it->second : "Unknown";
}

size_t Device::total_memory_bytes() const {
    utils::CHECK_CUDA_EX(cudaSetDevice(id_));
    size_t free, total;
    utils::CHECK_CUDA_EX(cudaMemGetInfo(&free, &total));
    return total;
}

size_t Device::available_memory_bytes() const {
    utils::CHECK_CUDA_EX(cudaSetDevice(id_));
    size_t free, total;
    utils::CHECK_CUDA_EX(cudaMemGetInfo(&free, &total));
    return free;
}

bool Device::nvlink_supported() const {
    int access = 0;
    cudaDeviceCanAccessPeer(&access, id_, id_);
    return access != 0;
}

int Device::num_sm() const { return props_.multiProcessorCount; }

int Device::cuda_cores_per_sm() const {
    int sm = compute_capability_major() * 10 + compute_capability_minor();
    static const std::unordered_map<int, int> sm_cores = {
        {30,192},{32,192},{35,192},{37,192},
        {50,128},{52,128},{53,128},
        {60, 64},{61,128},{62,128},
        {70, 64},{72, 64},
        {75, 64},
        {80, 64},{86,128},{87,128},
        {89,128},{90,128},
        {100,128},{101,128},{102,128},
        {105,128},{110,128}
    };
    auto it = sm_cores.find(sm);
    return it != sm_cores.end() ? it->second : -1;
}

int Device::total_cuda_cores() const {
    int per_sm = cuda_cores_per_sm();
    return per_sm > 0 ? per_sm * num_sm() : -1;
}

int Device::tensor_cores_per_sm() const {
    return compute_capability_major() >= 7 ? 8 : 0;
}

int Device::total_tensor_cores() const {
    return tensor_cores_per_sm() * num_sm();
}

float Device::clock_ghz() const {
    return props_.clockRate * 1e-6f;
}

void Device::dump_info(std::ostream& os /* = std::cout */) const {
    os << "CUDA Device #" << id_ << ": " << name() << "\n";
    os << "  Architecture: " << architecture_name() << " (SM "
        << compute_capability_major() << "." << compute_capability_minor() << ")\n";
    os << "  VRAM: " << (total_memory_bytes() >> 20) << " MB\n";
    os << "  Available VRAM: " << (available_memory_bytes() >> 20) << " MB\n";
    os << "  SMs: " << num_sm() << "\n";
    int cores = total_cuda_cores();
    os << "  CUDA Cores: " << (cores > 0 ? std::to_string(cores) : "Unknown") << "\n";
    os << "  Tensor Cores: " << total_tensor_cores() << "\n";
    os << "  Clock: " << std::fixed << std::setprecision(2) << clock_ghz() << " GHz\n";
    os << "  NVLink: " << (nvlink_supported() ? "Yes" : "No") << "\n";
}

} // namespace dnn

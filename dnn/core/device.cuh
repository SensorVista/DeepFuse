#pragma once

#include "../utils/common.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <iomanip>

namespace dnn {

class CudaDevice {
public:
    explicit CudaDevice(int device_id = 0) : id_(device_id) {
        utils::CHECK_CUDA_EX(cudaSetDevice(id_));
        utils::CHECK_CUDA_EX(cudaGetDeviceProperties(&props_, id_));
    }

    ~CudaDevice() = default;

    int id() const { return id_; }

    std::string name() const { return props_.name; }

    int compute_capability_major() const { return props_.major; }
    int compute_capability_minor() const { return props_.minor; }

    std::string architecture_name() const {
        const int version = compute_capability_major() * 10 + compute_capability_minor();
        static const std::unordered_map<int, std::string> arch_map = {
            {30, "Kepler"},    {32, "Kepler"},   {35, "Kepler"},   {37, "Kepler"},
            {50, "Maxwell"},   {52, "Maxwell"},  {53, "Maxwell"},
            {60, "Pascal"},    {61, "Pascal"},   {62, "Pascal"},
            {70, "Volta"},
            {72, "Xavier"},
            {75, "Turing"},
            {80, "Ampere"},    {86, "Ampere"},   {87, "Ampere"},
            {89, "Ada"},       {90, "Ada"},
            {100, "Hopper"},   {101, "Hopper"},  {102, "Hopper"},
            {105, "Blackwell"}, {110, "Blackwell"}
        };
        auto it = arch_map.find(version);
        return it != arch_map.end() ? it->second : "Unknown";
    }

    size_t total_memory_bytes() const {
        size_t free_mem, total_mem;
        utils::CHECK_CUDA_EX(cudaMemGetInfo(&free_mem, &total_mem));
        return total_mem;
    }

    bool nvlink_supported() const {
        int access = 0;
        cudaDeviceCanAccessPeer(&access, id_, id_);
        return access != 0;
    }

    int num_sm() const { return props_.multiProcessorCount; }

    int cuda_cores_per_sm() const {
        const int version = compute_capability_major() * 10 + compute_capability_minor();
        static const std::unordered_map<int, int> sm_cores = {
            {30, 192}, {32, 192}, {35, 192}, {37, 192}, // Kepler
            {50, 128}, {52, 128}, {53, 128},             // Maxwell
            {60,  64}, {61, 128}, {62, 128},             // Pascal
            {70,  64},                                   // Volta
            {72,  64},                                   // Xavier
            {75,  64},                                   // Turing
            {80,  64}, {86, 128}, {87, 128},             // Ampere
            {89, 128}, {90, 128},                        // Ada
            {100, 128}, {101, 128}, {102, 128},          // Hopper
            {105, 128}, {110, 128}                       // Blackwell (estimate)
        };
        auto it = sm_cores.find(version);
        return it != sm_cores.end() ? it->second : -1;
    }

    int total_cuda_cores() const {
        int cores_per_sm = cuda_cores_per_sm();
        return (cores_per_sm > 0) ? cores_per_sm * num_sm() : -1;
    }

    int tensor_cores_per_sm() const {
        if (compute_capability_major() >= 7) return 8; // conservative fallback
        return 0;
    }

    int total_tensor_cores() const {
        return tensor_cores_per_sm() * num_sm();
    }

    float clock_ghz() const {
        return props_.clockRate * 1e-6f;
    }

    void dump_info(std::ostream& os = std::cout) const {
        os << "CUDA Device #" << id_ << ": " << name() << "\n";
        os << "  Architecture: " << architecture_name() << " (SM " 
           << compute_capability_major() << "." << compute_capability_minor() << ")\n";
        os << "  VRAM: " << (total_memory_bytes() / (1024 * 1024)) << " MB\n";
        os << "  SMs: " << num_sm() << "\n";
        int c_cores = total_cuda_cores();
        os << "  CUDA Cores: " << (c_cores > 0 ? std::to_string(c_cores) : "Unknown") << "\n";
        os << "  Tensor Cores: " << total_tensor_cores() << "\n";
        os << "  Clock: " << std::fixed << std::setprecision(2) << clock_ghz() << " GHz\n";
        os << "  NVLink: " << (nvlink_supported() ? "Yes" : "No") << "\n";
    }

private:
    int id_;
    cudaDeviceProp props_;
};

} // namespace dnn

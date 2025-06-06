#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <string>

namespace dnn {

// Device: Holds hardware capability info
class Device {
public:
    Device(int device_id = 0);

    int id() const;
    std::string name() const;
    int compute_capability_major() const;
    int compute_capability_minor() const;
    std::string architecture_name() const;
    size_t total_memory_bytes() const;
    size_t available_memory_bytes() const;
    bool nvlink_supported() const;

    int num_sm() const; 
    int cuda_cores_per_sm() const;
    int total_cuda_cores() const;
    int tensor_cores_per_sm() const;
    int total_tensor_cores() const;
    float clock_ghz() const;

    void dump_info(std::ostream& os = std::cout) const;

private:
    int id_;
    cudaDeviceProp props_;
};

} // namespace dnn

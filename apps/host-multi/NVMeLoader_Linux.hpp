#ifndef NVMELOADER_LINUX_HPP
#define NVMELOADER_LINUX_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include <fstream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/stat.h>
#include <cstring>
#include <memory>

constexpr size_t SECTOR_ALIGN = 4096;

inline void* aligned_cuda_pinned_alloc(size_t size) {
    void* raw = nullptr;
    cudaHostAlloc(&raw, size + SECTOR_ALIGN, cudaHostAllocDefault);
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (addr + SECTOR_ALIGN - 1) & ~(SECTOR_ALIGN - 1);
    return reinterpret_cast<void*>(aligned);
}

struct Slot {
    void* host_ptr = nullptr;
    void* device_ptr = nullptr;
    cudaStream_t stream = nullptr;
    size_t size = 0;
    bool pending = false;
    int fd = -1;
    off_t offset = 0;
};

class NVMeLoader {
public:
    NVMeLoader(const std::string& file_path, size_t block_size, int slot_count)
        : block_size_(block_size), slot_count_(slot_count), offset_(0), next_slot_(0)
    {
        assert(block_size % SECTOR_ALIGN == 0);
        assert(slot_count >= 2);

        // Open file with O_DIRECT for direct I/O (bypasses page cache)
        file_fd_ = open(file_path.c_str(), O_RDONLY | O_DIRECT);
        if (file_fd_ == -1) {
            // Fallback to regular file I/O if O_DIRECT fails
            file_fd_ = open(file_path.c_str(), O_RDONLY);
            if (file_fd_ == -1) {
                throw std::runtime_error("Failed to open file: " + file_path + " - " + strerror(errno));
            }
            std::cout << "Warning: Using regular file I/O (O_DIRECT not supported)" << std::endl;
        }

        // Create epoll instance for async I/O
        epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
        if (epoll_fd_ == -1) {
            close(file_fd_);
            throw std::runtime_error("Failed to create epoll instance: " + std::string(strerror(errno)));
        }

        // Initialize slots
        slots_.resize(slot_count_);
        for (int i = 0; i < slot_count_; ++i) {
            slots_[i].host_ptr = aligned_cuda_pinned_alloc(block_size_);
            cudaMalloc(&slots_[i].device_ptr, block_size_);
            cudaStreamCreate(&slots_[i].stream);
            slots_[i].fd = file_fd_;
        }
    }

    ~NVMeLoader() {
        for (auto& s : slots_) {
            cudaStreamDestroy(s.stream);
            cudaFree(s.device_ptr);
            cudaFreeHost(s.host_ptr);
        }
        if (epoll_fd_ != -1) close(epoll_fd_);
        if (file_fd_ != -1) close(file_fd_);
    }

    void preload_all() {
        for (int i = 0; i < slot_count_; ++i) {
            issue_read(i);
        }
    }

    bool load_next(void** device_ptr_out, size_t* size_out, cudaStream_t* stream_out) {
        struct epoll_event events[slot_count_];
        int num_events = epoll_wait(epoll_fd_, events, slot_count_, -1);
        
        if (num_events == -1) {
            std::cerr << "epoll_wait error: " << strerror(errno) << std::endl;
            return false;
        }

        // Find the slot that completed
        int slot_idx = -1;
        for (int i = 0; i < num_events; ++i) {
            slot_idx = find_slot_by_fd(events[i].data.fd);
            if (slot_idx >= 0) break;
        }

        if (slot_idx == -1) {
            std::cerr << "Could not find slot for completed I/O" << std::endl;
            return false;
        }

        Slot& slot = slots_[slot_idx];
        
        // Get the actual bytes read
        ssize_t bytes_read = read(slot.fd, slot.host_ptr, block_size_);
        if (bytes_read == -1) {
            std::cerr << "read error: " << strerror(errno) << std::endl;
            return false;
        }

        slot.size = static_cast<size_t>(bytes_read);
        slot.pending = false;

        // Copy to GPU asynchronously
        cudaMemcpyAsync(slot.device_ptr, slot.host_ptr, slot.size, 
                       cudaMemcpyHostToDevice, slot.stream);

        *device_ptr_out = slot.device_ptr;
        *size_out = slot.size;
        *stream_out = slot.stream;

        // Reissue read to keep pipeline full
        issue_read(slot_idx);
        return true;
    }

private:
    int file_fd_ = -1;
    int epoll_fd_ = -1;
    std::vector<Slot> slots_;
    size_t block_size_;
    int slot_count_;
    off_t offset_;
    int next_slot_;

    void issue_read(int i) {
        Slot& s = slots_[i];
        
        // Set up epoll event for this slot
        struct epoll_event ev;
        ev.events = EPOLLIN | EPOLLET; // Edge-triggered
        ev.data.fd = s.fd;
        
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, s.fd, &ev) == -1) {
            if (errno != EEXIST) { // Ignore if already added
                throw std::runtime_error("epoll_ctl failed: " + std::string(strerror(errno)));
            }
        }

        // Issue async read using pread for thread safety
        ssize_t bytes_read = pread(s.fd, s.host_ptr, block_size_, offset_);
        if (bytes_read == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // This is expected for async I/O
                s.pending = true;
            } else {
                throw std::runtime_error("pread failed: " + std::string(strerror(errno)));
            }
        } else {
            // Immediate completion
            s.size = static_cast<size_t>(bytes_read);
            s.pending = false;
        }

        offset_ += block_size_;
    }

    int find_slot_by_fd(int fd) {
        for (int i = 0; i < slot_count_; ++i) {
            if (slots_[i].fd == fd) return i;
        }
        return -1;
    }
};

//////////////////////////
// INLINE USAGE EXAMPLE //
//////////////////////////

void example_nvme_loader() {
    const std::string filepath = "/tmp/nvme_test.bin";
    const size_t block_size = 128 * 1024;
    const int slot_count = 4;
    const int max_blocks = 100;

    NVMeLoader loader(filepath, block_size, slot_count);
    loader.preload_all();

    for (int i = 0; i < max_blocks; ++i) {
        void* dev_ptr = nullptr;
        size_t size = 0;
        cudaStream_t stream = nullptr;

        if (!loader.load_next(&dev_ptr, &size, &stream))
            break;

        // Simulate compute: dummy kernel or memcpy
        cudaStreamSynchronize(stream);
        std::cout << "Block " << i << " loaded to GPU: " << size << " bytes\n";
    }
}

#endif // NVMELOADER_LINUX_HPP

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include <fstream>

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
    OVERLAPPED ov = {};
    size_t size = 0;
    bool pending = false;
};

class NVMeLoader {
public:
    NVMeLoader(const std::string& file_path, size_t block_size, int slot_count)
        : block_size_(block_size), slot_count_(slot_count), offset_(0), next_slot_(0)
    {
        assert(block_size % SECTOR_ALIGN == 0);
        assert(slot_count >= 2);

        file_ = CreateFileA(
            file_path.c_str(), GENERIC_READ,
            FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr
        );
        if (file_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Failed to open NVMe file");

        iocp_ = CreateIoCompletionPort(file_, nullptr, 0, 0);
        if (!iocp_) throw std::runtime_error("CreateIoCompletionPort failed");

        slots_.resize(slot_count_);
        for (int i = 0; i < slot_count_; ++i) {
            slots_[i].host_ptr = aligned_cuda_pinned_alloc(block_size_);
            cudaMalloc(&slots_[i].device_ptr, block_size_);
            cudaStreamCreate(&slots_[i].stream);
        }
    }

    ~NVMeLoader() {
        for (auto& s : slots_) {
            cudaStreamDestroy(s.stream);
            cudaFree(s.device_ptr);
            cudaFreeHost(s.host_ptr);
        }
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        if (iocp_) CloseHandle(iocp_);
    }

    void preload_all() {
        for (int i = 0; i < slot_count_; ++i)
            issue_read(i);
    }

    bool load_next(void** device_ptr_out, size_t* size_out, cudaStream_t* stream_out) {
        DWORD bytes_read = 0;
        ULONG_PTR key = 0;
        LPOVERLAPPED ov = nullptr;

        if (!GetQueuedCompletionStatus(iocp_, &bytes_read, &key, &ov, INFINITE)) {
            std::cerr << "IOCP error\n";
            return false;
        }

        int slot_idx = find_slot_by_ov(ov);
        assert(slot_idx >= 0);

        Slot& slot = slots_[slot_idx];
        slot.size = bytes_read;
        slot.pending = false;

        cudaMemcpyAsync(slot.device_ptr, slot.host_ptr, bytes_read, cudaMemcpyHostToDevice, slot.stream);

        *device_ptr_out = slot.device_ptr;
        *size_out = bytes_read;
        *stream_out = slot.stream;

        issue_read(slot_idx); // reissue to keep pipeline full
        return true;
    }

private:
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE iocp_ = nullptr;
    std::vector<Slot> slots_;
    size_t block_size_;
    int slot_count_;
    size_t offset_;
    int next_slot_;

    void issue_read(int i) {
        Slot& s = slots_[i];
        ZeroMemory(&s.ov, sizeof(OVERLAPPED));
        LARGE_INTEGER li;
        li.QuadPart = offset_;
        s.ov.Offset = li.LowPart;
        s.ov.OffsetHigh = li.HighPart;

        BOOL ok = ReadFile(file_, s.host_ptr, (DWORD)block_size_, nullptr, &s.ov);
        if (!ok && GetLastError() != ERROR_IO_PENDING)
            throw std::runtime_error("ReadFile failed");

        s.pending = true;
        offset_ += block_size_;
    }

    int find_slot_by_ov(LPOVERLAPPED ov) {
        for (int i = 0; i < slot_count_; ++i)
            if (&slots_[i].ov == ov) return i;
        return -1;
    }
};

//////////////////////////
// INLINE USAGE EXAMPLE //
//////////////////////////

void example_nvme_loader() {
    const std::string filepath = "D:\\nvme_test.bin";
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

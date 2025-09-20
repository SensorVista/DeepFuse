// Platform-specific includes
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <sys/mman.h>
    #include <aio.h>
    #include <errno.h>
    #include <poll.h>
    #include <cstring>  // for memset and strerror
#endif

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include <fstream>
#include <memory>

constexpr size_t SECTOR_ALIGN = 4096;

inline void* aligned_cuda_pinned_alloc(size_t size) {
    void* raw = nullptr;
    cudaHostAlloc(&raw, size + SECTOR_ALIGN, cudaHostAllocDefault);
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (addr + SECTOR_ALIGN - 1) & ~(SECTOR_ALIGN - 1);
    return reinterpret_cast<void*>(aligned);
}

// Platform-specific I/O structures
#ifdef _WIN32
struct Slot {
    void* host_ptr = nullptr;
    void* device_ptr = nullptr;
    cudaStream_t stream = nullptr;
    OVERLAPPED ov = {};
    size_t size = 0;
    bool pending = false;
};
#else
struct Slot {
    void* host_ptr = nullptr;
    void* device_ptr = nullptr;
    cudaStream_t stream = nullptr;
    struct aiocb aio_cb = {};
    size_t size = 0;
    bool pending = false;
};
#endif

class NVMeLoader {
public:
    NVMeLoader(const std::string& file_path, size_t block_size, int slot_count)
        : block_size_(block_size), slot_count_(slot_count), offset_(0), next_slot_(0)
    {
        assert(block_size % SECTOR_ALIGN == 0);
        assert(slot_count >= 2);

#ifdef _WIN32
        file_ = CreateFileA(
            file_path.c_str(), GENERIC_READ,
            FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr
        );
        if (file_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Failed to open NVMe file");

        iocp_ = CreateIoCompletionPort(file_, nullptr, 0, 0);
        if (!iocp_) throw std::runtime_error("CreateIoCompletionPort failed");
#else
        // Linux: Open file with O_DIRECT for direct I/O (bypasses page cache)
        file_ = open(file_path.c_str(), O_RDONLY | O_DIRECT);
        if (file_ == -1)
            throw std::runtime_error("Failed to open NVMe file: " + std::string(strerror(errno)));
#endif

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
#ifdef _WIN32
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        if (iocp_) CloseHandle(iocp_);
#else
        if (file_ != -1) close(file_);
#endif
    }

    void preload_all() {
        for (int i = 0; i < slot_count_; ++i)
            issue_read(i);
    }

    bool load_next(void** device_ptr_out, size_t* size_out, cudaStream_t* stream_out) {
#ifdef _WIN32
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
#else
        // Linux: Use aio_suspend to wait for async I/O completion
        struct aiocb* aio_list[slot_count_];
        int aio_count = 0;
        
        // Build list of pending aio operations
        for (int i = 0; i < slot_count_; ++i) {
            if (slots_[i].pending) {
                aio_list[aio_count++] = &slots_[i].aio_cb;
            }
        }
        
        if (aio_count == 0) return false;
        
        // Wait for any aio operation to complete
        int result = aio_suspend(aio_list, aio_count, nullptr);
        if (result == -1) {
            std::cerr << "aio_suspend error: " << strerror(errno) << std::endl;
            return false;
        }
        
        // Find which operation completed
        int slot_idx = -1;
        for (int i = 0; i < slot_count_; ++i) {
            if (slots_[i].pending && aio_error(&slots_[i].aio_cb) == 0) {
                slot_idx = i;
                break;
            }
        }
        
        if (slot_idx == -1) return false;
        
        Slot& slot = slots_[slot_idx];
        slot.size = aio_return(&slot.aio_cb);
        slot.pending = false;
#endif

        cudaMemcpyAsync(slot.device_ptr, slot.host_ptr, slot.size, cudaMemcpyHostToDevice, slot.stream);

        *device_ptr_out = slot.device_ptr;
        *size_out = slot.size;
        *stream_out = slot.stream;

        issue_read(slot_idx); // reissue to keep pipeline full
        return true;
    }

private:
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE iocp_ = nullptr;
#else
    int file_ = -1;
#endif
    std::vector<Slot> slots_;
    size_t block_size_;
    int slot_count_;
    size_t offset_;
    int next_slot_;

    void issue_read(int i) {
        Slot& s = slots_[i];
#ifdef _WIN32
        ZeroMemory(&s.ov, sizeof(OVERLAPPED));
        LARGE_INTEGER li;
        li.QuadPart = offset_;
        s.ov.Offset = li.LowPart;
        s.ov.OffsetHigh = li.HighPart;

        BOOL ok = ReadFile(file_, s.host_ptr, (DWORD)block_size_, nullptr, &s.ov);
        if (!ok && GetLastError() != ERROR_IO_PENDING)
            throw std::runtime_error("ReadFile failed");

        s.pending = true;
#else
        // Linux: Setup aio control block
        memset(&s.aio_cb, 0, sizeof(struct aiocb));
        s.aio_cb.aio_fildes = file_;
        s.aio_cb.aio_buf = s.host_ptr;
        s.aio_cb.aio_nbytes = block_size_;
        s.aio_cb.aio_offset = offset_;
        
        int result = aio_read(&s.aio_cb);
        if (result == -1)
            throw std::runtime_error("aio_read failed: " + std::string(strerror(errno)));
        
        s.pending = true;
#endif
        offset_ += block_size_;
    }

#ifdef _WIN32
    int find_slot_by_ov(LPOVERLAPPED ov) {
        for (int i = 0; i < slot_count_; ++i)
            if (&slots_[i].ov == ov) return i;
        return -1;
    }
#endif
};

//////////////////////////
// INLINE USAGE EXAMPLE //
//////////////////////////

void example_nvme_loader() {
#ifdef _WIN32
    const std::string filepath = "D:\\nvme_test.bin";
#else
    const std::string filepath = "/tmp/nvme_test.bin";
#endif
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

#pragma once

#include <gpu/loon_gpu.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>


namespace loon {

class RingBuffer {
   public:
    RingBuffer() = default;
    RingBuffer(gpu::Device device, uint32_t size, uint32_t num_frames_in_flight);
    ~RingBuffer();
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer(RingBuffer&& other);
    RingBuffer& operator=(const RingBuffer&) = delete;
    RingBuffer& operator=(RingBuffer&&);

    template <class T>
    gpu::GpuPtr append(uint64_t frame_idx, const T&);

    template <class T>
    gpu::GpuPtr append(uint64_t frame_idx, const std::vector<T>&);

    gpu::GpuPtr append_raw(uint64_t    frame_idx,
                           const void* ptr,
                           size_t      size,
                           size_t      alignment = alignof(max_align_t));

   private:
    uint32_t contiguous_free_space() const;

    struct AllocationRange {
        uint64_t frame_idx = 0;
        uint32_t start     = 0;
        uint32_t end       = 0;
    };

    static constexpr size_t kMaxFramesInFlight   = 8;  // Must be power of 2.
    static constexpr size_t kAllocatedRangesMask = kMaxFramesInFlight - 1;

    // We treat the allocated ranges like a small ring buffer.
    AllocationRange m_allocated_ranges[kMaxFramesInFlight]{0};
    uint32_t        m_allocated_range_tail = 0;
    uint32_t        m_allocated_range_head = 0;

    gpu::Device m_device     = nullptr;
    gpu::GpuPtr m_device_ptr = 0;
    void*       m_host_ptr;
    uint32_t    m_mask;
    uint32_t    m_num_frames_in_flight;
};

template <class T>
gpu::GpuPtr RingBuffer::append(uint64_t frame_idx, const T& t) {
    return append_raw(frame_idx, &t, sizeof(T), alignof(T));
}

template <class T>
gpu::GpuPtr RingBuffer::append(uint64_t frame_idx, const std::vector<T>& vec) {
    return append_raw(frame_idx, vec.data(), vec.size() * sizeof(T), alignof(T));
}

}  // namespace loon
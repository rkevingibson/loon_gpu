#pragma once

#include <gpu/loon_gpu.h>

#include <cassert>
#include <cstdint>
#include <cstring>



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

    gpu::Device              m_device = nullptr;
    gpu::Handle<gpu::Buffer> m_buffer;
    void*                    m_host_ptr;
    gpu::GpuPtr              m_device_ptr;
    uint32_t                 m_mask;
    uint32_t                 m_num_frames_in_flight;
};

template <class T>
gpu::GpuPtr RingBuffer::append(uint64_t frame_idx, const T& t) {
    const uint32_t completed_frame_idx
        = frame_idx > m_num_frames_in_flight ? (frame_idx - m_num_frames_in_flight) : 0;
    while (m_allocated_ranges[m_allocated_range_tail & kAllocatedRangesMask].frame_idx
           < completed_frame_idx) {
        m_allocated_range_tail++;  // Effectively "free" this range.
    }

    // For our examples, we're not really handling errors/oom - just assert we have enough space.
    assert(contiguous_free_space() >= sizeof(T));

    AllocationRange* range = nullptr;
    if (m_allocated_ranges[m_allocated_range_head & kAllocatedRangesMask].frame_idx == frame_idx) {
        // Extending the current frame's range.
        range = &m_allocated_ranges[m_allocated_range_head & kAllocatedRangesMask];
    } else {
        // Allocate a new range, initializing it to the end of the previous range.
        const auto& prev_range = m_allocated_ranges[m_allocated_range_head & kAllocatedRangesMask];
        m_allocated_range_head++;
        assert((m_allocated_range_head & kAllocatedRangesMask)
               != (m_allocated_range_tail & kAllocatedRangesMask));

        range  = &m_allocated_ranges[m_allocated_range_head & kAllocatedRangesMask];
        *range = {
            .frame_idx = frame_idx,
            .start     = prev_range.end,
            .end       = prev_range.end,
        };
    }

    const auto align = [](uint32_t offset, uint32_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    };

    // TODO: if we'd wrap around the ring buffer, need to align to multiple of m_size instead.
    uint32_t offset_start = align(range->end, alignof(T));
    uint32_t offset_end   = offset_start + sizeof(T);
    if ((offset_start & m_mask) > (offset_end & m_mask)) {
        offset_start = align(offset_start, m_mask + 1);
        offset_end   = offset_start + sizeof(T);
    }
    range->end      = offset_end;
    void* host_dest = (char*)m_host_ptr + (offset_start & m_mask);
    memcpy(host_dest, &t, sizeof(T));

    return m_device_ptr + (offset_start & m_mask);
}


}  // namespace loon
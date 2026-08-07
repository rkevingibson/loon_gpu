#include "gpu_args.h"


namespace loon {

RingBuffer::RingBuffer(gpu::Device device, uint32_t size, uint32_t num_frames_in_flight) :
    m_device(device), m_mask{size - 1}, m_num_frames_in_flight{num_frames_in_flight} {
    m_device_ptr = gpu::malloc(device, size);
    m_host_ptr   = gpu::get_host_pointer(device, m_device_ptr);
}

RingBuffer::~RingBuffer() {
    if (m_device_ptr) { gpu::free(m_device, m_device_ptr); }
}

RingBuffer::RingBuffer(RingBuffer&& other) {
    memcpy(m_allocated_ranges, other.m_allocated_ranges, sizeof(m_allocated_ranges));
    m_allocated_range_tail = other.m_allocated_range_tail;
    m_allocated_range_head = other.m_allocated_range_head;

    m_device               = std::move(other.m_device);
    m_device_ptr           = std::exchange(other.m_device_ptr, 0);
    m_host_ptr             = other.m_host_ptr;
    m_mask                 = other.m_mask;
    m_num_frames_in_flight = other.m_num_frames_in_flight;
}

RingBuffer& RingBuffer::operator=(RingBuffer&& other) {
    memcpy(m_allocated_ranges, other.m_allocated_ranges, sizeof(m_allocated_ranges));
    m_allocated_range_tail = other.m_allocated_range_tail;
    m_allocated_range_head = other.m_allocated_range_head;

    m_device               = std::exchange(other.m_device, m_device);
    m_device_ptr           = std::exchange(other.m_device_ptr, m_device_ptr);
    m_host_ptr             = other.m_host_ptr;
    m_mask                 = other.m_mask;
    m_num_frames_in_flight = other.m_num_frames_in_flight;
    return *this;
}

uint32_t RingBuffer::contiguous_free_space() const {
    const auto start = m_allocated_ranges[m_allocated_range_tail & kAllocatedRangesMask].start;
    const auto end   = m_allocated_ranges[m_allocated_range_head & kAllocatedRangesMask].end;

    const uint32_t size = m_mask + 1;
    if ((start & m_mask) > (end & m_mask)) {
        // Allocated space wraps around the ring buffer, so need to consider that
        const uint32_t free_size = (start & m_mask) - (end & m_mask);
        return free_size;
    } else {
        // Allocated space doesn't wrap, so free space wraps around.
        const size_t tail_size = size - (end & m_mask);
        const size_t head_size = start & m_mask;
        return tail_size > head_size ? tail_size : head_size;
    }
}

gpu::GpuPtr RingBuffer::append_raw(uint64_t    frame_idx,
                                   const void* ptr,
                                   size_t      size,
                                   size_t      alignment) {
    while (frame_idx - m_allocated_ranges[m_allocated_range_tail & kAllocatedRangesMask].frame_idx >
           m_num_frames_in_flight) {
        m_allocated_range_tail++;  // Effectively "free" this range.
    }

    // If we don't have enough space to do the copy, return nullptr.
    if (contiguous_free_space() < size) { return 0; }

    AllocationRange* range = nullptr;
    if (m_allocated_ranges[m_allocated_range_head & kAllocatedRangesMask].frame_idx == frame_idx) {
        // Extending the current frame's range.
        range = &m_allocated_ranges[m_allocated_range_head & kAllocatedRangesMask];
    } else {
        // Allocate a new range, initializing it to the end of the previous range.
        const auto& prev_range = m_allocated_ranges[m_allocated_range_head & kAllocatedRangesMask];
        m_allocated_range_head++;
        assert((m_allocated_range_head & kAllocatedRangesMask) !=
               (m_allocated_range_tail & kAllocatedRangesMask));

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

    uint32_t offset_start = align(range->end, alignment);
    uint32_t offset_end   = offset_start + size;
    if ((offset_start & m_mask) > (offset_end & m_mask)) {
        // Adjust start in the case where we're wrapping around the ring buffer.
        offset_start = align(offset_start, m_mask + 1);
        offset_end   = offset_start + size;
    }
    range->end      = offset_end;
    void* host_dest = (char*)m_host_ptr + (offset_start & m_mask);
    memcpy(host_dest, ptr, size);

    return m_device_ptr + (offset_start & m_mask);
}

}  // namespace loon
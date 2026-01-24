#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>

#include "gpu/loon_gpu.h"
#include "platform_utils.h"


#ifdef _MSVC_LANG
#    define NO_UNIQUE_ADDR [[msvc::no_unique_address]]
#else
#    define NO_UNIQUE_ADDR [[no_unique_address]]
#endif

namespace loon::gpu {

// MARK: Allocator
class Allocator {
   public:
    Allocator();
    Allocator(ProcAllocatorCallback alloc, void* userdata) :
        m_alloc{alloc}, m_userdata{userdata} {};

    friend void swap(Allocator& a, Allocator& b) {
        std::swap(a.m_alloc, b.m_alloc);
        std::swap(a.m_userdata, b.m_userdata);
    }

    constexpr MemoryBlock realloc(MemoryBlock blk, size_t new_size) const {
        return m_alloc(m_userdata, blk.ptr, blk.len, new_size);
    }
    constexpr MemoryBlock alloc(size_t size) const { return m_alloc(m_userdata, nullptr, 0, size); }
    constexpr void        free(MemoryBlock blk) const { m_alloc(m_userdata, blk.ptr, blk.len, 0); }

   private:
    ProcAllocatorCallback m_alloc    = nullptr;
    void*                 m_userdata = nullptr;
};

// MARK: Vector
template <class T>
class Vector {
   public:
    Vector() = default;
    Vector(Allocator allocator, uint32_t initial_capacity = 0);
    Vector(Allocator allocator, const T* buff, uint32_t count);
    Vector(Allocator allocator, const T& value, uint32_t count);
    Vector(Vector&& rhs) :
        m_allocator(rhs.m_allocator),
        m_count{std::exchange(rhs.m_count, 0)},
        m_capacity{std::exchange(rhs.m_capacity, 0)},
        m_data{std::exchange(rhs.m_data, nullptr)} {};
    ~Vector() { clear(); }
    Vector& operator=(Vector&& rhs) {
        swap(*this, rhs);
        return *this;
    }

    operator Span<T>() { return Span(m_data, m_count); }
    operator Span<const T>() const { return Span(m_data, m_count); }

    friend void swap(Vector& a, Vector& b) {
        using std::swap;
        swap(a.m_allocator, b.m_allocator);
        swap(a.m_count, b.m_count);
        swap(a.m_capacity, b.m_capacity);
        swap(a.m_data, b.m_data);
    }

    void clear();
    void reserve(uint32_t capacity);

    T& push_back(const T& v) {
        if (m_count == m_capacity) { reserve(grow_capacity(m_count + 1)); }
        T* result = std::construct_at(m_data + m_count, v);
        m_count++;
        return *result;
    }

    template <class... Args>
    T& emplace_back(Args&&... args) {
        if (m_count == m_capacity) { reserve(grow_capacity(m_count + 1)); }
        T* result = std::construct_at(m_data + m_count, std::forward<Args>(args)...);
        m_count++;
        return *result;
    }

    void pop_back() {
        m_count--;
        std::destroy_at(m_data + m_count);
    }

    void erase(T* first, T* last) {
        const auto old_end = end();
        m_count -= std::distance(first, last);
        // Move from [last, old_end) to [first, new_end)
        for (; last != old_end; ++first, ++last) *first = std::move(*last);
        std::destroy(end(), old_end);
    }

    const T&           operator[](uint32_t idx) const { return m_data[idx]; }
    T&                 operator[](uint32_t idx) { return m_data[idx]; }
    T*                 data() { return m_data; }
    const T*           data() const { return m_data; }
    constexpr T*       begin() { return m_data; }
    constexpr const T* begin() const { return m_data; }
    constexpr T*       end() { return m_data + m_count; }
    constexpr const T* end() const { return m_data + m_count; }
    constexpr uint32_t size() const { return m_count; }
    constexpr bool     is_empty() const { return m_count == 0; }

   private:
    void             grow(size_t new_capacity);
    constexpr size_t grow_capacity(size_t sz) const {
        const size_t new_capacity = m_capacity ? (m_capacity + m_capacity / 2) : 8;
        return new_capacity > sz ? new_capacity : sz;
    }
    Allocator m_allocator;
    uint32_t  m_count    = 0;
    uint32_t  m_capacity = 0;
    T*        m_data     = nullptr;
};


// MARK: ObjectPool

template <typename T, size_t N>
class ObjectPool {
   public:
    ObjectPool() { reset(); }
    ~ObjectPool() = default;

    void reset() noexcept {
        m_head = N - 1;
        for (uint32_t i = N - 1; i > 0; --i) {
            // TODO: Proper destruction needed.
            // For non-trivial objects, may need to destruct here, but would need to know which ones
            // aren't in the list first.
            m_freelist[i].next = i - 1;
        }
        m_freelist[0].next = INVALID_INDEX;
    }

    void release(uint32_t idx) noexcept {
        if (idx >= N) return;
        std::unique_lock<std::mutex> lock(m_mutex);
        m_freelist[idx].next = m_head;
        m_head               = idx;
    }

    static constexpr uint32_t INVALID_INDEX = ~0;
    uint32_t                  get() noexcept {
        std::unique_lock<std::mutex> lock(m_mutex);
        const uint32_t               idx = m_head;
        if (idx == INVALID_INDEX) { return idx; }

        m_head = m_freelist[idx].next;
        return idx;
    }

    T&       operator[](uint32_t idx) { return m_freelist[idx].data; }
    const T& operator[](uint32_t idx) const { return m_freelist[idx].data; }

    static constexpr size_t size() { return N; }

   private:
    std::mutex m_mutex;
    struct Node {
        T        data;
        uint32_t next;
    };
    Node     m_freelist[N];
    uint32_t m_head;
};

// MARK: Segment Array

// As described https://danielchasehooper.com/posts/segment_array
// This is effectively a Vector but with pointer stability on reallocations, at the cost of slightly
// slower iteration.

template <class T>
class SegmentArray {
   public:
    SegmentArray() = default;
    SegmentArray(Allocator allocator) : m_allocator(allocator) {}
    SegmentArray(SegmentArray&& rhs) : SegmentArray() { swap(*this, rhs); }
    SegmentArray& operator=(SegmentArray&& rhs) {
        swap(*this, rhs);
        return *this;
    }
    ~SegmentArray() { clear(); }

    friend void swap(SegmentArray& a, SegmentArray& b) {
        std::swap(a.m_allocator, b.m_allocator);
        std::swap(a.m_count, b.m_count);
        std::swap(a.m_used_segments, b.m_used_segments);
        std::swap(a.m_segments, b.m_segments);
    }

    T* get(uint32_t index) {
        const uint64_t segment = int_log_2((index >> kSmallSegmentsToSkip) + 1);
        uint32_t       slot    = index - capacity_for_segment_count(segment);
        return &m_segments[segment][slot];
    }

    const T* get(uint32_t index) const {
        const uint64_t segment = int_log_2((index >> kSmallSegmentsToSkip) + 1);
        uint32_t       slot    = index - capacity_for_segment_count(segment);
        return &m_segments[segment][slot];
    }

    T* push() {
        if (m_count >= capacity_for_segment_count(m_used_segments)) {
            const size_t segment_size     = sizeof(T) * slots_in_segment(m_used_segments);
            const auto   blk              = m_allocator.alloc(segment_size);
            m_segments[m_used_segments++] = reinterpret_cast<T*>(blk.ptr);
        }

        T* result = std::construct_at(get(m_count));
        m_count++;
        return result;
    }

    template <class... Args>
    T& emplace_back(Args&&... args) {
        if (m_count >= capacity_for_segment_count(m_used_segments)) {
            const size_t segment_size     = sizeof(T) * slots_in_segment(m_used_segments);
            const auto   blk              = m_allocator.alloc(segment_size);
            m_segments[m_used_segments++] = reinterpret_cast<T*>(blk.ptr);
        }

        T* result = std::construct_at(get(m_count), T(std::forward<Args>(args)...));
        m_count++;
        return *result;
    }

    void pop() {
        if (m_count > 0) {
            std::destroy_at(get(m_count - 1));
            m_count--;
        }
    }

    uint32_t size() const { return m_count; }

    void clear() {
        for (uint32_t s_idx = 0; s_idx < m_used_segments; ++s_idx) {
            const uint32_t segment_size = slots_in_segment(s_idx);
            T*             segment      = m_segments[s_idx];
            // Segments may not be full - and may be empty if pops have happened.
            const uint32_t freed_count = capacity_for_segment_count(m_used_segments - 1);
            if (m_count > freed_count) {
                const uint32_t element_count = std::min(m_count - freed_count, segment_size);
                std::destroy_n(segment, element_count);
            }

            m_allocator.free({
                .ptr = segment,
                .len = static_cast<uint32_t>(slots_in_segment(s_idx) * sizeof(T)),
            });
            m_segments[s_idx] = nullptr;
        }
        m_count         = 0;
        m_used_segments = 0;
    }

   private:
    static constexpr uint32_t kSmallSegmentsToSkip = 6;
    static constexpr uint32_t slots_in_segment(uint32_t segment_index) {
        return (1 << kSmallSegmentsToSkip) << segment_index;
    }
    static constexpr uint32_t capacity_for_segment_count(uint32_t segment_count) {
        return ((1 << kSmallSegmentsToSkip) << segment_count) - (1 << kSmallSegmentsToSkip);
    }


    Allocator m_allocator;
    uint32_t  m_count         = 0;
    uint32_t  m_used_segments = 0;
    // Smallest segment is 64 items, so 26 segments gets us to ~4 billion items.
    T* m_segments[26] = {nullptr};
};


class TwoLevelBitset {
   public:
    TwoLevelBitset() = default;
    explicit TwoLevelBitset(Allocator alloc, uint32_t size);

    TwoLevelBitset(const TwoLevelBitset&)            = delete;
    TwoLevelBitset& operator=(const TwoLevelBitset&) = delete;
    TwoLevelBitset(TwoLevelBitset&&)                 = default;
    TwoLevelBitset& operator=(TwoLevelBitset&&)      = default;

    // Set the lowest 0 bit and return its index.
    uint32_t set_leading_zero();

    void clear_bit(uint32_t idx);

   private:
    Vector<uint64_t> m_data;
    uint32_t         m_size = 0;
};

// MARK: Implementations:

// MARK: Vector

template <class T>
Vector<T>::Vector(Allocator allocator, uint32_t initial_capacity) : m_allocator(allocator) {
    reserve(initial_capacity);
};

template <class T>
Vector<T>::Vector(Allocator allocator, const T* buff, uint32_t count) : m_allocator{allocator} {
    reserve(count);
    std::uninitialized_copy_n(buff, count, m_data);
}

template <class T>
Vector<T>::Vector(Allocator allocator, const T& value, uint32_t count) : Vector(allocator, count) {
    m_count = count;
    std::uninitialized_fill_n(m_data, count, value);
}

template <class T>
void Vector<T>::clear() {
    if (m_data) {
        std::destroy_n(m_data, m_count);
        m_allocator.free({.ptr = static_cast<void*>(m_data),
                          .len = static_cast<uint32_t>(m_capacity * sizeof(T))});
    }
    m_data     = nullptr;
    m_count    = 0;
    m_capacity = 0;
}

template <class T>
void Vector<T>::reserve(uint32_t capacity) {
    if (capacity > m_capacity) { grow(capacity); }
}

template <class T>
void Vector<T>::grow(size_t new_capacity) {
    new_capacity                  = new_capacity > 4 ? new_capacity : 4;
    const MemoryBlock current_blk = {
        .ptr = static_cast<void*>(m_data),
        .len = static_cast<uint32_t>(m_capacity * sizeof(T)),
    };
    if constexpr (std::is_trivially_copyable_v<T>) {
        MemoryBlock blk = m_allocator.realloc(current_blk, new_capacity * sizeof(T));
        if (blk.ptr == nullptr) { return; }
        new_capacity = blk.len / sizeof(T);
        m_capacity   = new_capacity;
        m_data       = reinterpret_cast<T*>(blk.ptr);
    } else {
        MemoryBlock blk = m_allocator.alloc(new_capacity * sizeof(T));
        if (blk.ptr == nullptr) { return; }
        std::uninitialized_move_n(m_data, m_count, reinterpret_cast<T*>(blk.ptr));
        std::destroy_n(m_data, m_count);
        new_capacity = blk.len / sizeof(T);
        m_allocator.free(current_blk);
        m_capacity = new_capacity;
        m_data     = reinterpret_cast<T*>(blk.ptr);
    }
}

}  // namespace loon::gpu
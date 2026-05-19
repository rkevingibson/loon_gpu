#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>

#include "gpu/loon_gpu.h"
#include "platform_utils.h"

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

// MARK: Label
class Label {
   public:
    Label() = default;
    Label(const Allocator& backup_alloc, Span<const char> label) : Label() {
        set(backup_alloc, label);
    }
    Span<const char> get() const;
    void             set(const Allocator& backup_alloc, Span<const char> label);

   private:
    static constexpr uint32_t label_buffer_size = 256;
    uint32_t                  m_capacity        = label_buffer_size;
    uint32_t                  m_len             = 0;
    union {
        char* m_label = nullptr;
        char  m_inline_buffer[label_buffer_size];
    };
};

// MARK: Arena

class Arena {
   public:
    Arena() = default;
    Arena(void* ptr, size_t size, ProcLogCallback cb, void* userdata) noexcept :
        m_ptr(reinterpret_cast<uintptr_t>(ptr)),
        m_begin(m_ptr),
        m_size(size),
        m_log_callback(cb),
        m_log_userdata(userdata) {};

    Arena(const Arena&)            = default;
    Arena& operator=(const Arena&) = default;

    [[nodiscard]] void* alloc(size_t size) {
        const uintptr_t ptr    = m_ptr;
        const uintptr_t newptr = ptr + size;
        const uintptr_t end    = m_begin + m_size;
        if (newptr > end) {
            m_log_callback(LogLevel::Error,
                           "Thread-local arena out of memory"_sv,
                           __LINE__,
                           "containers.h",
                           m_log_userdata);
            return nullptr;
        }
        m_ptr = newptr;
        return reinterpret_cast<void*>(ptr);
    }

    void free(const void* ptr, size_t size) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        if (p + size == m_ptr && p >= m_begin) { m_ptr = p; }
    }

    [[nodiscard]] bool owns(const void* ptr) const {
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        return p >= m_begin && p < m_begin + m_size;
    }

   private:
    template <class T>
    friend Span<T> concat(Arena* a, Span<T> head, Span<const T> tail);
    template <class T>
    friend Span<const T> concat(Arena* a, Span<const T> head, Span<const T> tail);

    uintptr_t m_ptr{0};
    uintptr_t m_begin{0};
    size_t    m_size{0};

    ProcLogCallback m_log_callback;
    void*           m_log_userdata;
};

template <class T>
[[nodiscard]] Span<T> clone(Arena* a, Span<const T> x) {
    T* output = reinterpret_cast<T*>(a->alloc(x.as_bytes().size()));
    if (output == nullptr) { return {}; }

    Span<T> result(output, x.size());
    for (auto first = x.begin(), last = x.end(); first != last; ++output, ++first) {
        ::new (output) T(*first);
    }
    return result;
}

template <class T>
[[nodiscard]] Span<T> concat(Arena* a, Span<T> head, Span<const T> tail) {
    if ((uintptr_t)head.end() != a->m_ptr) { head = clone<T>(a, head); }
    return {head.data(), head.size() + clone<T>(a, tail).size()};
}

template <class T>
[[nodiscard]] Span<const T> concat(Arena* a, Span<const T> head, Span<const T> tail) {
    if ((uintptr_t)head.end() != a->m_ptr) { head = clone<T>(a, head); }
    return {head.data(), head.size() + clone<T>(a, tail).size()};
}

template <class T>
[[nodiscard]] Span<T> concat(Arena* a, Span<T> head, const T& tail) {
    return concat(a, head, Span<const T>(&tail, 1));
}

[[nodiscard]] inline const char* make_null_terminated(Arena* a, Span<const char> s) {
    char* result = (char*)a->alloc(s.size() + 1);
    if (result) {
        memcpy(result, s.data(), s.size());
        result[s.size()] = '\0';
    }
    return result;
}

// MARK: Vector
template <class T>
class Vector {
   public:
    Vector() = default;
    Vector(Allocator allocator, uint32_t initial_capacity = 0);
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
        T* result = ::new (m_data + m_count) T(v);
        m_count++;
        return *result;
    }

    template <class... Args>
    T& emplace_back(Args&&... args) {
        if (m_count == m_capacity) { reserve(grow_capacity(m_count + 1)); }
        T* result = ::new (m_data + m_count) T(std::forward<Args>(args)...);
        m_count++;
        return *result;
    }

    void pop_back() {
        m_count--;
        (m_data + m_count)->~T();
    }

    void erase(T* first, T* last) {
        const auto old_end = end();
        m_count -= (last - first);
        // Move from [last, old_end) to [first, new_end)
        for (; last != old_end; ++first, ++last) *first = std::move(*last);
        for (auto it = end(); it != old_end; ++it) { it->~T(); }
    }

    T* insert(T* pos, const T& value) {
        if (m_count == m_capacity) {
            // NOTE: This is suboptimal - does 2 moves of elements after the insertion.
            const auto idx = pos - m_data;
            reserve(grow_capacity(m_count + 1));
            pos = m_data + idx;
        }

        // Move from pos, end to pos + 1, end+1
        ::new (end()) T(std::move(*(end() - 1)));
        for (auto it = end() - 1; it > pos; --it) { *it = std::move(*(it - 1)); }
        ::new (pos) T(value);
        m_count++;
        return pos;
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

// MARK: SegmentArray

// As described by https://danielchasehooper.com/posts/segment_array, an array that gives
// pointer-stability on growth.
template <class T>
class SegmentArray {
   public:
    using DestructorFn = Function<void, T*>;
    SegmentArray()     = default;
    SegmentArray(Allocator alloc) : m_allocator(alloc) {};
    ~SegmentArray() { clear(); };

    SegmentArray(const SegmentArray&)            = delete;
    SegmentArray& operator=(const SegmentArray&) = delete;
    SegmentArray(SegmentArray&& other) : SegmentArray() { swap(*this, other); }
    SegmentArray& operator=(SegmentArray&& other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(SegmentArray& a, SegmentArray& b) {
        using std::swap;
        swap(a.m_allocator, b.m_allocator);
        swap(a.m_count, b.m_count);
        swap(a.m_used_segments, b.m_used_segments);
        swap(a.m_segments, b.m_segments);
    }

    void clear() {
        uint32_t remaining_count = m_count;
        for (uint32_t segment_idx = 0; segment_idx < m_used_segments; ++segment_idx) {
            const uint32_t segment_size = slots_in_segment(segment_idx);
            T*             segment      = m_segments[segment_idx];
            // Destroy any objects in this segment up to count
            for (uint32_t idx = 0; idx < segment_size && remaining_count > 0;
                 ++idx, --remaining_count) {
                segment[idx].~T();
            }

            m_allocator.free({
                .ptr = segment,
                .len = static_cast<uint32_t>(segment_size * sizeof(T)),
            });
            m_segments[segment_idx] = nullptr;
        }
        m_used_segments = 0;
        m_count         = 0;
    }

    template <class... Args>
    T& emplace_back(Args&&... args) {
        if (m_count == capacity_for_segment_count(m_used_segments)) { add_segment(); }
        T* entry = get(m_count++);
        entry    = ::new (entry) T(std::forward<Args>(args)...);
        return *entry;
    }

    const T&           operator[](uint32_t idx) const { return *get(idx); }
    T&                 operator[](uint32_t idx) { return *get(idx); }
    constexpr uint32_t size() const { return m_count; }

   private:
    static constexpr uint32_t kSmallSegmentsToSkip = 6;
    static constexpr uint32_t slots_in_segment(uint32_t segment_index) {
        return (1 << kSmallSegmentsToSkip) << segment_index;
    }
    static constexpr uint32_t capacity_for_segment_count(uint32_t segment_count) {
        return ((1 << kSmallSegmentsToSkip) << segment_count) - (1 << kSmallSegmentsToSkip);
    }
    void add_segment() {
        const size_t segment_size     = slots_in_segment(m_used_segments);
        const auto   blk              = m_allocator.alloc(sizeof(T) * segment_size);
        m_segments[m_used_segments++] = reinterpret_cast<T*>(blk.ptr);
    }
    T* get(uint32_t idx) const {
        const uint64_t segment = int_log_2((idx >> kSmallSegmentsToSkip) + 1);
        uint32_t       slot    = idx - capacity_for_segment_count(segment);
        return &m_segments[segment][slot];
    }

    uint32_t  m_used_segments = 0;
    uint32_t  m_count         = 0;
    Allocator m_allocator;
    // Smallest segment is 64 items, 26 segments get us to ~4 billion items
    T* m_segments[26] = {nullptr};
};

// MARK: Slot Map

// This combines a segment array, as described in https://danielchasehooper.com/posts/segment_array
// with a intrusive freelist. This gives us a way to create handles to objects, with fast O(1)
// accesses, pointer stability (you can access an object while other threads create/destroy objects)
// and dyanmic growth.

template <class T>
class SlotMap {
   public:
    using DestructorFn = Function<void, T*>;
    SlotMap()          = default;
    SlotMap(Allocator alloc, DestructorFn destructor);
    ~SlotMap();

    SlotMap(const SlotMap&)            = delete;
    SlotMap& operator=(const SlotMap&) = delete;

    void clear();

    Handle<T> emplace(T&& val);
    void      erase(Handle<T> h);

    T&       operator[](Handle<T> h);
    const T& operator[](Handle<T> h) const;

   private:
    static constexpr uint32_t kSmallSegmentsToSkip = 6;
    static constexpr uint32_t kNotInFreelist       = UINT32_MAX;
    static constexpr uint32_t kEndOfList           = kNotInFreelist - 1;
    struct Entry {
        T        data;
        uint32_t next;
        uint32_t gen;
    };
    struct DecomposedHandle {
        uint32_t idx;
        uint32_t gen;
    };

    static constexpr uint32_t slots_in_segment(uint32_t segment_index) {
        return (1 << kSmallSegmentsToSkip) << segment_index;
    }
    static constexpr uint32_t capacity_for_segment_count(uint32_t segment_count) {
        return ((1 << kSmallSegmentsToSkip) << segment_count) - (1 << kSmallSegmentsToSkip);
    }
    void                    add_segment();
    Entry*                  get(uint32_t idx);
    static Handle<T>        create_handle(uint32_t idx, uint32_t gen);
    static DecomposedHandle decompose_handle(Handle<T> h);

    mutex        m_mutex         = LOON_MUTEX_INIT;
    uint32_t     m_used_segments = 0;
    uint32_t     m_head          = kEndOfList;
    Allocator    m_allocator;
    DestructorFn m_destructor_fn;
    // Smallest segment is 64 items, 26 segments get us to ~4 billion items
    Entry* m_segments[26] = {nullptr};
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
Vector<T>::Vector(Allocator allocator, const T& value, uint32_t count) : Vector(allocator, count) {
    m_count = count;
    for (uint32_t i = 0; i < count; ++i) { ::new (m_data + i) T(value); }
}

template <class T>
void Vector<T>::clear() {
    if (m_data) {
        for (uint32_t i = 0; i < m_count; ++i) { m_data[i].~T(); }
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

        for (uint32_t i = 0; i < m_count; ++i) {
            ::new (reinterpret_cast<T*>(blk.ptr) + i) T(std::move(m_data[i]));
            m_data[i].~T();
        }
        new_capacity = blk.len / sizeof(T);
        m_allocator.free(current_blk);
        m_capacity = new_capacity;
        m_data     = reinterpret_cast<T*>(blk.ptr);
    }
}

// MARK: Slot Map

template <class T>
SlotMap<T>::SlotMap(Allocator alloc, DestructorFn destructor) :
    m_allocator(alloc), m_destructor_fn{std::move(destructor)} {}

template <class T>
SlotMap<T>::~SlotMap() {
    clear();
}

template <class T>
void SlotMap<T>::clear() {
    mutex_lock(&m_mutex);
    for (uint32_t segment_idx = 0; segment_idx < m_used_segments; ++segment_idx) {
        const uint32_t segment_size = slots_in_segment(segment_idx);
        Entry*         segment      = m_segments[segment_idx];

        // Destroy any objects in this segment that aren't in the free'd state
        for (uint32_t idx = 0; idx < segment_size; ++idx) {
            if (segment[idx].next == kNotInFreelist) { m_destructor_fn(&segment[idx].data); }
        }

        m_allocator.free({
            .ptr = segment,
            .len = static_cast<uint32_t>(segment_size * sizeof(Entry)),
        });
        m_segments[segment_idx] = nullptr;
    }
    m_used_segments = 0;
    mutex_unlock(&m_mutex);
}

template <class T>
Handle<T> SlotMap<T>::emplace(T&& val) {
    mutex_lock(&m_mutex);
    if (m_head == kEndOfList) { add_segment(); }

    const uint32_t idx = m_head;
    assert(idx != kNotInFreelist &&
           idx != kEndOfList);  // Out of memory, need to identify a leak somewhere.

    Entry* entry = get(idx);
    assert(entry->next != kNotInFreelist);
    m_head      = entry->next;
    entry->next = kNotInFreelist;
    ::new (&entry->data) T(std::forward<T&&>(val));
    mutex_unlock(&m_mutex);

    return create_handle(idx, ++entry->gen);
}

template <class T>
void SlotMap<T>::erase(Handle<T> h) {
    mutex_lock(&m_mutex);
    const auto [idx, gen] = decompose_handle(h);
    Entry* entry          = get(idx);
    assert(entry->gen == gen);
    m_destructor_fn(&entry->data);
    entry->next = m_head;
    m_head      = idx;
    mutex_unlock(&m_mutex);
}

template <class T>
T& SlotMap<T>::operator[](Handle<T> h) {
    const auto [idx, gen] = decompose_handle(h);
    Entry* e              = get(idx);
    assert(e && e->gen == gen);
    return e->data;
}

template <class T>
const T& SlotMap<T>::operator[](Handle<T> h) const {
    const auto [idx, gen] = decompose_handle(h);
    Entry* e              = get(idx);
    assert(e && e->gen == gen);
    return e->data;
}

template <class T>
void SlotMap<T>::add_segment() {
    const size_t segment_size     = slots_in_segment(m_used_segments);
    const auto   blk              = m_allocator.alloc(sizeof(Entry) * segment_size);
    auto         segment          = reinterpret_cast<Entry*>(blk.ptr);
    m_segments[m_used_segments++] = segment;
    // Now add this segment to the freelist, back to front.
    const uint32_t segment_offset = capacity_for_segment_count(m_used_segments - 1);
    for (size_t i = segment_size; i > 0; --i) {
        segment[i - 1].gen  = 0;
        segment[i - 1].next = m_head;
        m_head              = i + segment_offset;
    }
}

template <class T>
SlotMap<T>::Entry* SlotMap<T>::get(uint32_t idx) {
    const uint64_t segment = int_log_2((idx >> kSmallSegmentsToSkip) + 1);
    uint32_t       slot    = idx - capacity_for_segment_count(segment);
    return &m_segments[segment][slot];
}

template <class T>
Handle<T> SlotMap<T>::create_handle(uint32_t idx, uint32_t gen) {
    return {.h = (0x8000'0000'0000'0000 | (uint64_t)gen) << 32ull | idx};
}

template <class T>
SlotMap<T>::DecomposedHandle SlotMap<T>::decompose_handle(Handle<T> h) {
    return {
        .idx = static_cast<uint32_t>(h.h & 0xFFFF'FFFFull),
        .gen = static_cast<uint32_t>((h.h >> 32) & 0x7FFF'FFFFull),
    };
}

}  // namespace loon::gpu
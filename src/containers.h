#pragma once

#include <mutex>
#include <utility>

#include "gpu/loon_gpu.h"


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
    Allocator(WGPULoonProcAllocatorCallback alloc, void* userdata) :
        m_alloc{alloc}, m_userdata{userdata} {};
    Allocator(const WGPULoonInstanceConfiguration& alloc);

    friend void swap(Allocator& a, Allocator& b) {
        std::swap(a.m_alloc, b.m_alloc);
        std::swap(a.m_userdata, b.m_userdata);
    }

    constexpr WGPULoonMemoryBlock realloc(WGPULoonMemoryBlock blk, size_t new_size) const {
        return m_alloc(m_userdata, blk.ptr, blk.len, new_size);
    }
    constexpr WGPULoonMemoryBlock alloc(size_t size) const {
        return m_alloc(m_userdata, nullptr, 0, size);
    }
    constexpr void free(WGPULoonMemoryBlock blk) const { m_alloc(m_userdata, blk.ptr, blk.len, 0); }

   private:
    WGPULoonProcAllocatorCallback m_alloc    = nullptr;
    void*                         m_userdata = nullptr;
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

// MARK: Hash Table

template <class Key, class Value>
class HashTable {
   public:
    HashTable() {};
    HashTable(Allocator& allocator, uint32_t initial_size) :
        m_allocator(allocator),
        m_key_values(allocator),
        m_metadata(allocator, Metadata{0, 0}, initial_size) {
        m_resize_threshold = (initial_size * kLoadFactorPercent) / 100;
        m_mask             = initial_size - 1;
    }
    HashTable(HashTable const& other) = delete;
    HashTable(HashTable&& other) : HashTable() { swap(*this, other); }
    HashTable& operator=(const HashTable& other) = delete;
    HashTable& operator=(HashTable&& other) {
        swap(*this, other);
        return *this;
    }

    ~HashTable() = default;

    friend void swap(HashTable& a, HashTable& b) {
        using std::swap;
        swap(a.m_key_values, b.m_key_values);
        swap(a.m_metadata, b.m_metadata);
        swap(a.m_allocator, b.m_allocator);
        swap(a.m_resize_threshold, b.m_resize_threshold);
        swap(a.m_mask, b.m_mask);
        swap(a.m_hasher, b.m_hasher);
    }

    struct KeyValuePair {
        Key   key;
        Value value;
    };

    KeyValuePair*       begin() { return m_key_values.begin(); }
    KeyValuePair*       end() { return m_key_values.end(); }
    const KeyValuePair* begin() const { return m_key_values.begin(); }
    const KeyValuePair* end() const { return m_key_values.end(); }

    size_t size() const { return m_key_values.size(); }

    const KeyValuePair* find(const Key& key) const { return find_impl(key); };
    KeyValuePair*       find(const Key& key) { return const_cast<KeyValuePair*>(find_impl(key)); }
    bool                contains(const Key& key) { return find(key) != nullptr; }

    struct InsertResult {
        KeyValuePair* pair;
        bool          inserted;
    };

    InsertResult insert(KeyValuePair&& slot);
    template <class M>
    InsertResult insert_or_assign(const Key& k, M&& v);


    void clear() {
        m_key_values.clear();
        m_metadata.clear();
    }

   private:
    struct Metadata {
        uint32_t hash  = 0;
        uint32_t index = 0;
    };
    uint32_t              hash_key(const Key& key) const;
    static constexpr bool is_deleted(uint32_t hash) { return (hash >> 31) != 0; }
    uint32_t              desired_pos(uint32_t hash) const { return hash & m_mask; }
    uint32_t              probe_distance(uint32_t hash, uint32_t slot_index) const {
        return (slot_index + m_metadata.size() - desired_pos(hash)) & m_mask;
    }
    const KeyValuePair*       find_impl(const Key& key) const;
    void                      grow();
    uint32_t&                 elem_hash(int idx) { return m_metadata[idx].hash; }
    uint32_t                  elem_hash(int idx) const { return m_metadata[idx].hash; }
    uint32_t                  elem_idx(int idx) const { return m_metadata[idx].index; }
    static constexpr uint32_t kLoadFactorPercent = 90;

    Vector<KeyValuePair> m_key_values;
    Vector<Metadata>     m_metadata;
    uint32_t             m_resize_threshold = 0;
    uint32_t             m_mask             = 0;
    Allocator            m_allocator;
    NO_UNIQUE_ADDR std::hash<Key> m_hasher{};
};

// MARK: Stack

template <class T, uint32_t Size>
class Stack {
   public:
    void                      push(const T& val) { m_data[m_size++] = val; }
    void                      resize(uint32_t size) { m_size = size; }
    T*                        data() { return m_data; }
    const T*                  data() const { return m_data; }
    const T*                  begin() const { return m_data; }
    const T*                  end() const { return m_data + m_size; }
    uint32_t                  size() const { return m_size; }
    static constexpr uint32_t capacity() { return Size; }
    T&                        operator[](uint32_t i) { return m_data[i]; }
    const T&                  operator[](uint32_t i) const { return m_data[i]; }

   private:
    T        m_data[Size];
    uint32_t m_size = 0;
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
    new_capacity                          = new_capacity > 4 ? new_capacity : 4;
    const WGPULoonMemoryBlock current_blk = {
        .ptr = static_cast<void*>(m_data),
        .len = static_cast<uint32_t>(m_capacity * sizeof(T)),
    };
    if constexpr (std::is_trivially_copyable_v<T>) {
        WGPULoonMemoryBlock blk = m_allocator.realloc(current_blk, new_capacity * sizeof(T));
        if (blk.ptr == nullptr) { return; }
        new_capacity = blk.len / sizeof(T);
        m_capacity   = new_capacity;
        m_data       = reinterpret_cast<T*>(blk.ptr);
    } else {
        WGPULoonMemoryBlock blk = m_allocator.alloc(new_capacity * sizeof(T));
        if (blk.ptr == nullptr) { return; }
        std::uninitialized_move_n(m_data, m_count, reinterpret_cast<T*>(blk.ptr));
        std::destroy_n(m_data, m_count);
        new_capacity = blk.len / sizeof(T);
        m_allocator.free(current_blk);
        m_capacity = new_capacity;
        m_data     = reinterpret_cast<T*>(blk.ptr);
    }
}


// MARK: Hash table

template <class K, class V>
const HashTable<K, V>::KeyValuePair* HashTable<K, V>::find_impl(const K& key) const {
    if (m_key_values.is_empty()) { return nullptr; }
    const uint32_t hash = hash_key(key);
    uint32_t       pos  = desired_pos(hash);
    uint32_t       dist = 0;
    for (;;) {
        if (elem_hash(pos) == 0 || dist > probe_distance(elem_hash(pos), pos)) {
            return nullptr;
        } else if (elem_hash(pos) == hash && m_key_values[elem_idx(pos)].key == key) {
            return &m_key_values[elem_idx(pos)];
        }
        pos = (pos + 1) & m_mask;
        ++dist;
    }
    return nullptr;
}

template <class K, class V>
HashTable<K, V>::InsertResult HashTable<K, V>::insert(KeyValuePair&& slot) {
    // Construct the metadata based on where we would insert it if it's not already in the table.
    if (m_key_values.size() >= m_resize_threshold) { grow(); }

    Metadata metadata{
        .hash  = hash_key(slot.key),
        .index = m_key_values.size(),
    };

    uint32_t pos  = desired_pos(metadata.hash);
    uint32_t dist = 0;
    for (;;) {
        // Empty slot:
        if (elem_hash(pos) == 0) {
            m_metadata[pos] = metadata;
            m_key_values.emplace_back(std::move(slot));
            return {
                .pair     = &m_key_values[metadata.index],
                .inserted = true,
            };
        }

        if (elem_hash(pos) == metadata.hash && m_key_values[elem_idx(pos)].key == slot.key) {
            // Found the existing key
            return {
                .pair     = &m_key_values[elem_idx(pos)],
                .inserted = false,
            };
        }

        const uint32_t existing_dist = probe_distance(elem_hash(pos), pos);
        if (dist > existing_dist) {
            // Probe length tells us this map does not contain this key, so we're safe to
            // insert/swap
            m_key_values.emplace_back(std::move(slot));
            if (is_deleted(elem_hash(pos))) {
                // Insert here
                m_metadata[pos] = metadata;
                return {
                    .pair     = &m_key_values[elem_idx(pos)],
                    .inserted = true,
                };
            }

            // Swap with entry at pos
            // Note: we only need to swap metadata, not the actual data.
            std::swap(metadata, m_metadata[pos]);
        }

        pos = (pos + 1) & m_mask;
        ++dist;
    }
}

template <class K, class V>
template <class M>
HashTable<K, V>::InsertResult HashTable<K, V>::insert_or_assign(const K& k, M&& v) {
    auto slot = find(k);
    if (slot) {
        slot->value = std::forward<M>(v);
        return {.pair = slot, .inserted = false};
    } else {
        return insert(KeyValuePair(k, std::forward<M>(v)));
    }
}

template <class K, class V>
void HashTable<K, V>::grow() {
    const uint32_t   old_capacity = m_metadata.size();
    const uint32_t   new_capacity = old_capacity * 2 > 16 ? old_capacity * 2 : 16;
    Vector<Metadata> old_metadata = std::move(m_metadata);

    m_metadata = Vector<Metadata>(m_allocator, Metadata{.hash = 0, .index = 0}, new_capacity);
    m_resize_threshold = (new_capacity * kLoadFactorPercent) / 100;
    m_mask             = new_capacity - 1;

    for (Metadata m : old_metadata) {
        if (m.hash != 0 && !is_deleted(m.hash)) {
            // Rehash the metadata into the new array
            uint32_t pos  = desired_pos(m.hash);
            uint32_t dist = 0;
            for (;;) {
                if (elem_hash(pos) == 0) {
                    m_metadata[pos] = m;
                    break;
                }

                const uint32_t existing_dist = probe_distance(elem_hash(pos), pos);
                if (dist > existing_dist) {
                    // Swap with entry at pos
                    // Note: we only need to swap metadata, not the actual data.
                    std::swap(m, m_metadata[pos]);
                }

                pos = (pos + 1) & m_mask;
                ++dist;
            }
        }
    }
}
template <class Key, class Value>
uint32_t HashTable<Key, Value>::hash_key(const Key& key) const {
    return static_cast<uint32_t>(m_hasher(key));
}

}  // namespace loon::gpu
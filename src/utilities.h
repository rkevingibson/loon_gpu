#pragma once

#include <cassert>
#include <mutex>

#include "containers.h"
#include "platform_utils.h"
#include "volk.h"


namespace loon::gpu {

// MARK: ReferenceCount
struct ReferenceCount {
    void               add() { loon::gpu::atomic_fetch_add(&reference_count, 1); }
    [[nodiscard]] bool release() {
        const auto old_count = loon::gpu::atomic_fetch_add(&reference_count, -1);
        assert((old_count) != 0);
        return (old_count - 1) == 0;
    }
    void    release_all() { loon::gpu::atomic_exchange(&reference_count, 0); }
    int64_t count() { return loon::gpu::atomic_load(&reference_count); };

   private:
    // TODO: Should probably start this at 1, and ensure that I can't add/release once it goes to 0
    int64_t reference_count{0};
};

template <typename T>
T* return_with_ownership(T* ptr) {
    ptr->add_ref();
    return ptr;
}

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
    Arena(void* ptr, size_t size) noexcept :
        m_ptr(reinterpret_cast<uintptr_t>(ptr)), m_begin(m_ptr), m_size(size) {};

    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;

    [[nodiscard]] void* alloc(size_t size) {
        const uintptr_t ptr    = m_ptr;
        const uintptr_t newptr = ptr + size;
        const uintptr_t end    = m_begin + m_size;
        if (newptr > end) { return nullptr; }
        m_ptr = newptr;
        return reinterpret_cast<void*>(ptr);
    }

    void free(const void* ptr, size_t size) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        if (p + size == m_ptr && p >= m_begin) { m_ptr = p; }
    }

    bool owns(const void* ptr) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        return p >= m_begin && p < m_begin + m_size;
    }

   private:
    uintptr_t m_ptr{0};
    uintptr_t m_begin{0};
    size_t    m_size{0};
};

// MARK: ArenaVector

// A simple adapter for creating a stack-like interface on top of an arena.
// Note: This takes advantage of the fact that Arena will have subsequent allocations located
// one after another. As such, only one ArenaVector can be used at a time for an arena - ie.
// push calls should not be interleaved.
// TODO: Get rid of this, replace with some functions on Arena, just to be more explicit.
template <class T>
class ArenaVector {
   public:
    ArenaVector(Arena* arena) :
        m_data(reinterpret_cast<T*>(arena->alloc(0))), m_size(0), m_arena(arena) {}
    ~ArenaVector() { m_arena->free(m_data, m_size * sizeof(T)); }

    bool push(const T& val) {
        // TODO: Worry about alignment?
        auto ptr = m_arena->alloc(sizeof(T));
        if (!ptr) { return false; }
        memcpy(ptr, &val, sizeof(T));
        ++m_size;
        return true;
    }

    T*       data() { return m_data; }
    const T* data() const { return m_data; }
    const T* begin() const { return m_data; }
    const T* end() const { return m_data + m_size; }
    uint32_t size() const { return m_size; }

   private:
    T*       m_data = nullptr;
    uint32_t m_size{0};
    Arena*   m_arena;
};

// MARK: ScopeGuard - a tiny RAII wrapper for deferring work to scope exit.
template <typename T>
class ScopeGuard {
   public:
    ScopeGuard(T&& fn) noexcept : m_fn(fn) {};
    ScopeGuard(ScopeGuard&& other) noexcept {}
    ScopeGuard(const ScopeGuard&)            = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard& operator=(ScopeGuard&&)      = delete;
    ~ScopeGuard() { m_fn(); }

   private:
    T m_fn;
};
template <class T>
ScopeGuard(T) -> ScopeGuard<T>;



};  // namespace loon::gpu
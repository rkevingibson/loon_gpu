#pragma once


#include <memory>

#include "containers.h"

namespace loon::gpu {

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

    Arena(const Arena&)            = default;
    Arena& operator=(const Arena&) = default;

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
    template <class T>
    friend Span<T> concat(Arena* a, Span<T> head, Span<T> tail);

    uintptr_t m_ptr{0};
    uintptr_t m_begin{0};
    size_t    m_size{0};
};

template <class T>
[[nodiscard]] Span<T> clone(Arena* a, Span<T> x) {
    T* output = reinterpret_cast<T*>(a->alloc(x.as_bytes().size()));
    if (output == nullptr) { return {}; }
    std::uninitialized_copy(x.begin(), x.end(), output);
    return {output, x.size()};
}

template <class T>
[[nodiscard]] Span<T> concat(Arena* a, Span<T> head, Span<T> tail) {
    if ((uintptr_t)head.end() != a->m_ptr) { head = clone(a, head); }
    return {head.data(), head.size() + clone<T>(a, tail).size()};
}

template <class T>
[[nodiscard]] Span<T> concat(Arena* a, Span<T> head, T tail) {
    return concat(a, head, Span<T>(&tail, 1));
}

};  // namespace loon::gpu
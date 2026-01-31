#pragma once

#include <cstdint>
namespace loon {

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

    [[nodiscard]] bool owns(const void* ptr) const {
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        return p >= m_begin && p < m_begin + m_size;
    }

    uintptr_t current_ptr() const { return m_ptr; }

   private:
    uintptr_t m_ptr{0};
    uintptr_t m_begin{0};
    size_t    m_size{0};
};

}  // namespace loon
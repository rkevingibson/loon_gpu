#pragma once
#include <cstring>

namespace loon {

class StringView {
   public:
    // Construct an empty Span
    constexpr StringView() noexcept = default;
    constexpr StringView(const char* ptr, size_t len) noexcept : m_ptr{ptr}, m_len{len} {}
    constexpr StringView(const char* begin, const char* end) noexcept :
        m_ptr{begin}, m_len{static_cast<size_t>(end - begin)} {}

    // Construct from fixed size array
    StringView(const char* z_str) noexcept : StringView(z_str, strlen(z_str)) {}

    constexpr StringView(const StringView& src) noexcept = default;
    StringView& operator=(const StringView& src)         = default;

    // Accessors
    constexpr const char*        data() const noexcept { return m_ptr; }
    constexpr size_t             size() const noexcept { return m_len; }
    [[nodiscard]] constexpr bool is_empty() const noexcept { return m_len == 0; }
    constexpr char               operator[](size_t i) const noexcept { return m_ptr[i]; }
    constexpr char               front() const noexcept { return *m_ptr; }
    constexpr char               back() const noexcept { return *(m_ptr + m_len - 1); }

    // Comparisons
    friend constexpr bool operator==(StringView lhs, StringView rhs) noexcept {
        return lhs.size() == rhs.size() && memcmp(lhs.data(), rhs.data(), lhs.size()) == 0;
    }
    friend constexpr bool operator!=(StringView lhs, StringView rhs) noexcept {
        return lhs.size() != rhs.size() || memcmp(lhs.data(), rhs.data(), lhs.size()) != 0;
    }

    // Iterators
    constexpr const char* begin() const noexcept { return m_ptr; }
    constexpr const char* end() const noexcept { return m_ptr + m_len; }
    constexpr const char* cbegin() const noexcept { return begin(); }
    constexpr const char* cend() const noexcept { return end(); }

    // Modifiers
    constexpr void remove_prefix(size_t n) noexcept {
        n = n > m_len ? m_len : n;
        m_ptr += n;
        m_len -= n;
    }
    constexpr void remove_suffix(size_t n) noexcept {
        n = n > m_len ? m_len : n;
        m_len -= n;
    }
    constexpr void swap(StringView& v) noexcept {
        StringView tmp = v;
        v              = *this;
        *this          = tmp;
    }

    constexpr StringView substr(size_t pos = 0, size_t count = ~0) noexcept {
        const size_t rlen = (m_len - pos > count) ? count : m_len - pos;
        return StringView(m_ptr + pos, rlen);
    }

    [[nodiscard]] constexpr bool starts_with(StringView sv) const noexcept {
        return sv.size() <= m_len && StringView(m_ptr, sv.size()) == sv;
    }

    [[nodiscard]] constexpr bool ends_with(StringView sv) const noexcept {
        return sv.size() <= m_len && StringView(m_ptr + m_len - sv.size(), sv.size()) == sv;
    }

    [[nodiscard]] constexpr size_t find_last_of(StringView v) const noexcept {
        for (size_t i = m_len; i > 0; --i) {
            for (size_t j = 0; j < v.m_len; ++j) {
                if (m_ptr[i - 1] == v[j]) { return i - 1; }
            }
        }
        return ~0;
    }

   private:
    const char* m_ptr = nullptr;
    size_t      m_len = 0;
};

};  // namespace loon
#pragma once
#include <cstddef>

#ifndef NO_UNIQUE_ADDRESS
#    if _MSVC_LANG
#        define NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#    else
#        define NO_UNIQUE_ADDRESS [[no_unique_address]]
#    endif
#endif

namespace loon {

template <class T>
struct DefaultDeleter {
    void operator()(T* ptr) { delete ptr; }
};

template <class T, class Deleter = DefaultDeleter<T>>
class Box {
   public:
    constexpr Box() noexcept = default;
    constexpr Box(std::nullptr_t) noexcept {};
    constexpr explicit Box(T* ptr) noexcept : m_ptr(ptr) {};
    constexpr Box(const Box<T>& other) = delete;
    constexpr Box(Box<T>&& other) noexcept : m_ptr(other.release()) {}

    template <class U>
    constexpr Box(Box<U>&& other) noexcept : m_ptr(other.release()) {}

    ~Box() {
        if (m_ptr) { get_deleter()(m_ptr); }
    }

    constexpr Box& operator=(Box&& other) noexcept {
        T* ptr      = other.m_ptr;
        other.m_ptr = m_ptr;
        m_ptr       = ptr;
        return *this;
    }
    constexpr Box& operator=(const Box&) = delete;

    // Modifiers

    constexpr T* release() noexcept {
        T* ptr = m_ptr;
        m_ptr  = nullptr;
        return ptr;
    }

    constexpr void reset(T* ptr = nullptr) noexcept {
        if (m_ptr) { get_deleter()(m_ptr); }
        m_ptr = ptr;
    }

    // Observers

    constexpr T*             get() const noexcept { return m_ptr; }
    constexpr Deleter&       get_deleter() noexcept { return m_deleter; }
    constexpr const Deleter& get_deleter() const noexcept { return m_deleter; }
    explicit                 operator bool() const noexcept { return m_ptr != nullptr; }

    constexpr T& operator*() const noexcept { return *m_ptr; }
    constexpr T* operator->() const noexcept { return m_ptr; }

   private:
    T*                        m_ptr = nullptr;
    NO_UNIQUE_ADDRESS Deleter m_deleter;
};

template <class T, class... Args>
Box<T> make_box(Args&&... args) {
    return Box<T>(new T(static_cast<Args&&>(args)...));
}

}  // namespace loon
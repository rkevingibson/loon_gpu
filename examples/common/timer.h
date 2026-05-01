#pragma once
#include <cstdint>

namespace loon {

class Instant;
class Duration;

class Duration {
   public:
    constexpr Duration(uint64_t nanoseconds) noexcept : m_nanoseconds(nanoseconds) {};

    uint64_t as_nanoseconds() const noexcept { return m_nanoseconds; };
    uint64_t as_microseconds() const noexcept { return m_nanoseconds / 1'000; }
    uint64_t as_milliseconds() const noexcept { return m_nanoseconds / 1'000'000; };

    float as_millseconds_f32() const noexcept {
        return static_cast<float>(m_nanoseconds) / 1'000'000.f;
    }

   private:
    uint64_t m_nanoseconds = 0;
};

class Instant {
   public:
    Instant()                          = delete;
    Instant(const Instant&)            = default;
    Instant& operator=(const Instant&) = default;

    static Instant now() noexcept;
    Duration       duration_since(Instant earlier) const noexcept;
    Duration       elapsed() const noexcept;

   private:
    Instant(uint64_t ns) : m_nanoseconds(ns) {}
    uint64_t m_nanoseconds;
};

}  // namespace loon
#include "timer.h"

#if _WIN32
extern "C" {
__declspec(dllimport) bool __stdcall QueryPerformanceCounter(int64_t* lpPerformanceCount);
__declspec(dllimport) bool __stdcall QueryPerformanceFrequency(int64_t* lpFreq);
}

static uint64_t current_time_ns() {
    static const int64_t qpc_frequency = []() {
        int64_t freq = 1;
        QueryPerformanceFrequency(&freq);
        return freq;
    }();

    int64_t qpc = 0;
    QueryPerformanceCounter(&qpc);

    return static_cast<uint64_t>(qpc * 1'000'000'000 / qpc_frequency);
}

#elif __APPLE__
#    include <time.h>

static uint64_t current_time_ns() {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

#elif __linux__

#    include <time.h>
static uint64_t current_time_ns() {
    timespec result;
    clock_gettime(CLOCK_MONOTONIC, &result);
    return static_cast<uint64_t>(result.tv_sec) * 1'000'000'000ull
           + static_cast<uint64_t>(result.tv_nsec);
}

#endif

namespace loon {

Instant Instant::now() noexcept {
    return Instant(current_time_ns());
};

Duration Instant::duration_since(Instant earlier) const noexcept {
    return Duration(m_nanoseconds - earlier.m_nanoseconds);
}

Duration Instant::elapsed() const noexcept {
    Instant n = Instant::now();
    return n.duration_since(*this);
}

}  // namespace loon
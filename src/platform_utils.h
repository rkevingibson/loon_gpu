#pragma once
// Internal header - platform utilities for implementing webgpu

#include <cstddef>
#include <cstdint>

#if __linux__ || __APPLE__
#    include <pthread.h>
#endif

namespace loon::gpu {

#ifdef _MSVC_LANG
#    define NO_UNIQUE_ADDR [[msvc::no_unique_address]]
#else
#    define NO_UNIQUE_ADDR [[no_unique_address]]
#endif

// Minimal thread-local storage helpers
using tls_key        = ptrdiff_t;
using tls_destructor = void (*)(void*);
tls_key tls_alloc(tls_destructor d);
void    tls_free(tls_key key);
void    tls_set_data(tls_key key, void* data);
void*   tls_get_data(tls_key key);

// Atomic functions
int64_t atomic_exchange(int64_t* x, int64_t val);  // Returns the previous stored value
int64_t atomic_load(int64_t* x);
int64_t atomic_fetch_add(int64_t* x, int64_t val);
bool    atomic_compare_exchange(int64_t* dst, int64_t* expected, int64_t desired);

// Mutex

#if _WIN32
using mutex  = uintptr_t;
using rwlock = uintptr_t;
#    define LOON_MUTEX_INIT  {0}
#    define LOON_RWLOCK_INIT {0}
#elif __linux__ || __APPLE__
using mutex  = pthread_mutex_t;
using rwlock = pthread_rwlock_t;
#    define LOON_MUTEX_INIT  PTHREAD_MUTEX_INITIALIZER
#    define LOON_RWLOCK_INIT PTHREAD_RWLOCK_INITIALIZER
#endif

void mutex_lock(mutex* mtx);
void mutex_unlock(mutex* mtx);
bool mutex_try_lock(mutex* mtx);

void rwlock_lock_read(rwlock* l);
void rwlock_unlock_read(rwlock* l);
void rwlock_lock_write(rwlock* l);
void rwlock_unlock_write(rwlock* l);

// Bit manipulation

inline constexpr uint64_t count_leading_zeros(uint64_t x) {
    if (x == 0) { return 64; }
#if __clang__
    return __builtin_clzll(x);
#elif _MSC_VER
    unsigned long index;
    _BitScanReverse64(&index, x);
    return 63 - index;
#else
#    error Unknown compiler
#endif
}

inline constexpr uint64_t count_trailing_zeros(uint64_t x) {
    if (x == 0) { return 64; }
#if __clang__
    return __builtin_ctzll(x);
#elif _MSC_VER
    unsigned long index;
    _BitScanForward64(&index, x);
    return index;
#else
#    error "Unknown compiler"
#endif
}

inline constexpr uint64_t int_log_2(uint64_t x) {
    return 63 - count_leading_zeros(x);
}

};  // namespace loon::gpu
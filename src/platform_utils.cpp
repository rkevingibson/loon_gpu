#include "platform_utils.h"

#if _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <Windows.h>
#elif __linux__ || __APPLE__
#    include <pthread.h>
#else
#    error "Unimplemented platform"
#endif

namespace webgpu {
#if _WIN32
tls_key tls_alloc(tls_destructor d) {
    return FlsAlloc(d);
}

void tls_free(tls_key k) {
    FlsFree(k);
}

void tls_set_data(tls_key key, void* data) {
    FlsSetValue(key, data);
}

void* tls_get_data(tls_key key) {
    return FlsGetValue(key);
}

int64_t atomic_exchange(int64_t* x, int64_t val) {
    return InterlockedExchange64(x, val);
}

int64_t atomic_load(int64_t* x) {
    return *x;
}

int64_t atomic_fetch_add(int64_t* x, int64_t val) {
    return InterlockedExchangeAdd64(x, val);
}

bool atomic_compare_exchange(int64_t* dst, int64_t* expected, int64_t desired) {
    int64_t original = *expected;
    *expected        = InterlockedCompareExchange64(dst, desired, *expected);
    return original == *expected;
}

#elif __linux__ || __APPLE__
tls_key tls_alloc(tls_destructor d) {
    pthread_key_t key;
    pthread_key_create(&key, d);
    return static_cast<tls_key>(key);
}

void tls_free(tls_key k) {
    pthread_key_delete(static_cast<pthread_key_t>(k));
}

void tls_set_data(tls_key key, void* data) {
    pthread_setspecific(static_cast<pthread_key_t>(key), data);
}

void* tls_get_data(tls_key key) {
    return pthread_getspecific(static_cast<pthread_key_t>(key));
}

int64_t atomic_exchange(int64_t* x, int64_t val) {
    return __atomic_exchange_n(x, val, __ATOMIC_ACQ_REL);
}

int64_t atomic_load(int64_t* x) {
    return __atomic_load_n(x, __ATOMIC_ACQUIRE);
}

int64_t atomic_fetch_add(int64_t* x, int64_t val) {
    return __atomic_fetch_add(x, val, __ATOMIC_ACQ_REL);
}

bool atomic_compare_exchange(int64_t* dst, int64_t* expected, int64_t desired) {
    return __atomic_compare_exchange_n(dst,
                                       expected,
                                       desired,
                                       true,
                                       __ATOMIC_ACQ_REL,
                                       __ATOMIC_RELAXED);
}
#endif

}  // namespace webgpu
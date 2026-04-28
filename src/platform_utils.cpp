#include "platform_utils.h"


#if _WIN32

// We avoid including Windows.h by just defining the handful of functions we care about.
// We still need intrin.h for the atomic intrinsics.
#    include <intrin.h>
extern "C" {
typedef void(__stdcall* PFLS_CALLBACK_FUNCTION)(void* lpFlsData);
__declspec(dllimport) unsigned long __stdcall FlsAlloc(PFLS_CALLBACK_FUNCTION lpCallback);
__declspec(dllimport) void* __stdcall         FlsGetValue(unsigned long dwFlsIndex);
__declspec(dllimport) int __stdcall  FlsSetValue(unsigned long dwFlsIndex, void* lpFlsData);
__declspec(dllimport) int __stdcall  FlsFree(unsigned long dwFlsIndex);
__declspec(dllimport) void __stdcall AcquireSRWLockExclusive(void* SRWLock);
__declspec(dllimport) void __stdcall AcquireSRWLockShared(void* SRWLock);
__declspec(dllimport) int __stdcall  TryAcquireSRWLockExclusive(void* SRWLock);
__declspec(dllimport) void __stdcall ReleaseSRWLockExclusive(void* SRWLock);
__declspec(dllimport) void __stdcall ReleaseSRWLockShared(void* SWRLock);
}

#elif __linux__ || __APPLE__
#    include <pthread.h>
#else
#    error "Unimplemented platform"
#endif

namespace loon::gpu {
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
    return _InterlockedExchange64(x, val);
}

int64_t atomic_load(int64_t* x) {
    return *x;
}

int64_t atomic_fetch_add(int64_t* x, int64_t val) {
    return _InterlockedExchangeAdd64(x, val);
}

bool atomic_compare_exchange(int64_t* dst, int64_t* expected, int64_t desired) {
    int64_t original = *expected;
    *expected        = _InterlockedCompareExchange64(dst, desired, *expected);
    return original == *expected;
}

void mutex_lock(mutex* mtx) {
    AcquireSRWLockExclusive(mtx);
}

void mutex_unlock(mutex* mtx) {
    ReleaseSRWLockExclusive(mtx);
}
bool mutex_try_lock(mutex* mtx) {
    return TryAcquireSRWLockExclusive(mtx) != 0;
}

void rwlock_lock_read(rwlock* l) {
    AcquireSRWLockShared(l);
}

void rwlock_unlock_read(rwlock* l) {
    ReleaseSRWLockShared(l);
}

void rwlock_lock_write(rwlock* l) {
    AcquireSRWLockExclusive(l);
}

void rwlock_unlock_write(rwlock* l) {
    ReleaseSRWLockExclusive(l);
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

void mutex_lock(mutex* mtx) {
    pthread_mutex_lock(mtx);
}

void mutex_unlock(mutex* mtx) {
    pthread_mutex_unlock(mtx);
}

bool mutex_try_lock(mutex* mtx) {
    return pthread_mutex_trylock(mtx) == 0;
}

void rwlock_lock_read(rwlock* l) {
    pthread_rwlock_rdlock(l);
}

void rwlock_unlock_read(rwlock* l) {
    pthread_rwlock_unlock(l);
}

void rwlock_lock_write(rwlock* l) {
    pthread_rwlock_wrlock(l);
}

void rwlock_unlock_write(rwlock* l) {
    pthread_rwlock_unlock(l);
}

#endif

}  // namespace loon::gpu
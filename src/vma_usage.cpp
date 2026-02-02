#define VMA_IMPLEMENTATION

#define VMA_CONFIGURATION_USER_INCLUDES_H <cassert>
#define VMA_USE_STL_SHARED_MUTEX          0
#include <utility>

#include "platform_utils.h"
class VmaMutex {
   public:
    VmaMutex()                           = default;
    VmaMutex(const VmaMutex&)            = delete;
    VmaMutex& operator=(const VmaMutex&) = delete;
    void      Lock() { loon::gpu::mutex_lock(&m_mutex); }
    void      Unlock() { loon::gpu::mutex_unlock(&m_mutex); }
    bool      TryLock() { return loon::gpu::mutex_try_lock(&m_mutex); }

   private:
    loon::gpu::mutex m_mutex = LOON_MUTEX_INIT;
};

// Sorting is very much not a bottleneck in VMA and I'm not even sure we hit it in our code paths.
// We define it here to avoid vma including <algorithm>, which slows down compilation. This is not
// the fastest sort in the world, but it's fast enough.
template <class RandomIt, class Compare>
static void sort(RandomIt beg, RandomIt end, Compare cmp) {
    const auto partition_pivot = [&](const RandomIt a, const RandomIt b) -> RandomIt {
        auto pivot_value = *(b - 1);  // Our pivot value is the last element
        auto pivot       = a;         // Temporary pivot iterator

        // Loop over the list.
        // If we find an entry less than the pivot value, we want it towards the front of the list.
        for (auto it = a; it != b - 1; ++it) {
            if (cmp(*it, pivot_value)) {
                std::swap(*pivot, *it);
                ++pivot;
            }
        }
        std::swap(*pivot, *(b - 1));
        return pivot;
    };

    while (end - beg > 1) {
        auto cut = partition_pivot(beg, end);
        sort(cut, end, cmp);
        end = cut;
    }
}

#define VMA_MUTEX               VmaMutex
#define VMA_MIN(v1, v2)         (((v1) < (v2)) ? (v1) : (v2))
#define VMA_MAX(v1, v2)         (((v1) > (v2)) ? (v1) : (v2))
#define VMA_SORT(beg, end, cmp) (sort(beg, end, cmp))

#include "vma_usage.h"
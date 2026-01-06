#include "containers.h"

#include "utilities.h"

namespace loon::gpu {

Allocator::Allocator() {
    m_alloc = [](void*, void* ptr, uint32_t, uint32_t new_size) -> WGPULoonMemoryBlock {
        if (new_size == 0) {
            ::free(ptr);
            return {.ptr = nullptr, .len = 0};
        } else {
            void* new_ptr = ::realloc(ptr, new_size);
            return {.ptr = new_ptr, .len = new_size};
        }
    };
}


}  // namespace loon::gpu
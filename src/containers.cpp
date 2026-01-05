#include "containers.h"

#include "utilities.h"

namespace webgpu {

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
Allocator::Allocator(const WGPULoonInstanceConfiguration& alloc) :
    m_alloc{alloc.alloc}, m_userdata{alloc.alloc_userdata} {}

WGPUStringView Label::get() const {
    if (m_len <= label_buffer_size) {
        return {.data = m_inline_buffer, .length = m_len};
    } else {
        return {.data = m_label, .length = m_len};
    }
}

void Label::set(const Allocator& backup_alloc, WGPUStringView label) {
    const uint32_t len
        = label.length == WGPU_STRLEN ? (label.data ? strlen(label.data) : 0) : label.length;

    if (len > m_capacity) {
        // Need to realloc.
        if (m_capacity > label_buffer_size) {
            // Old string is stored, need to free it
            backup_alloc.free({m_label, m_capacity});
        }

        auto new_block = backup_alloc.alloc(len);

        if (new_block.ptr == nullptr) {  // If alloc fails, we can just truncate the copy
            m_capacity = label_buffer_size;
        } else {
            m_capacity = new_block.len;
            m_label    = (char*)new_block.ptr;
        }
    }

    m_len = len <= m_capacity ? len : m_capacity;
    if (m_capacity > label_buffer_size) {
        memcpy(m_label, label.data, m_len);
    } else {
        memcpy(m_inline_buffer, label.data, m_len);
    }
}

}  // namespace webgpu
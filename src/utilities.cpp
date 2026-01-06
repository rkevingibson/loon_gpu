#include "utilities.h"

#include <cstring>

#include "gpu/loon_gpu.h"
#include "volk.h"


namespace loon::gpu {

Span<const char> Label::get() const {
    if (m_len <= label_buffer_size) {
        return Span<const char>(m_inline_buffer, m_len);
    } else {
        return Span<const char>(m_label, m_len);
    }
}

void Label::set(const Allocator& backup_alloc, Span<const char> label) {
    const uint32_t len = label.size();

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
        memcpy(m_label, label.data(), m_len);
    } else {
        memcpy(m_inline_buffer, label.data(), m_len);
    }
}

}  // namespace loon::gpu
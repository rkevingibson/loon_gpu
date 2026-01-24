#include "containers.h"

#include "platform_utils.h"

namespace loon::gpu {

Allocator::Allocator() {
    m_alloc = [](void*, void* ptr, uint32_t, uint32_t new_size) -> MemoryBlock {
        if (new_size == 0) {
            ::free(ptr);
            return {.ptr = nullptr, .len = 0};
        } else {
            void* new_ptr = ::realloc(ptr, new_size);
            return {.ptr = new_ptr, .len = new_size};
        }
    };
}

static constexpr uint32_t div_round_up(uint32_t n, uint32_t div) {
    return (n - 1) / div + 1;
}

static constexpr uint32_t num_entries(uint32_t n) {
    const uint32_t num_leaf_entries   = div_round_up(n, 64);
    const uint32_t num_header_entries = div_round_up(n, 64 * 64);
    return num_leaf_entries + num_header_entries;
}

static constexpr uint32_t num_header_entries(uint32_t n) {
    return div_round_up(n, 64 * 64);
}

TwoLevelBitset::TwoLevelBitset(Allocator alloc, uint32_t size) :
    m_data(alloc, 0ull, num_entries(size)), m_size{size} {}

uint32_t TwoLevelBitset::set_leading_zero() {
    // Two stage lookup: First, among the header entries find the first zero bit:

    // TODO: Clean this up, make thread safe via compare-and-swap?

    uint32_t       header_idx  = 0;
    const uint32_t header_size = num_header_entries(m_size);
    while (header_idx < header_size && m_data[header_idx] == UINT64_MAX) { header_idx++; }
    if (header_idx == header_size) {  // Bitset is all full up.
        return ~0;
    }

    const uint32_t chunk_idx = count_trailing_zeros(~m_data[header_idx]);
    const uint32_t chunk_pos = count_trailing_zeros(~m_data[header_size + chunk_idx]);
    m_data[header_size + chunk_idx] |= 1ull << chunk_pos;
    if (m_data[header_size + chunk_idx] == UINT64_MAX) { m_data[header_idx] |= 1ull << chunk_idx; }
    return 64 * chunk_idx + chunk_pos;
}

void TwoLevelBitset::clear_bit(uint32_t idx) {
    const uint32_t header_size = num_header_entries(m_size);
    const uint32_t chunk_idx   = idx / 64;
    const uint32_t header_idx  = chunk_idx / 64;
    const uint64_t mask        = ~(1ull << (idx & 63));
    const uint64_t header_mask = ~(1ull << (chunk_idx & 63));
    m_data[header_size + chunk_idx] &= mask;
    m_data[header_idx] &= header_mask;
}


}  // namespace loon::gpu
#include <gtest/gtest.h>

#include "containers.h"
using namespace loon::gpu;

TEST(utilities_tests, vector_basic_usage) {
    // With no allocator, all operations should work fine, using the default allocator
    Vector<uint32_t> a{};
    auto             result = a.push_back(42);
    ASSERT_EQ(result, 42);
    ASSERT_EQ(a.size(), 1);
    ASSERT_NE(a.data(), nullptr);

    // Push a bunch to ensure a re-allocation is made:
    uint32_t* ptr = a.data();
    while (ptr == a.data()) { ASSERT_EQ(a.push_back(1), 1); }
    ASSERT_EQ(*a.begin(), 42);
    a.clear();
    ASSERT_EQ(a.size(), 0);

    // TODO: Check other allocators, work on allocation failure case, etc.
}


TEST(utilities_tests, two_level_bitset) {
    TwoLevelBitset bitset(Allocator(), 1024);

    for (uint32_t i = 0; i < 1024; ++i) {
        const uint32_t val = bitset.set_leading_zero();
        ASSERT_EQ(val, i);
    }

    for (uint32_t i = 0; i < 1024; i += 2) { bitset.clear_bit(i); }

    for (uint32_t i = 0; i < 1024 / 2; ++i) {
        const uint32_t val = bitset.set_leading_zero();
        ASSERT_EQ(val, 2 * i);
    }
}
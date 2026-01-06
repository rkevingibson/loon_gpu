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

TEST(utilities_tests, hash_table_test) {
    HashTable<uint64_t, uint32_t> table;

    const auto result = table.insert({42, 999});
    ASSERT_TRUE(result.inserted);
    ASSERT_NE(result.pair, nullptr);
    ASSERT_EQ(result.pair->value, 999);

    // Trying to reinsert won't overwrite previously inserted value
    const auto second_result = table.insert({42, 1234});
    ASSERT_FALSE(second_result.inserted);
    ASSERT_EQ(second_result.pair, result.pair);

    for (uint64_t i = 0; i < 100; i++) { table.insert({i, {}}); }

    const auto lookup_result = table.find(42);
    ASSERT_NE(lookup_result, nullptr);

    const auto replaced_result = table.insert_or_assign(42, 0);
    ASSERT_EQ(replaced_result.inserted, false);
    ASSERT_EQ(replaced_result.pair->value, 0);
}
#include "filesystem.h"
#include "geometry.h"
#include "utest.h"

void utest_type_printer(const loon::StringView& v) {
    UTEST_PRINTF("%.*s", (int)v.size(), v.data());
}

using namespace loon;
using namespace geometry;

void assert_float3_eq(int* utest_result, const float3& a, const float3& b) {
    ASSERT_NEAR(a.x, b.x, 1e-5);
    ASSERT_NEAR(a.y, b.y, 1e-5);
    ASSERT_NEAR(a.z, b.z, 1e-5);
}

void assert_transform_eq(int* utest_result, const transform3d& a, const transform3d& b) {
    for (int i = 0; i < 3; ++i) { assert_float3_eq(utest_result, a.basis[i], b.basis[i]); }
    assert_float3_eq(utest_result, a.origin, b.origin);
}

UTEST(geometry_tests, rotations) {
    struct AxisAngle {
        float3 axis;
        float  angle;
    };

    const AxisAngle test_params[] = {
        {.axis = {1, 0, 0}, .angle = kPi},
        {.axis = {0, 1, 0}, .angle = kPi / 2},
        {.axis = {0, 0, 1}, .angle = kPi / 2},
        {.axis = normalized({1, 1, 0}), .angle = kPi / 2},
    };

    for (size_t i = 0; i < sizeof(test_params) / sizeof(test_params[0]); ++i) {
        const auto& p  = test_params[i];
        auto        t1 = transform3d::identity().rotated_local(p.axis, p.angle);
        auto        t2 = transform3d::identity().rotated(p.axis, p.angle);
        // When transform is identity, rotated and rotated local should be identical
        assert_transform_eq(utest_result, t1, t2);
    }
}

UTEST(filesystem_tests, root_path) {
    ASSERT_EQ(loon::filesystem::root_name("C:/testA"), StringView("C:"));
    ASSERT_EQ(loon::filesystem::root_name("C:testARelativePath/test"), StringView("C:"));
    ASSERT_EQ(loon::filesystem::root_name("\\\\test_server\\path_to_file\\hello.txt"),
              StringView("\\\\test_server"));
    ASSERT_EQ(loon::filesystem::root_name("/home/linux/test_path"), StringView(""));
}

UTEST(filesystem_tests, normalize_path) {
    constexpr size_t kBufferSize = 2 * 1024ull;
    char             buffer[kBufferSize];
    Arena            arena(buffer, kBufferSize);
    ASSERT_EQ(filesystem::normalize_path(&arena, "C:", '/'), StringView("C:"));
    ASSERT_EQ(filesystem::normalize_path(&arena, "C:\\foo\\..\\bar", '/'), StringView("C:/bar"));
    ASSERT_EQ(filesystem::normalize_path(&arena, "a/.///b/../", '/'), StringView("a/"));
    ASSERT_EQ(filesystem::normalize_path(&arena, "/Users/kgibson/../foo", '/'),
              StringView("/Users/foo"));
    ASSERT_EQ(filesystem::normalize_path(&arena, "/Users/kgibson/../foo/", '/'),
              StringView("/Users/foo/"));
}
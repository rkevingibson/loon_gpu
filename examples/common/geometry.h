#pragma once

#include <cmath>

/// Matrix/vector conventions:
// Since I come from a math background, I generally prefer treating vectors as column vectors, and
// pre-multiplying by transform matrices. As a result, this library sticks to a convention where
// vectors are columns, and matrices are column-major. This doesn't match the default behaviour of
// the slang compiler, but is easily configurable by the session options, and the generated shader
// code should be correct.
// For coordinates, we operate on an assumption that world space is right-handed, Y-up (+z is
// towards camera, out of the screen.)
///

namespace geometry {

struct float3 {
    float x, y, z;
};

struct alignas(16) float4 {
    float x, y, z, w;

    static float4 make(const float3& xyz, float w) noexcept { return {xyz.x, xyz.y, xyz.z, w}; }
};

struct alignas(16) quatf {
    float x, y, z, w;
};

struct float4x4 {
    float4 columns[4];
};

struct transform3d {
    float3 basis[3];
    float3 origin;

    [[nodiscard]] static transform3d identity();
    [[nodiscard]] static transform3d from_basis_and_origin(float3 basis[3], const float3& origin);

    [[nodiscard]] transform3d rotated(const float3& axis, float angle) const;
    [[nodiscard]] transform3d rotated_local(const float3& axis, float angle) const;
    [[nodiscard]] transform3d translated(const float3& offset) const;
    [[nodiscard]] transform3d translated_local(const float3& offset) const;

    [[nodiscard]] float4x4 to_matrix() const;
};

inline constexpr float kPi = 3.14159265358979323846f;
inline constexpr float radians_from_degrees(float degrees) {
    constexpr float kRadiansFromDegrees = kPi / 180.f;
    return degrees * kRadiansFromDegrees;
}

// Computes a*b - c*d, avoiding catastrophic cancellations via Karan's algorithm
inline constexpr float differenceOfProducts(float a, float b, float c, float d) {
    float cd  = c * d;
    float err = std::fmaf(-c, d, cd);
    float dop = std::fmaf(a, b, -cd);
    return dop + err;
}

// Computes a*b + c*d, avoiding catastrophic cancellations via Karan's algorithm
inline constexpr float sumOfProducts(float a, float b, float c, float d) {
    float cd  = c * d;
    float err = std::fmaf(-c, d, cd);
    float sop = std::fmaf(a, b, cd);
    return sop + err;
}

// MARK: float3 operations

inline constexpr bool operator==(const float3& a, const float3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.y;
}

inline constexpr float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline constexpr float3 cross(const float3& a, const float3& b) {
    return {
        .x = differenceOfProducts(a.x, b.z, a.z, b.y),
        .y = differenceOfProducts(a.z, b.x, a.x, b.z),
        .z = differenceOfProducts(a.x, b.y, a.y, b.z),
    };
}

inline constexpr float3 operator*(const float3& a, const float3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline constexpr float3 operator*(const float3& a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

inline constexpr float3 operator*(float a, const float3& b) {
    return {a * b.x, a * b.y, a * b.z};
}

inline constexpr float3 operator+(const float3& a, const float3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline constexpr float squaredLength(const float3& a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

inline constexpr float length(const float3& a) {
    return std::sqrtf(squaredLength(a));
}

inline constexpr float3 normalized(const float3& a) {
    const float invLen = 1.f / length(a);
    return {a.x * invLen, a.y * invLen, a.z * invLen};
}

// MARK: float4 operations

[[nodiscard]] constexpr inline bool operator==(const float4& a, const float4& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

// MARK: quatf operations

quatf fromAxisAngle(const float3& axis, float angle);

// MARK: float4x4 operations

struct ProjectionInfo {
    float view_width;
    float view_height;
    float y_fov;
    float depth_near = INFINITY;
    float depth_far;
};

float4x4 projection(const ProjectionInfo& info);

constexpr inline bool operator==(const float4x4& a, const float4x4& b) {
    return a.columns[0] == b.columns[0] && a.columns[1] == b.columns[1]
           && a.columns[2] == b.columns[2] && a.columns[3] == b.columns[3];
}

// MARK: transform3d operations

[[nodiscard]] constexpr inline bool operator==(const transform3d& a, const transform3d& b) {
    return a.basis[0] == b.basis[0] && a.basis[1] == b.basis[1] && a.basis[2] == b.basis[2]
           && a.origin == b.origin;
}


}  // namespace geometry
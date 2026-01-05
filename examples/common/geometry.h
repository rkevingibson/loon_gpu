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

struct alignas(16) float3 {
    float x, y, z;
};

struct alignas(16) float4 {
    float x, y, z, w;
};

struct alignas(16) quatf {
    float x, y, z, w;
};

struct float4x4 {
    float4 columns[4];
};

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


// MARK: quatf operations

quatf fromAxisAngle(const float3& axis, float angle);


// MARK: float4x4 operations

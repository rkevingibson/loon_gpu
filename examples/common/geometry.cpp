#include "geometry.h"

namespace geometry {

float3x3 float3x3::identity() {
    return {.columns = {
                {1.f, 0.f, 0.f},
                {0.f, 1.f, 0.f},
                {0.f, 0.f, 1.f},
            }};
}

float3x3 float3x3::transpose() const {
    return {.columns = {
                {columns[0].x, columns[1].x, columns[2].x},
                {columns[0].y, columns[1].y, columns[2].y},
                {columns[0].z, columns[1].z, columns[2].z},
            }};
}

float float3x3::determinant() const {
    // This numeric rigorousness might be unnecessary, but it's fairly nice and clean.
    const float min0 =
        difference_of_products(columns[1].y, columns[2].z, columns[1].z, columns[2].y);
    const float min1 =
        difference_of_products(columns[0].y, columns[2].z, columns[0].z, columns[2].y);
    const float min2 =
        difference_of_products(columns[0].y, columns[1].z, columns[0].z, columns[1].y);
    const float det = std::fmaf(columns[2].x,
                                min2,
                                difference_of_products(columns[0].x, min0, columns[1].x, min1));
    return det;
}

float3x3 float3x3::inverse() const {
    const float m00 =
        difference_of_products(columns[1].y, columns[2].z, columns[1].z, columns[2].y);
    const float m10 =
        difference_of_products(columns[0].z, columns[2].y, columns[0].y, columns[2].z);
    const float m20 =
        difference_of_products(columns[0].y, columns[1].z, columns[0].z, columns[1].y);
    const float det =
        std::fmaf(columns[2].x, m20, sum_of_products(columns[0].x, m00, columns[1].x, m10));
    const float inv_det = 1.f / det;
    if (std::isinf(inv_det)) {
        return {
            float3::infinity(),
            float3::infinity(),
            float3::infinity(),
        };
    }
    const float m01 =
        difference_of_products(columns[2].x, columns[1].z, columns[2].z, columns[1].x);
    const float m11 =
        difference_of_products(columns[0].x, columns[2].z, columns[0].z, columns[2].x);
    const float m21 =
        difference_of_products(columns[1].x, columns[0].z, columns[1].z, columns[0].x);
    const float m02 =
        difference_of_products(columns[1].x, columns[2].y, columns[1].y, columns[2].x);
    const float m12 =
        difference_of_products(columns[2].x, columns[0].y, columns[2].y, columns[0].x);
    const float m22 =
        difference_of_products(columns[0].x, columns[1].y, columns[0].y, columns[1].x);

    return float3x3{
        .columns =
            {
                {inv_det * m00, inv_det * m10, inv_det * m20},
                {inv_det * m01, inv_det * m11, inv_det * m21},
                {inv_det * m02, inv_det * m12, inv_det * m22},
            },
    };
}

float4x4 float4x4::inverse() const noexcept {
    const float coef00 =
        difference_of_products(columns[2].z, columns[3].w, columns[3].z, columns[2].z);
    const float coef02 =
        difference_of_products(columns[1].z, columns[3].w, columns[3].z, columns[1].w);
    const float coef03 =
        difference_of_products(columns[1].z, columns[2].w, columns[2].z, columns[1].w);
    const float coef04 =
        difference_of_products(columns[2].y, columns[3].w, columns[3].y, columns[2].w);
    const float coef06 =
        difference_of_products(columns[1].y, columns[3].w, columns[3].y, columns[1].w);
    const float coef07 =
        difference_of_products(columns[1].y, columns[2].w, columns[2].y, columns[1].w);
    const float coef08 =
        difference_of_products(columns[2].y, columns[3].z, columns[3].y, columns[2].z);
    const float coef10 =
        difference_of_products(columns[1].y, columns[3].z, columns[3].y, columns[1].z);
    const float coef11 =
        difference_of_products(columns[1].y, columns[2].z, columns[2].y, columns[1].z);
    const float coef12 =
        difference_of_products(columns[2].x, columns[3].w, columns[3].x, columns[2].w);
    const float coef14 =
        difference_of_products(columns[1].x, columns[3].w, columns[3].x, columns[1].w);
    const float coef15 =
        difference_of_products(columns[1].x, columns[2].w, columns[2].x, columns[1].w);

    const float coef16 =
        difference_of_products(columns[2].x, columns[3].z, columns[3].x, columns[2].z);
    const float coef18 =
        difference_of_products(columns[1].x, columns[3].z, columns[3].x, columns[1].z);
    const float coef19 =
        difference_of_products(columns[1].x, columns[2].z, columns[2].x, columns[1].z);

    const float coef20 =
        difference_of_products(columns[2].x, columns[3].y, columns[3].x, columns[2].y);
    const float coef22 =
        difference_of_products(columns[1].x, columns[3].y, columns[3].x, columns[1].y);
    const float coef23 =
        difference_of_products(columns[1].x, columns[2].y, columns[2].x, columns[1].y);

    const float4 fac0{coef00, coef00, coef02, coef03};
    const float4 fac1{coef04, coef04, coef06, coef07};
    const float4 fac2{coef08, coef08, coef10, coef11};
    const float4 fac3{coef12, coef12, coef14, coef15};
    const float4 fac4{coef16, coef16, coef18, coef19};
    const float4 fac5{coef20, coef20, coef22, coef23};

    const float4 vec0{columns[1].x, columns[0].x, columns[0].x, columns[0].x};
    const float4 vec1{columns[1].y, columns[0].y, columns[0].y, columns[0].y};
    const float4 vec2{columns[1].z, columns[0].z, columns[0].z, columns[0].z};
    const float4 vec3{columns[1].w, columns[0].w, columns[0].w, columns[0].w};

    const float4 inv0{vec1 * fac0 - vec2 * fac1 + vec3 * fac2};
    const float4 inv1{vec0 * fac0 - vec2 * fac3 + vec3 * fac4};
    const float4 inv2{vec0 * fac1 - vec1 * fac3 + vec3 * fac5};
    const float4 inv3{vec0 * fac2 - vec1 * fac4 + vec2 * fac5};

    const float4 signA{1, -1, 1, -1};
    const float4 signB{-1, 1, -1, 1};

    const float4x4 inv = {.columns = {inv0 * signA, inv1 * signB, inv2 * signA, inv3 * signB}};
    const float4   row0{inv.columns[0].x, inv.columns[1].x, inv.columns[2].x, inv.columns[3].x};

    const float4 dot0    = columns[0] * row0;
    const float  det     = (dot0.x + dot0.y) + (dot0.z + dot0.w);
    const float  inv_det = 1.0f / det;


    return inv_det * inv;
}


transform3d transform3d::identity() {
    return transform3d{.basis = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, .origin = {0, 0, 0}};
}

transform3d transform3d::from_axis_angle_and_origin(const float3& axis,
                                                    float         angle,
                                                    const float3& origin) {
    // This is just `rotated()` below, but special-cased where the rotating basis is the identity
    // matrix.
    const float  ca  = cosf(angle);
    const float  sa  = sinf(angle);
    const float  ca1 = (1.0f - ca);
    const float3 x   = {
        ca + ca1 * axis.x * axis.x,
        sa * axis.z + ca1 * axis.x * axis.y,
        sa * -axis.y + ca1 * axis.x * axis.z,
    };
    const float3 y = {
        sa * -axis.z + ca1 * axis.y * axis.x,
        ca + ca1 * axis.y * axis.y,
        sa * axis.x + ca1 * axis.y * axis.z,
    };
    const float3 z = {
        sa * axis.y + ca1 * axis.z * axis.x,
        sa * -axis.x + ca1 * axis.z * axis.y,
        ca + ca1 * axis.z * axis.z,
    };

    return {
        .basis  = {x, y, z},
        .origin = origin,
    };
}

transform3d transform3d::from_basis_and_origin(float3 basis[3], const float3& origin) {
    return transform3d{.basis = {basis[0], basis[1], basis[2]}, .origin = origin};
}

[[nodiscard]] transform3d transform3d::rotated(const float3& axis, float angle) const {
    // Using Rodrigues's rotation formula, we can rotate the basis and the offset.
    const float ca  = cosf(angle);
    const float sa  = sinf(angle);
    const auto  rot = [=](const float3& v) {
        return v * ca + sa * cross(axis, v) + (1.0f - ca) * dot(axis, v) * axis;
    };

    return {
        .basis  = {rot(basis[0]), rot(basis[1]), rot(basis[2])},
        .origin = rot(origin),
    };
}

[[nodiscard]] transform3d transform3d::rotated_local(const float3& axis, float angle) const {
    // Want to compute B * R, where B is our basis, and R is the rotation matrix defined by
    // axis/angle. Rodrigues' rotation formula can give us R * B, which we modify to give us B * R,
    // via some transposing and reversing the order of a cross product (equivalently negating the
    // angle passed to cosf/sinf).
    const float  ca  = cosf(angle);
    const float  sa  = sinf(angle);
    const float3 bx  = {basis[0].x, basis[1].x, basis[2].x};
    const float3 by  = {basis[0].y, basis[1].y, basis[2].y};
    const float3 bz  = {basis[0].z, basis[1].z, basis[2].z};
    const auto   rot = [=](const float3& v) {
        return v * ca + sa * cross(v, axis) + (1.0f - ca) * dot(axis, v) * axis;
    };

    const float3 new_x = rot(bx);
    const float3 new_y = rot(by);
    const float3 new_z = rot(bz);

    return {
        .basis =
            {
                {new_x.x, new_y.x, new_z.x},
                {new_x.y, new_y.y, new_z.y},
                {new_x.z, new_y.z, new_z.z},
            },
        .origin = origin,
    };
}

[[nodiscard]] transform3d transform3d::translated(const float3& offset) const {
    return {.basis = {basis[0], basis[1], basis[2]}, .origin = {origin + offset}};
}

[[nodiscard]] transform3d transform3d::translated_local(const float3& offset) const {
    return {};
}

[[nodiscard]] float4x4 transform3d::to_matrix() const {
    return float4x4{
        .columns =
            {
                float4::make(basis[0], 0),
                float4::make(basis[1], 0),
                float4::make(basis[2], 0),
                float4::make(origin, 1),
            },
    };
}

[[nodiscard]] transform3d transform3d::inverse() const {
    // y = Ax+t <=>  x = (A^-1)(y-t)  <=>  x = (A^-1)y + (A^-1)(-t)

    float3x3 inverse = float3x3::from_columns(basis).inverse();

    return {
        .basis =
            {
                inverse.columns[0],
                inverse.columns[1],
                inverse.columns[2],
            },
        .origin = {inverse * (-origin)},
    };
}

float3 operator*(const float3x3& a, const float3& b) noexcept {
    const float x = a.columns[0].x * b.x + a.columns[1].x * b.y + a.columns[2].x * b.z;
    const float y = a.columns[0].y * b.x + a.columns[1].y * b.y + a.columns[2].y * b.z;
    const float z = a.columns[0].z * b.x + a.columns[1].z * b.y + a.columns[2].z * b.z;
    return {x, y, z};
}

float3x3 operator*(const float3x3& a, const float3x3& b) noexcept {
    return {
        .columns =
            {
                a * b.columns[0],
                a * b.columns[1],
                a * b.columns[2],
            },
    };
}

// Reference: https://iolite-engine.com/blog_posts/reverse_z_cheatsheet
float4x4 projection(const ProjectionInfo& info) {
    // Vulkan clip space goes from -w to w in x/y and 0 to w in z.
    const bool  reverse_z    = info.depth_near > info.depth_far;
    const float n            = reverse_z ? info.depth_far : info.depth_near;
    const float f            = reverse_z ? info.depth_near : info.depth_far;
    const float aspect_ratio = info.view_height / info.view_width;
    const float fov          = 1.f / std::tanf(info.y_fov);
    const float sx           = aspect_ratio * fov;
    const float sy           = -fov;
    if (std::isinf(f)) {
        return float4x4{
            .columns =
                {
                    {sx, 0, 0, 0},
                    {0, sy, 0, 0},
                    {0.f, 0.f, 0.f, -1.f},
                    {0, 0, n, 0.f},
                },
        };
    } else {
    }

    return float4x4{};
}

[[nodiscard]] transform3d operator*(const transform3d& a, transform3d& b) {
    // ( A t )(B u) == ( AB Au+t)
    // ( 0 1 )(0 1)    ( 0    1 )

    const auto A  = float3x3::from_columns(a.basis);
    const auto B  = float3x3::from_columns(b.basis);
    const auto AB = A * B;
    return {
        .basis =
            {
                AB.columns[0],
                AB.columns[1],
                AB.columns[2],
            },
        .origin =
            {
                A * b.origin + a.origin,
            },
    };
}

}  // namespace geometry
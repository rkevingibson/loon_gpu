#include "geometry.h"

namespace geometry {

transform3d transform3d::identity() {
    return transform3d{.basis = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, .origin = {0, 0, 0}};
}

transform3d transform3d::from_basis_and_origin(float3 basis[3], const float3& origin) {
    return transform3d{.basis = {basis[0], basis[1], basis[2]}, .origin = origin};
}

[[nodiscard]] transform3d transform3d::rotated(const float3& axis, float angle) const {
    return *this;
}

[[nodiscard]] transform3d transform3d::rotated_local(const float3& axis, float angle) const {
    // Want to compute B * R, where B is our basis, and R is the rotation matrix defined by
    // axis/angle. Rodrigues' rotation formula can give us R * B, which we can manipulate to get
    // what we want: Negate the angle to get the inverse rotation. Then, instead of applying it to
    // B, apply to B^T, then transpose the output.
    // https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    // Is this faster than just constructing the rotation matrix and multiplying? Unsure.
    const float ca = cosf(angle);
    const float sa = -sinf(angle);

    const float3 bx = {basis[0].x, basis[1].x, basis[2].x};
    const float3 by = {basis[0].y, basis[1].y, basis[2].y};
    const float3 bz = {basis[0].z, basis[1].z, basis[2].z};

    const float3 new_x = ca * bx + sa * cross(axis, bx) + (1.f - ca) * dot(axis, bx) * axis;
    const float3 new_y = ca * by + sa * cross(axis, by) + (1.f - ca) * dot(axis, by) * axis;
    const float3 new_z = ca * bz + sa * cross(axis, bz) + (1.f - ca) * dot(axis, bz) * axis;


    return {
        .basis
        = {{new_x.x, new_y.x, new_z.x}, {new_x.y, new_y.y, new_z.y}, {new_x.z, new_y.z, new_z.z},},
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
    return float4x4 {
        .columns = {
            float4::make(basis[0], 0),
            float4::make(basis[1], 0),
            float4::make(basis[2], 0),
            float4::make(origin, 1),
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
    const float sy           = fov;
    if (std::isinf(f)) {
        return float4x4{
            .columns = {{sx, 0,0,0}, {0,sy, 0,0}, {0.f, 0.f, 0.f, -1.f}, {0, 0, n, 0.f},},
        };
    } else {
    }

    return float4x4{};
}

}  // namespace geometry
#include "geometry.h"

// Reference: https://iolite-engine.com/blog_posts/reverse_z_cheatsheet
float4x4 projection(const ProjectionInfo& info) {
    // Vulkan clip space goes from -w to w in x/y and 0 to w in z.
    const bool  reverse_z    = info.depth_near > info.depth_far;
    const float n            = reverse_z ? info.depth_far : info.depth_near;
    const float f            = reverse_z ? info.depth_near : info.depth_far;
    const float aspect_ratio = info.view_height / info.view_width;
    const float fov          = 1.f / std::tanf(info.y_fov);
    const float sx           = aspect_ratio * fov;
    const float sy           = 1 * fov;
    if (std::isinf(f)) {
        return float4x4{
            .columns = {{sx, 0,0,0}, {0,sy, 0,0}, {0.f, 0.f, 0.f, -1.f}, {0, 0, n, 0.f},},
        };
    } else {
    }

    return float4x4{};
}
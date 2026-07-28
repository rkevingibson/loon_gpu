#include <metal_stdlib>
#include <metal_texture>
using namespace metal;

struct EquirectangularToCubemapArgs {
    uint2 resolution;
    uint2 padding;
    texture2d<float, access::sample> input_texture;
    sampler sampler;
    texture2d_array<half, access::write> output_texture;
};

float2 sphere_pos_to_equirectangular_uv(float3 v)
{
    const float2 scale= float2(0.1591, 0.3183); // Convert from [-pi/2, pi/2] range to [-0.5,0.5] range.
    float u = 0;
    // HACK: atan2(y,x) with ffast-math has a discontinuity with y < 0 and x == 0. 
    if (v.x==0 && v.z < 0) {
        u = -0.5f*3.14159265359f;
    } else {
        u = atan2(v.z, v.x);
    }
    float2 uv = float2(u, asin(v.y));
    uv *= scale;
    uv += 0.5;
    return uv;
}

float3 sphere_pos_from_cubemap_pos(float2 uv, uint16_t face)
{
    float2 face_pos = uv - 0.5f;
    switch(face)
    {
        case 0: return normalize(float3(0.5, face_pos.y, -face_pos.x));
        case 1: return normalize(float3(-0.5, face_pos.y, face_pos.x));
        case 2: return normalize(float3(face_pos.x, -0.5, face_pos.y));
        case 3: return normalize(float3(face_pos.x, 0.5, -face_pos.y));
        case 4: return normalize(float3(face_pos.x, face_pos.y, 0.5));
        case 5: return normalize(float3(-face_pos.x, face_pos.y, -0.5));
    }
    return float3(0,0,0);
}

[[kernel, required_threads_per_threadgroup(8,8,1)]]
void equirectangular_to_cubemap(constant EquirectangularToCubemapArgs* args [[buffer(0)]], uint3 gid [[thread_position_in_grid]])
{
    // Use face_idx & gid to compute a 3d position on the sphere.
    float2 face_uv = float2(gid.xy) / float2(args->resolution);

    for (ushort face_idx = 0; face_idx < 6; face_idx++)
    {
        float3 sphere_pos = sphere_pos_from_cubemap_pos(face_uv, face_idx);
        // TODO: Better filtering here - we're not doing mipmaps but could directly filter samples from the map.
        float2 uv = sphere_pos_to_equirectangular_uv(sphere_pos);
        float4 color = args->input_texture.sample(args->sampler, uv, level(0.0));
        args->output_texture.write(half4(half3(color.xyz),1.0h), ushort2(gid.xy), face_idx);
    }
}

struct IrradianceCubemapFromCubemap
{
    uint2 resolution;
    texturecube<float, access::sample> input_texture;
    sampler sampler;
    texture2d_array<half, access::write> output_texture;
};

[[kernel, required_threads_per_threadgroup(8,8,1)]]
void irradiance_cubemap_from_cubemap(constant IrradianceCubemapFromCubemap* args [[buffer(0)]], uint3 gid [[thread_position_in_grid]])
{
    constexpr float M_PI = 3.14159265359f;
    float2 face_uv = float2(gid.xy) / float2(args->resolution);

    for (ushort face_idx = 0; face_idx < 6; face_idx++)
    {
        float3 normal = sphere_pos_from_cubemap_pos(face_uv, face_idx);

        float3 irradiance = float3(0,0,0);
        float3 up = float3(0,1,0);
        float3 right = normalize(cross(up, normal));
        up = normalize(cross(normal, right));

        const float sample_delta = 0.025;
        uint num_samples = 0;
        for (float phi = 0.0; phi < 2.0 * M_PI; phi += sample_delta) {
            for(float theta = 0.0f; theta < 0.5 * M_PI; theta += sample_delta) {
                float3 tangent_sample = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
                float3 sample_vec = tangent_sample.x * right + tangent_sample.y * up + tangent_sample.z * normal;
                float3 sample = args->input_texture.sample(args->sampler, sample_vec, level(0.0)).rgb;
                irradiance += min(sample, float3(1.0,1.0,1.0)) * cos(theta) * sin(theta);
                num_samples++;
            }
        }

        irradiance = M_PI * irradiance / float(num_samples);
        args->output_texture.write(half4(half3(irradiance), 1.0h), ushort2(gid.xy), face_idx);
    }
}

struct SkyboxArgs {
    float4x4 world_from_camera;
    float2 viewport_size; 

    texturecube<float, access::sample> skybox;
    sampler sampler; 
};

struct SkyboxVertexOutput {
    float4 position [[position]];
    float3 world_space_dir; 
};

[[vertex]]
SkyboxVertexOutput skybox_vertex(constant SkyboxArgs* args [[buffer(0)]], uint32_t vertex_idx [[vertex_id]])
{
    // Draws a fullscreen triangle with verts (0,0), ()
    SkyboxVertexOutput output = {};
    // Coords are (-1,-1), (3, -1), (-1,3)
    const float2 ndc_pos = 2.0*float2((vertex_idx << 1) & 2, vertex_idx & 2) - 1.0f;
    // Unproject ndc position to world space - should multiply by aspect ratio to keep things square
    float aspect_ratio = args->viewport_size.x/args->viewport_size.y;
    const float3 camera_pos = float3(aspect_ratio*ndc_pos.x, ndc_pos.y, 1.0f);

    output.position = float4(ndc_pos, 0.0f, 1.0f);
    output.world_space_dir = (args->world_from_camera * float4(camera_pos, 0.0f)).xyz; 
    return output;
}

[[fragment]]
float4 skybox_fragment(constant SkyboxArgs* args [[buffer(0)]], SkyboxVertexOutput input [[stage_in]])
{
    float4 out_color = args->skybox.sample(args->sampler, normalize(input.world_space_dir));
    return out_color;
}

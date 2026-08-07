#include <metal_stdlib>

using namespace metal;

struct CameraData {
    float4x4 projection;
    float4x4 camera_from_world;
};

struct Mesh {
    float4x4 world_from_mesh;
    constant packed_float3* positions;
    constant packed_float2* uvs;
    constant packed_float3* normals;
};

struct VertexInput {
    CameraData camera;
    Mesh mesh;
};

struct VertexStageOutput 
{
    float4 position [[position]];
    float3 world_pos;
    float2 uv;
    float3 normal;
};

struct FragmentInput {
    texturecube<half, access::sample> irradiance_map;
    sampler sampler;
    packed_float3 camera_pos_world;
    packed_float3 albedo;
};

[[vertex]]
VertexStageOutput vertex_main(constant VertexInput* vert [[buffer(0)]], uint32_t vertexIdx [[vertex_id]]) 
{
    VertexStageOutput output;
    const Mesh mesh = vert->mesh;
    float4x4 camera_from_mesh = vert->camera.camera_from_world * vert->mesh.world_from_mesh;
    float4x4 mvp = vert->camera.projection * camera_from_mesh;


    output.position = mvp * float4(mesh.positions[vertexIdx], 1.0);
    output.world_pos = (vert->mesh.world_from_mesh * float4(mesh.positions[vertexIdx], 1.0)).xyz;
    output.uv = mesh.uvs[vertexIdx];
    output.normal = (vert->mesh.world_from_mesh * float4(mesh.normals[vertexIdx], 1.0)).xyz;
    return output;
}

static float3 fresnel_schlick(float cos_theta, float3 F0) 
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

[[fragment]]
float4 fragment_main(constant FragmentInput* frag [[buffer(1)]], VertexStageOutput vert [[stage_in]])
{
    float3 normal = normalize(vert.normal);
    float3 view_dir = normalize(frag->camera_pos_world - vert.world_pos);
    const float3 F0 = float3(0.04, 0.04, 0.04);
    float3 kS = fresnel_schlick(max(dot(normal, view_dir), 0.0), F0);
    float3 kD = 1.0 - kS;
    float3 irradiance = float3(frag->irradiance_map.sample(frag->sampler, normal).rgb);
    float3 diffuse = irradiance * frag->albedo;
    float3 ambient = (kD * diffuse);

    return float4(ambient, 1.0);
}
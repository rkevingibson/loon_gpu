#include <metal_stdlib>

using namespace metal;

struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct Mesh {
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
    float2 uv;
    float3 normal;
};

struct FragmentInput {
};

[[vertex]]
VertexStageOutput vertex_main(constant VertexInput* vert [[buffer(0)]], uint32_t vertexIdx [[vertex_id]]) 
{
    VertexStageOutput output;
    const Mesh mesh = vert->mesh;
    float4x4 mvp = vert->camera.projection * vert->camera.cameraFromWorld;

    output.position = mvp * float4(mesh.positions[vertexIdx], 1.0);
    output.uv = mesh.uvs[vertexIdx];
    output.normal = mesh.normals[vertexIdx];
    return output;
}

[[fragment]]
float4 fragment_main(VertexStageOutput vert [[stage_in]])
{
    return float4(0.5*vert.normal + 0.5, 1.0);
}
#include <metal_stdlib>
#include <metal_texture>

using namespace metal;


struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct Mesh
{
    constant float3* position;
    constant float2* uvs;
    float4x4 worldFromMesh;
};

struct VertexInput {
    CameraData camera;
    Mesh mesh;
};

struct VertexStageOutput
{
    float4 position [[position]];
    float2 uv;
};

struct FragmentInput {
    texture2d<float, access::sample> texture;
    sampler samp;
};

[[vertex]]
VertexStageOutput vertex_main(
    constant VertexInput* vert [[buffer(0)]],
    uint32_t vertexIdx [[vertex_id]]
)
{
    VertexStageOutput output; 
    const Mesh mesh = vert->mesh;

    float4x4 mvp = vert->camera.projection * vert->camera.cameraFromWorld * mesh.worldFromMesh;

    output.position = mvp * float4(mesh.position[vertexIdx], 1.0);
    output.uv = mesh.uvs[vertexIdx];
    return output;
}

[[fragment]]
float4 fragment_main(
    constant FragmentInput* frag [[buffer(1)]], 
    VertexStageOutput vert [[stage_in]]
)
{
    return frag->texture.sample(frag->samp, vert.uv);
}
#include <metal_stdlib>
#include <metal_texture>

using namespace metal;

struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct VertexInput
{
    float4x4 worldFromMesh;
    constant CameraData* camera;
    constant float3* position;
    constant float2* uvs;
    constant float* padding;
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
VertexStageOutput vertexMain(
    constant VertexInput* vert [[buffer(0)]], 
    uint32_t vertexIdx [[vertex_id]],
    uint32_t instanceIdx [[instance_id]]
)
{
    VertexStageOutput output; 
    constant CameraData* cam = vert[instanceIdx].camera;

    float4x4 mvp = cam->projection * cam->cameraFromWorld * vert[instanceIdx].worldFromMesh;

    output.position = mvp * float4(vert[instanceIdx].position[vertexIdx], 1.0);
    output.uv = vert[instanceIdx].uvs[vertexIdx];
    return output;
}

[[fragment]]
float4 fragmentMain(
    constant FragmentInput* frag [[buffer(1)]],
    VertexStageOutput vert[[stage_in]]
)
{
    return frag->texture.sample(frag->samp, vert.uv);
}
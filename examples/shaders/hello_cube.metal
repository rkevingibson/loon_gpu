#include <metal_stdlib>

using namespace metal;

struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct Mesh
{
    constant float3* position;
    constant float3* color;
    float4x4 worldFromMesh;
};

struct InputData {
    CameraData camera;
    Mesh mesh;
};

struct VertexStageOutput
{
    float4 position [[position]];
    float3 color;
};

[[vertex]] VertexStageOutput vertex_main(constant InputData* args [[buffer(0)]], uint vertexIdx [[vertex_id]])
{
    VertexStageOutput output; 
    const Mesh mesh = args->mesh;

    const float4x4 mvp = args->camera.projection * args->camera.cameraFromWorld * mesh.worldFromMesh;

    output.position = mvp * float4(mesh.position[vertexIdx], 1.0);
    output.color = mesh.color[vertexIdx];
    return output;
}

[[fragment]] float4 fragment_main(VertexStageOutput coarse_vertex [[stage_in]])
{
    return float4(coarse_vertex.color, 1.0);
}
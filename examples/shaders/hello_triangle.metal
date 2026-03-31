#include <metal_stdlib>

using namespace metal;

struct VertexStageOutput
{
    float4 position [[position]];
    float3 color;
};

[[vertex]] VertexStageOutput vertex_main(uint vertexIdx [[vertex_id]])
{
    const float2 verts[3] = {float2(0, 0.5), float2(0.5, -0.5), float2(-0.5,-0.5)};
    const float3 colors[3] = {float3(1,0,0), float3(0,1,0), float3(0,0,1)};
    const float2 uv = verts[vertexIdx % 3];
    const float4 p = float4(uv, 0.5 , 1);
    
    VertexStageOutput output = {
        .position = p, 
        .color = colors[vertexIdx % 3],
    };
    return output;
}

[[fragment]] float4 fragment_main(VertexStageOutput in [[stage_in]])
{
    return float4(in.color, 1.0);
}
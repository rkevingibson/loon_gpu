#include <metal_stdlib>
#include <metal_texture>

using namespace metal;

struct Vertex {
    float2 pos;
    float2 uv;
    uchar4 color;
};

struct VertexInput {
    float2 scale;
    float2 translate;
    float2 padding;
    constant Vertex* verts;
};

struct VertexStageOutput
{
    float4 position [[position]];
    float4 color;
    float2 uv;
};

struct FragInput {
    texture2d<float, access::sample> texture;
    sampler samp;
};

[[vertex]]
VertexStageOutput vertex_main(uint32_t vertexIdx [[vertex_id]], constant VertexInput* vert[[buffer(0)]])
{
    constant Vertex* vertices = vert->verts;
    Vertex vert = vertices[vertexIdx];
    VertexStageOutput out;
    out.position = float4(vert.pos * vert->scale + vert->translate, 0.5, 1);
    out.color = float4(vert.color) / 255.0;
    out.uv = vert.uv;
    return out;
}

[[fragment]]
float4 fragment_main(VertexStageOutput in[[stage_in]], constant FragInput* frag [[buffer(1)]])
{
    let sample = frag->texture.sample(frag->samp, in.uv);
    return in.color * sample;
}
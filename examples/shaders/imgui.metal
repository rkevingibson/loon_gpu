#import <metal_stdlib>
#import <metal_texture>

using namespace metal;

struct Vertex {
    packed_float2 pos;
    packed_float2 uv;
    packed_uchar4 color;
};

struct VertexInput {
    packed_float2 scale;
    packed_float2 translate;
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
    Vertex v = vertices[vertexIdx];
    VertexStageOutput out;
    out.position = float4(v.pos * vert->scale + vert->translate, 0.5, 1);
    out.color = float4(v.color) / 255.0;
    out.uv = v.uv;
    return out;
}

[[fragment]]
float4 fragment_main(VertexStageOutput in[[stage_in]], constant FragInput* frag [[buffer(1)]])
{
    float4 sample = frag->texture.sample(frag->samp, in.uv);
    return in.color * sample;
}
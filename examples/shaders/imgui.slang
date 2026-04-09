struct Vertex {
    float2 pos;
    float2 uv;
    uint8_t4 color;
};

struct VertexInput {
    float2 scale;
    float2 translate;
    Vertex* verts;
};

struct VertexStageOutput
{
    float4 position : SV_Position;
    float4 color;
    float2 uv;
};

struct FragInput {
    DescriptorHandle<Texture2D> texture;
    DescriptorHandle<SamplerState> sampler;
};

struct Args {
    VertexInput* vert;
    FragInput* frag;
};

[[vk::push_constant]] uniform Args args;

[shader("vertex")]
VertexStageOutput vertex_main(uint32_t vertexIdx : SV_VertexID)
{
    Vertex* vertices = args.vert->verts;
    Vertex vert = vertices[vertexIdx];
    VertexStageOutput out;
    out.position = float4(vert.pos * args.vert->scale + args.vert->translate, 0.5, 1);
    out.color = float4(vert.color) / 255.0;
    out.uv = vert.uv;
    return out;
}


[shader("fragment")]
float4 fragment_main(VertexStageOutput vertex) : SV_Target
{
    let sample = args.frag->texture.Sample(args.frag->sampler, vertex.uv);
    return vertex.color * sample;
}
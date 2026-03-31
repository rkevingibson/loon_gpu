struct CoarseVertex {
    float3 color;
};

struct VertexStageOutput
{
    float4 position : SV_Position;
    CoarseVertex coarse_vertex: VertexInterpolant;
};

[shader("vertex")]
VertexStageOutput vertex_main(uint32_t vertexIdx : SV_VertexID)
{
    const float2 verts[3] = {float2(0, 0.5), float2(0.5, -0.5), float2(-0.5,-0.5)};
    const float3 colors[3] = {float3(1,0,0), float3(0,1,0), float3(0,0,1)};
    const float2 uv = verts[vertexIdx % 3];
    const float4 p = float4(uv, 0.5 , 1);
    CoarseVertex vert_out;
    vert_out.color = colors[vertexIdx % 3];
    VertexStageOutput output = {p, vert_out};
    return output;
}

[shader("fragment")]
float4 fragment_main(CoarseVertex coarse_vertex : VertexInterpolant) : SV_Target
{
    float4 out_color = float4(coarse_vertex.color, 1.0);
    return out_color;
}
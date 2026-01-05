
struct CameraData {
    float3 translationWorldToCamera;
    float4 rotationWorldToCamera;

    float4x4 projection;
};

struct AssembledVertex
{
    float3 position : POSITION;
    float3 color : COLOR;
};

struct CoarseVertex {
    float3 color;
};

struct VertexStageOutput
{
    float4 position : SV_Position;
    CoarseVertex coarse_vertex: VertexInterpolant;
};

ParameterBlock<CameraData> cameraData;

[shader("vertex")]
VertexStageOutput vertexMain(AssembledVertex assembledVertex, uint32_t vertexIdx : SV_VertexID)
{
    VertexStageOutput output; 
    output.position = float4(assembledVertex.position - cameraData.translationWorldToCamera, 1.0);
    output.coarse_vertex.color = assembledVertex.color;
    return output;
}

[shader("fragment")]
float4 fragmentMain(CoarseVertex coarse_vertex : VertexInterpolant) : SV_Target
{
    float4 out_color = float4(coarse_vertex.color, 1.0);
    return out_color;
}
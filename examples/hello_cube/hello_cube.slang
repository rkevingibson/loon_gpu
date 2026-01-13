
struct CameraData {
    float4x4 projection;
    float4 rotationWorldToCamera;
    float3 translationWorldToCamera;
};

struct Mesh
{
    float3* position;
    float3* color;
};

struct InputData {
    CameraData camera;
    Mesh mesh;
};

struct VertexStageOutput
{
    float4 position : SV_Position;
    float3 color;
};

[shader("vertex")]
VertexStageOutput vertexMain(uniform InputData* args, uint32_t vertexIdx : SV_VertexID)
{
    VertexStageOutput output; 
    const Mesh mesh = args->mesh;
    output.position = mul(args->camera.projection, float4(mesh.position[vertexIdx] + float3(0,0,-3), 1.0));
    output.color = mesh.color[vertexIdx];
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput coarse_vertex ) : SV_Target
{
    float4 out_color = float4(coarse_vertex.color, 1.0);
    return out_color;
}
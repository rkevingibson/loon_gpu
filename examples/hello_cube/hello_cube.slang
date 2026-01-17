
struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct Mesh
{
    float3* position;
    float3* color;
    float4x4 worldFromMesh;
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

    float4x4 mvp = mul(args->camera.projection, mul(args->camera.cameraFromWorld, mesh.worldFromMesh));

    output.position = mul(mvp,float4(mesh.position[vertexIdx], 1.0));
    output.color = mesh.color[vertexIdx];
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput coarse_vertex ) : SV_Target
{
    float4 out_color = float4(coarse_vertex.color, 1.0);
    return out_color;
}
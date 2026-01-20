
struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct Mesh
{
    float3* position;
    float2* uvs;
    float4x4 worldFromMesh;
};

struct InputData {
    CameraData camera;
    Mesh mesh;
};

struct FragmentInput {
    DescriptorHandle<Sampler2D> texture;
};

struct VertexStageOutput
{
    float4 position : SV_Position;
    float2 uv;
};

[shader("vertex")]
VertexStageOutput vertexMain(uniform InputData* args, uint32_t vertexIdx : SV_VertexID)
{
    VertexStageOutput output; 
    const Mesh mesh = args->mesh;

    float4x4 mvp = mul(args->camera.projection, mul(args->camera.cameraFromWorld, mesh.worldFromMesh));

    output.position = mul(mvp,float4(mesh.position[vertexIdx], 1.0));
    output.uv = mesh.uvs[vertexIdx];
    return output;
}

[shader("fragment")]
float4 fragmentMain(uniform FragmentInput* args, VertexStageOutput vertex ) : SV_Target
{
    float4 out_color = args->texture.Sample(vertex.uv);
    return out_color;
}
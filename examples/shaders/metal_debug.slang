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

struct VertexStageOutput
{
    float4 position : SV_Position;
    float2 uv;
};

struct FragArgs {
    DescriptorHandle<Texture2D> texture;
    DescriptorHandle<SamplerState> sampler;
};

[shader("vertex")]
VertexStageOutput vertexMain(uint32_t vertexIdx : SV_VertexID, uniform InputData* vert)
{
    VertexStageOutput output; 
    const Mesh mesh = vert->mesh;

    float4x4 mvp = mul(vert->camera.projection, mul(vert->camera.cameraFromWorld, mesh.worldFromMesh));

    output.position = mul(mvp,float4(mesh.position[vertexIdx], 1.0));
    output.uv = mesh.uvs[vertexIdx];
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput vertex, uniform FragArgs* args) : SV_Target
{
    float4 out_color = args->texture.Sample(args->sampler, vertex.uv);
    return out_color;
}
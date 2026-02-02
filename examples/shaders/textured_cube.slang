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
    DescriptorHandle<Texture2D> texture;
    DescriptorHandle<SamplerState> sampler;
};

struct VertexStageOutput
{
    float4 position : SV_Position;
    float2 uv;
};

struct Args {
    InputData* vert;
    FragmentInput* frag;
};

[[vk::push_constant]] uniform Args args;

[shader("vertex")]
VertexStageOutput vertexMain(uint32_t vertexIdx : SV_VertexID)
{
    VertexStageOutput output; 
    const Mesh mesh = args.vert->mesh;

    float4x4 mvp = mul(args.vert->camera.projection, mul(args.vert->camera.cameraFromWorld, mesh.worldFromMesh));

    output.position = mul(mvp,float4(mesh.position[vertexIdx], 1.0));
    output.uv = mesh.uvs[vertexIdx];
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput vertex ) : SV_Target
{
    let sampler = args.frag->sampler;
    float4 out_color = args.frag->texture.Sample(sampler, vertex.uv);
    return out_color;
}
struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct Mesh {
    float3* positions;
    float2* uvs;
    float3* normals;
};

struct VertexInput {
    CameraData camera;
    Mesh mesh;
};

struct VertexStageOutput 
{
    float4 position : SV_Position;
    float2 uv;
};

struct FragmentInput {
};

struct Args {
    VertexInput* vert;
    FragmentInput* frag;
};

[[vk::push_constant]] uniform Args args;

[shader("vertex")]
VertexStageOutput vertex_main(uint32_t vertexIdx : SV_VertexID) 
{
    VertexStageOutput output;
    const Mesh = mesh = args.vert->mesh;
    float4x4 mvp = mul(args.vert->camera.projection, args.vert->camera.cameraFromWorld);

    output.position = mul(mvp, float4(mesh.position[vertexIdx], 1.0));
    output.uv = mesh.uvs[vertexIdx];
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexStageOutput vertex) : SV_Target
{
    return float4(1.0, 1.0, 0.0, 1.0);
}
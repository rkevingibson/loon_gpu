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
    float3 normal;
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
    const Mesh mesh = args.vert->mesh;
    float4x4 mvp = mul(args.vert->camera.projection, args.vert->camera.cameraFromWorld);

    output.position = mul(mvp, float4(mesh.positions[vertexIdx], 1.0));
    output.uv = mesh.uvs[vertexIdx];
    output.normal = mul(args.vert->camera.cameraFromWorld, float4(mesh.normals[vertexIdx], 1.0)).xyz;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexStageOutput vertex) : SV_Target
{
    return float4(0.5*vertex.normal + 0.5, 1.0);
}
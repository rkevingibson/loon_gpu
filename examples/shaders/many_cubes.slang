import loon;

struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct VertexInput
{
    float4x4 worldFromMesh;
    CameraData* camera;
    float3* position;
    float2* uvs;
    float* padding;
};

struct VertexStageOutput
{
    float4 position : SV_Position;
    float2 uv;
};

struct FragmentInput {
    DescriptorHandle<Texture2D> texture;
    DescriptorHandle<SamplerState> sampler;
};

struct Args {
    VertexInput* vert;
    FragmentInput* frag;
};

[[vk::push_constant]] uniform Args args;

[shader("vertex")]
VertexStageOutput vertex_main(uint32_t vertexIdx : SV_VertexID, uint32_t instanceIdx : SV_InstanceID)
{
    VertexStageOutput output; 
    CameraData* cam = args.vert[instanceIdx].camera;

    float4x4 mvp = mul(cam->projection, mul(cam->cameraFromWorld, args.vert[instanceIdx].worldFromMesh));

    output.position = mul(mvp,float4(args.vert[instanceIdx].position[vertexIdx], 1.0));
    output.uv = args.vert[instanceIdx].uvs[vertexIdx];
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexStageOutput vertex) : SV_Target
{
    float4 out_color = args.frag->texture.Sample(args.frag->sampler, vertex.uv);
    return out_color;
}
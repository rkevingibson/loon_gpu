struct CameraData {
    float4x4 projection;
    float4x4 camera_from_world;
};

struct Mesh {
    float4x4 world_from_mesh;
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
    float3 world_pos;
    float2 uv;
    float3 normal;
};

struct FragmentInput {
    DescriptorHandle<TextureCube> irradiance_map;
    DescriptorHandle<SamplerState> sampler;
    float3 camera_pos_world;
    float3 albedo;
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
    float4x4 camera_from_mesh = mul(args.vert->camera.camera_from_world, args.vert->mesh.world_from_mesh);
    float4x4 mvp = mul(args.vert->camera.projection, camera_from_mesh);


    output.position = mul(mvp, float4(mesh.positions[vertexIdx], 1.0));
    output.world_pos = mul(args.vert->mesh.world_from_mesh, float4(mesh.positions[vertexIdx], 1.0)).xyz;
    output.uv = mesh.uvs[vertexIdx];
    output.normal = mul(args.vert->mesh.world_from_mesh, float4(mesh.normals[vertexIdx], 1.0)).xyz;
    return output;
}

static float3 fresnel_schlick(float cos_theta, float3 F0) 
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

[shader("fragment")]
float4 fragment_main(VertexStageOutput vertex) : SV_Target
{
float3 normal = normalize(vertex.normal);
    float3 view_dir = normalize(args.frag->camera_pos_world - vertex.world_pos);
    const float3 F0 = float3(0.04, 0.04, 0.04);
    float3 kS = fresnel_schlick(max(dot(normal, view_dir), 0.0), F0);
    float3 kD = 1.0 - kS;
    float3 irradiance = float3(args.frag->irradiance_map.Sample(args.frag->sampler, normal).rgb);
    float3 diffuse = irradiance * args.frag->albedo;
    float3 ambient = (kD * diffuse);

    return float4(ambient, 1.0);
}
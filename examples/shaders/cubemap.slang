import loon;

struct EquirectangularToCubemapArgs {
    uint2 resolution;
    DescriptorHandle<Texture2D> input_texture;
    DescriptorHandle<SamplerState> sampler;
    DescriptorHandle<RWTexture2DArray<half4>> output_texture;
};

float2 sphere_pos_to_equirectangular_uv(float3 v)
{
    const float2 scale= float2(0.1591, 0.3183); // Convert from [-pi/2, pi/2] range to [-0.5,0.5] range.
    float2 uv = float2(atan2(v.z, v.x), asin(v.y));
    uv *= scale;
    uv += 0.5;
    return uv;
}

[shader("compute")]
[numthreads(8,8,1)]
void equirectangular_to_cubemap(uniform EquirectangularToCubemapArgs* args, uint3 gid: SV_DispatchThreadID)
{
    // Use face_idx & gid to compute a 3d position on the sphere.
    float2 face_pos = float2(gid.xy) / args->resolution - 0.5; // Map to (-0.5,0.5) on the face.
    float3 cube_pos[6] = {
        float3(0.5, face_pos.y, -face_pos.x),
        float3(-0.5, face_pos.y, face_pos.x),
        float3(face_pos.x, -0.5, face_pos.y),
        float3(face_pos.x, 0.5, -face_pos.y),
        float3(face_pos.x, face_pos.y, 0.5),
        float3(-face_pos.x, face_pos.y, -0.5),
    };

    for (uint16_t face_idx = 0; face_idx < 6; face_idx++)
    {
        // TODO: Better filtering here - we're not doing mipmaps but could directly filter samples from the map.
        float3 sphere_pos = normalize(cube_pos[face_idx]);
        float2 uv = sphere_pos_to_equirectangular_uv(sphere_pos);
        float4 color = args->input_texture.SampleLevel(args->sampler, uv, 0.0);
        args->output_texture.Store(uint3(gid.x, gid.y, face_idx), half4(half3(color.xyz), 1.0h));
    }
}

struct SkyboxArgs {
    float4x4 camera_from_world;
    DescriptorHandle<TextureCube> skybox;
    DescriptorHandle<SamplerState> sampler; 
};

struct SkyboxVertexOutput {
    float4 position : SV_Position;
    float3 world_space_dir; 
};

[[vk::push_constant]] uniform SkyboxArgs* skybox_args;

[shader("vertex")]
SkyboxVertexOutput skybox_vertex(uint32_t vertex_idx : SV_VertexID)
{
    // Draws a fullscreen triangle with verts (0,0), ()
    SkyboxVertexOutput output = {};
    const float2 ndc_pos = 2.0*float2((vertex_idx << 1) & 2, vertex_idx & 2) - 1.0f;

    output.position = float4(ndc_pos, 0.0f, 1.0f);
    output.world_space_dir = float3(ndc_pos, 1.0f); // TODO: Compute from camera matrix and vertex id.
    return output;
}

[shader("fragment")]
float4 skybox_fragment(SkyboxVertexOutput input) : SV_Target
{
    float4 out_color = skybox_args.skybox.Sample(skybox_args.sampler, input.world_space_dir);
    return out_color;
}
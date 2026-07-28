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

float3 sphere_pos_from_cubemap_pos(float2 uv, uint16_t face)
{
    float2 face_pos = uv - 0.5f;
    switch(face)
    {
        case 0: return normalize(float3(0.5, face_pos.y, -face_pos.x));
        case 1: return normalize(float3(-0.5, face_pos.y, face_pos.x));
        case 2: return normalize(float3(face_pos.x, -0.5, face_pos.y));
        case 3: return normalize(float3(face_pos.x, 0.5, -face_pos.y));
        case 4: return normalize(float3(face_pos.x, face_pos.y, 0.5));
        case 5: return normalize(float3(-face_pos.x, face_pos.y, -0.5));
    }
    return float3(0,0,0);
}

[shader("compute")]
[numthreads(8,8,1)]
void equirectangular_to_cubemap(uniform EquirectangularToCubemapArgs* args, uint3 gid: SV_DispatchThreadID)
{
    // Use face_idx & gid to compute a 3d position on the sphere.
    float2 face_uv = float2(gid.xy) / args->resolution;

    for (uint16_t face_idx = 0; face_idx < 6; face_idx++)
    {
        float3 cube_pos = sphere_pos_from_cubemap_pos(face_uv, face_idx);
        // TODO: Better filtering here - we're not doing mipmaps but could directly filter samples from the map.
        float3 sphere_pos = normalize(cube_pos);
        float2 uv = sphere_pos_to_equirectangular_uv(sphere_pos);
        float4 color = args->input_texture.SampleLevel(args->sampler, uv, 0.0);
        args->output_texture.Store(uint3(gid.x, gid.y, face_idx), half4(half3(color.xyz), 1.0h));
    }
}

// struct IrradianceCubemapFromCubemap
// {
//     uint2 resolution;
//     DescriptorHandle<TextureCube> input_texture;
//     DescriptorHandle<SamplerState> sampler;
//     DescriptorHandle<RWTexture2DArray<half4>> output_texture;
// };

// [shader("compute")]
// [numthreads(8,8,1)]
// void irradiance_cubemap_from_cubemap(uniform IrradianceCubemapFromCubemap* args, uint3 gid: SV_DispatchThreadID)
// {

// }

struct SkyboxArgs {
    float4x4 world_from_camera;
    float4x4 inverse_projection;
    float aspect_ratio; 

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
    // Coords are (-1,-1), (3, -1), (-1,3)
    const float2 ndc_pos = 2.0*float2((vertex_idx << 1) & 2, vertex_idx & 2) - 1.0f;
    // Unproject ndc position to world space - should multiply by fov ratio to keep things square
    const float3 camera_pos = float3(ndc_pos, 1.0f);

    output.position = float4(ndc_pos, 0.0f, 1.0f);
    output.world_space_dir = mul(mul(skybox_args.world_from_camera, skybox_args.inverse_projection),float4(camera_pos, 0.0f)).xyz; 
    return output;
}

[shader("fragment")]
float4 skybox_fragment(SkyboxVertexOutput input) : SV_Target
{
    float4 out_color = skybox_args.skybox.Sample(skybox_args.sampler, normalize(input.world_space_dir));
    return out_color;
}
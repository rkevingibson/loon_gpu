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
    static DescriptorHandle<SamplerState> sampler = {uint2(0, 0)};
    // Use face_idx & gid to compute a 3d position on the sphere.
    float2 face_pos = float2(gid.xy) / args->resolution - 0.5; // Map to (-0.5,0.5) on the face.
    float3 cube_pos[6] = {
        // TODO: work out the right orientations for these.
        float3(0.5, face_pos.x, face_pos.y),
        float3(-0.5, face_pos.x, face_pos.y),
        float3(face_pos.x, 0.5, face_pos.y),
        float3(face_pos.x, -0.5, face_pos.y),
        float3(face_pos.x, face_pos.y, 0.5),
        float3(face_pos.x, face_pos.y, -0.5),
    };

    for (uint16_t face_idx = 0; face_idx < 6; face_idx++)
    {
        float3 sphere_pos = normalize(cube_pos[face_idx]);
        float2 uv = sphere_pos_to_equirectangular_uv(sphere_pos);
        float4 color = args->input_texture.SampleLevel(sampler, uv, 0.0);
        args->output_texture.Store(uint3(gid.x, gid.y, face_idx), half4(color));
    }
}
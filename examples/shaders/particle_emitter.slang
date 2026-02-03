struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

enum CurveMode {
    Constant,
    Lerp,
    EaseIn,
    EaseOut,
    EaseInOut,
};

struct ParticleSimParam {
    float4 curve; // x = min start, y = max start, z = min end, z = max end
    CurveMode mode;
};

struct ParticleSimOptions {
    ParticleSimParam particle_lifetime;
    ParticleSimParam particle_size;
    float3 emitter_velocity;

    uint max_num_particles;
};

struct ParticleSim {
    ParticleSimOptions* options;
    float3* positions;
    float3* velocities;
    float3* colors;
};

struct VertexStageOutput {
};

[shader("compute")]
[numthreads(64, 1, 1)]
void update_particle_sim(
    uniform ParticleSim* particle_sim, 
    uint3 gid : SV_DispatchThreadID)
{
    const uint particle_id = gid.x;

}

[shader("vertex")]
VertexStageOutput vertex_main(uint32_t vertexIdx : SV_VertexID)
{
    return {};
}

[shader("fragment")]
float4 fragment_main(VertexStageOutput vertex) : SV_Target
{
    return float4(0,0,0,1);
}
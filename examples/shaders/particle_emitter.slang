struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct ParticleSim {
    uint num_particles;
    float3* positions;
    float3* velocities;
    float3* colors;
};

struct VertexStageOutput {
};

[shader("compute")]
void update_particle_sim(uniform ParticleSim* particle_sim)
{

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
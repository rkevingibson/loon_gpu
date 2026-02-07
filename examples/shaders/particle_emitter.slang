struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct DeadList {
    Atomic<int> size;
    int capacity;
    uint* indices;

    // Since we're either only ever decrementing (in the emitter stage) or incrementing (in the sim stage when particles age out), we can simplify a bit.
    [mutating]
    uint pop() {
        
        int index = size.decrement() - 1;
        if (index >= 0) {
            return indices[index];
        } else {
            // Probably a better way to do this, going to be a bunch of memory traffic.
            size = 0;
            return -1;
        }
    }

    [mutating]
    void push(uint i) {
        uint prev_size = size.increment();
        indices[prev_size] = i;
    }
};

// Adapted from https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
uint rand_pcg(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

enum CurveMode {
    Constant,
    Lerp,
    EaseIn,
    EaseOut,
    EaseInOut,
};


struct ParticleSimOptions {
    float lifetime;
    float spawn_pos;
    float spawn_radius;
    float spawn_size;
};

struct Particle {
    float3 position;
    float life;
    float3 velocity;
    float3 color;    
};

struct ParticleSim {
    ParticleSimOptions options;
    DeadList dead_list;
    uint max_num_particles;
    uint particles_to_emit;
    Particle* particles;    
};

struct VertexStageOutput {
};

    

[shader("compute")]
[numthreads(64, 1, 1)]
void emitter(uniform ParticleSim* sim, uint3 gid : SV_DispatchThreadID)
{
    if (gid.x >= sim->particles_to_emit) {
        return;
    }

    let particle_index = sim->dead_list.pop();

    ParticleSimOptions options = sim->options;
    // Spawn the particle based on 
    
}

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
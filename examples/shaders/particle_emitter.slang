struct DeadList {
    Atomic<int> size;
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

// Get uniform random number in 0-1 range
float uniform_rand(inout uint state)
{
    const uint x = rand_pcg(state);
    // This is imperfect but good enough - generate a float in [1, 2) using 23 bits from x, then subtract 1. 
    // This only covers 50% of the possible float values in [0,1), but that should be plenty. 
    // This feels like it should be fixable using some extra bits in x but eh.
    return asfloat((x & 0x7FFFF) | (0x7F800000)) - 1.0f;
}

// From PBRT 4E, Pharr, Jakob & Humphreys.
// https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#UniformlySamplingHemispheresandSpheres
float3 sample_uniform_sphere(inout uint rng_state)
{
    let u0 = uniform_rand(rng_state);
    let u1 = uniform_rand(rng_state);
    let z = 1.0f - 2.0f*u0;
    let r = sqrt(1.0f - z*z);
    const float kPi = 3.14159265359f;
    let phi = 2.0f * kPi * u1;
    return float3( r* cos(phi), r * sin(phi), z);
}

struct ParticleSimOptions {
    float3 spawn_pos;
    float  spawn_radius;
    float  lifetime;
    float  particle_size;
    float  delta_t;
    uint   max_num_particles;
    uint   particles_to_emit;
};

struct Particle {
    float3 position;
    float  life;
    float3 velocity;
    float  size;
    float4 color;    
};

struct ParticleSim {
    ParticleSimOptions options;
    DeadList           dead_list;
    Particle*          particles;    
};

[shader("compute")]
[numthreads(64,1,1)]
void reset_sim(uniform ParticleSim *sim, uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= sim->options.max_num_particles) {
        return;
    }

    if (gid.x == 0) {
        sim->dead_list.size = sim->options.max_num_particles;
    }

    sim->dead_list.indices[gid.x] = gid.x;
    sim->particles[gid.x].life = -1;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void emitter(uniform ParticleSim* sim, uint3 gid : SV_DispatchThreadID)
{
    if (gid.x >= sim->options.particles_to_emit) {
        return;
    }

    let particle_index = sim->dead_list.pop();

    ParticleSimOptions options = sim->options;
    // Spawn the particle based on random point in sphere.
    uint rng_state = rand_pcg(gid.x);

    float3 sphere_pos = sample_uniform_sphere(rng_state);
    const float3 initial_position = options.spawn_pos + sphere_pos * options.spawn_radius;

    const float3 initial_vel = sample_uniform_sphere(rng_state);
    // Could do better here.
    const float3 initial_color = float3(uniform_rand(rng_state), uniform_rand(rng_state), uniform_rand(rng_state));

    Particle particle = {};
    particle.position = initial_position;
    particle.life = options.lifetime;
    particle.velocity = initial_vel;
    particle.size = options.particle_size;
    particle.color = float4(initial_color, 1.0);
    sim->particles[particle_index] = particle;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_particle_sim(
    uniform ParticleSim* sim, 
    uint3 gid : SV_DispatchThreadID)
{
    const uint particle_id = gid.x;
    if (particle_id >= sim->options.max_num_particles) {
        return;
    }

    Particle p = sim->particles[particle_id];
    if (p.life < 0) { // Already dead particle, no need to sim it
        return;
    }

    float dt = sim->options.delta_t;
    p.life -= dt;
    if (p.life < 0) {
        sim->dead_list.push(particle_id);
    }

    
    p.position = p.position + dt * p.velocity;
    // TODO: Some sort of acceleration due to gravity.
    // Infinite posibilities - 


    // TODO: Could fill an indirect draw commmand here for more efficient rendering.
}


struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};


struct DrawSimArgs {
    CameraData camera;
    float3 camera_right_worldspace;
    float3 camera_up_worldspace;
    Particle* particles;
};

struct VertexStageOutput {
    float4 pos : SV_Position;
    float4 color;
    float2 uv;
};

[shader("vertex")]
VertexStageOutput vertex_main(uniform DrawSimArgs* args, uint32_t vertexIdx : SV_VertexID, uint32_t instance: SV_InstanceID)
{
    // Output a camera-facing quad
    // Initial quad is [-1, 1] in x, y, with normal as +z
    // 2 -- 3
    // |    |
    // 0 -- 1
    float2 quad_uv = float2(vertexIdx & 0x1, (vertexIdx >> 1) & 0x1);
    float2 quad_pos = 2.0f*quad_uv - 1.0f;

    Particle p = args.particles[instance];
    VertexStageOutput out;
    
    if (p.life < 0) {
        out.pos = float4(2,2,2,1); // Clip this quad, it's a dead particle.
    }
    else {
        const float3 vertex_world = p.position 
            + args->camera_right_worldspace * quad_pos.x * p.size 
            + args->camera_up_worldspace * quad_pos.y * p.size;
        out.pos = float4(vertex_world, 1.0f);
    }
    out.color = p.color;
    out.uv = quad_uv;
    return out;
}

[shader("fragment")]
float4 fragment_main(VertexStageOutput vertex) : SV_Target
{
    // TODO: Do a gaussian or something smoother than a solid quad.
    return vertex.color;
}
#include <metal_stdlib>
#include <metal_texture>
#include <metal_atomic>

using namespace metal;

struct DeadList {
    device int* indices;
};

// Since we're either only ever decrementing (in the emitter stage) or incrementing (in the sim stage when particles age out), we can simplify a bit.
int pop(device DeadList* list) {
    
    int index = atomic_fetch_sub_explicit(reinterpret_cast<device atomic_int*>(list->indices), 1, memory_order_relaxed);
    if (index > 0) {
        return list->indices[index];
    } else {
        // Probably a better way to do this, going to be a bunch of memory traffic.
        list->indices[0] = 0;
        return -1;
    }
}

void push(device DeadList* list, int i) {
    int prev_size = atomic_fetch_add_explicit(reinterpret_cast<device atomic_int*>(list->indices), 1, memory_order_relaxed);
    list->indices[prev_size + 1] = i;
}

// Adapted from https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
uint rand_pcg(thread uint& state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Get uniform random number in 0-1 range
float uniform_rand(thread uint& state)
{
    const uint x = rand_pcg(state);
    // This is imperfect but good enough - generate a float in [1, 2) using 23 bits from x, then subtract 1. 
    // This only covers 50% of the possible float values in [0,1), but that should be plenty. 
    // This feels like it should be fixable using some extra bits in x but eh.
    return as_type<float>((x & 0x7FFFFF) | (0x3F800000)) - 1.0f;
}

// From PBRT 4E, Pharr, Jakob & Humphreys.
// https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#UniformlySamplingHemispheresandSpheres
float3 sample_uniform_sphere(thread uint& rng_state)
{
    const float u0 = uniform_rand(rng_state);
    const float u1 = uniform_rand(rng_state);
    const float z = 1.0f - 2.0f*u0;
    const float r = sqrt(1.0f - z*z);
    const float kPi = 3.14159265359f;
    const float phi = 2.0f * kPi * u1;
    return float3(r* cos(phi), r * sin(phi), z);
}

struct ParticleSimOptions {
    float3 spawn_pos;
    float  spawn_radius;
    float  lifetime;
    float  particle_size;
    float  delta_t;
    uint   max_num_particles;
    uint   particles_to_emit;
    uint   rng_seed;
};

struct Particle {
    packed_float3 position;
    float  life;
    packed_float3 velocity;
    float  size;
    float4 color;    
};

struct DrawIndexedIndirectArgs {
    uint index_count;
    uint instance_count;
    uint first_index;
    int  vertex_offset;
    uint first_instance;
};

struct ParticleSim {
    ParticleSimOptions       options;
    DeadList                 dead_list;
    device Particle*                particles;
    device DrawIndexedIndirectArgs* draw_args;
    device uint*                    alive_particle_list;
};


[[kernel, required_threads_per_threadgroup(64,1,1)]]
void reset_sim(
    constant ParticleSim *sim [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]
    )
{
    if (gid.x >= sim->options.max_num_particles) {
        return;
    }

    if (gid.x == 0) {
        sim->dead_list.indices[0] = sim->options.max_num_particles;
    }

    sim->dead_list.indices[gid.x + 1] = gid.x;
    sim->particles[gid.x].life = -1;
}

[[kernel, required_threads_per_threadgroup(64, 1, 1)]]
void emitter(
    device ParticleSim* sim [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]
    )
{
    if (gid.x >= sim->options.particles_to_emit) {
        return;
    }

    int particle_index = pop(&sim->dead_list);
    if (particle_index == -1) {
        return;
    }

    ParticleSimOptions options = sim->options;
    // Spawn the particle based on random point in sphere.
    uint rng_state = gid.x + options.rng_seed;
    rand_pcg(rng_state);

    float3 sphere_pos = sample_uniform_sphere(rng_state);
    const float3 initial_position = options.spawn_pos + sphere_pos * options.spawn_radius;

    const float3 initial_vel = normalize(sphere_pos + sample_uniform_sphere(rng_state));
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

[[kernel, required_threads_per_threadgroup(64, 1, 1)]]
void update_particle_sim(
    device ParticleSim* sim [[buffer(0)]], 
    uint3 gid [[thread_position_in_grid]]
    )
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
        push(&sim->dead_list, particle_id);
    }

    p.position = p.position + dt * p.velocity;
    // TODO: Some sort of acceleration due to gravity.
    // Infinite posibilities - 

    sim->particles[particle_id] = p;
    // Add this particle to the indirect args. Also need to append it to the list of alive particles, for a re-ordering in the vertex shader - need a map from instance_id to particle_id. 
    uint instance_id = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&sim->draw_args->instance_count), 1, memory_order_relaxed);
    sim->alive_particle_list[instance_id] = particle_id;
}

struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct DrawSimArgs {
    CameraData camera;
    float3 camera_right_worldspace;
    float3 camera_up_worldspace;
    constant Particle* particles;
    constant uint* particle_ids;
};

struct VertexStageOutput {
    float4 pos [[position]];
    float4 color;
    float2 uv;
};

[[vertex]]
VertexStageOutput vertex_main(
    constant DrawSimArgs* args [[buffer(0)]],
    uint32_t vertexIdx [[vertex_id]],
    uint32_t instance [[instance_id]]
    )
{
    // Output a camera-facing quad
    // Initial quad is [-1, 1] in x, y, with normal as +z
    // 2 -- 3
    // |    |
    // 0 -- 1
    float2 quad_uv = float2(vertexIdx & 0x1, (vertexIdx >> 1) & 0x1);
    float2 quad_pos = 2.0f*quad_uv - 1.0f;

    uint particle_id = args->particle_ids[instance];
    Particle p = args->particles[particle_id];
    VertexStageOutput out;
    
    if (p.life < 0) {
        out.pos = float4(2,2,2,1); // Clip this quad, it's a dead particle.
    }
    else {
        const float3 vertex_world = p.position 
            + args->camera_right_worldspace * quad_pos.x * p.size 
            + args->camera_up_worldspace * quad_pos.y * p.size;

         float4x4 mvp = args->camera.projection * args->camera.cameraFromWorld;
        out.pos = mvp * float4(vertex_world, 1.0f);
    }
    out.color = p.color;
    out.uv = quad_uv;
    return out;
}

[[fragment]]
float4 fragment_main(VertexStageOutput vert [[stage_in]])
{
    // Simple circle for each quad.
    float alpha = clamp(1.f- 2.f*length(vert.uv - 0.5f), 0.f, 1.f);
    return float4(vert.color.xyz, alpha);
}
#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"
#include "common/geometry.h"
#include "common/gpu_args.h"


using namespace loon::gpu;

struct ParticleSimOptions {
    geometry::float3 spawn_pos;
    float            spawn_radius;
    float            lifetime;
    float            particle_size;
    float            delta_t;
    uint32_t         max_num_particles;
    uint32_t         particles_to_emit;
    uint32_t         rng_seed;
};

struct ParticleSim {
    Handle<Buffer> particle_buffer;
    Handle<Buffer> dead_list;
    Handle<Buffer> alive_list;

    ParticleSimOptions options;
};

class ParticleEmitter : public Example {
   public:
    ParticleEmitter(const WindowState& window_state);
    ~ParticleEmitter() override;

    void Update(const WindowState& window) override;

   private:
    void recreate_swapchain(uint32_t width, uint32_t height);

    Device        m_device;
    Handle<Queue> m_queue;
    Format        m_swapchain_format;
    uint32_t      m_swapchain_width  = 0;
    uint32_t      m_swapchain_height = 0;

    Handle<DepthStencilState> m_depth_stencil_state;
    Handle<Texture>           m_depth_texture{0};
    Handle<TextureHeap>       m_texture_heap;

    Handle<Pipeline> m_reset_sim_pipeline;
    Handle<Pipeline> m_emitter_pipeline;
    Handle<Pipeline> m_update_sim_pipeline;
    Handle<Pipeline> m_render_particle_pipeline;
    ParticleSim      m_sim;

    loon::RingBuffer m_ring_buffer;
    uint64_t         m_frame_idx;
};
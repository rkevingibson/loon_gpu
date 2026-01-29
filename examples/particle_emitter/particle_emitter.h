#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"

using namespace loon::gpu;

class ParticleEmitter : public Example {
   public:
    ParticleEmitter(const WindowState& window_state);
    ~ParticleEmitter() override;

    void Update(const WindowState& window) override;

   private:
    void recreate_swapchain(uint32_t width, uint32_t height);

    Device        m_device;
    Handle<Queue> m_queue;
    FORMAT        m_swapchain_format;
    uint32_t      m_swapchain_width  = 0;
    uint32_t      m_swapchain_height = 0;

    Handle<DepthStencilState> m_depth_stencil_state;
    Handle<Texture>           m_depth_texture{0};
    Handle<TextureView>       m_depth_view{0};


    Handle<Pipeline> m_update_sim_pipeline;
    Handle<Pipeline> m_render_particle_pipeline;
};
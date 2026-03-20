#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"
using namespace loon::gpu;

class TexturedCube : public Example {
   public:
    TexturedCube(const WindowState& window_state);
    ~TexturedCube() override;
    void Update(const WindowState& window) override;

   private:
    void recreate_swapchain(uint32_t width, uint32_t height);

    static constexpr uint64_t kMaxFramesInFlight = 3;
    Device                    m_device;
    Queue                     m_queue;
    Format                    m_swapchain_format;
    Handle<Pipeline>          m_render_pipeline;
    uint64_t                  m_frame_idx        = 0;
    uint32_t                  m_swapchain_width  = 0;
    uint32_t                  m_swapchain_height = 0;

    GpuPtr m_vertex_ptr;

    GpuPtr                    m_constant_buffer;
    Handle<Texture>           m_depth_texture;
    Handle<DepthStencilState> m_depth_stencil_state;

    Handle<TextureHeap> m_texture_heap;
    Handle<Texture>     m_color_texture;
    TextureView         m_color_view;
    Sampler             m_sampler;
};

#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"
using namespace loon::gpu;

class TexturedCube : public Example {
   public:
    TexturedCube(const WindowState& window_state);
    ~TexturedCube() override;
    bool update(const UpdateInfo& info) override;

   private:
    static constexpr uint64_t kMaxFramesInFlight = 3;
    Handle<Pipeline>          m_render_pipeline;
    uint64_t                  m_frame_idx = 0;

    GpuPtr m_vertex_ptr;

    GpuPtr                    m_constant_buffer;
    Handle<DepthStencilState> m_depth_stencil_state;

    Handle<TextureHeap> m_texture_heap;
    Handle<Texture>     m_color_texture;
    TextureView         m_color_view;
    Sampler             m_sampler;
};

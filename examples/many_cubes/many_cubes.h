#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"
#include "common/gpu_args.h"


using namespace loon::gpu;

class ManyCubes : public Example {
   public:
    ManyCubes(const WindowState& window_state);
    ~ManyCubes() override;

    bool update(const UpdateInfo& window) override;

   private:
    Handle<Pipeline>          m_render_pipeline;
    Handle<DepthStencilState> m_depth_stencil_state;
    Handle<Texture>           m_color_texture;
    Handle<TextureHeap>       m_texture_heap;

    GpuPtr      m_vertex_ptr;
    TextureView m_color_view;
    Sampler     m_sampler;

    loon::RingBuffer m_ring_buffer;
    uint64_t         m_frame_idx;

    static constexpr int kFrameTimeWindow                  = 300;
    int64_t              m_frame_time_us[kFrameTimeWindow] = {0};
    float                m_frame_time_ms[kFrameTimeWindow] = {0};
    int64_t              m_frame_time_average              = 0;

    int  m_grid_width          = 500;
    int  m_grid_height         = 500;
    bool m_use_instanced_draws = true;
};
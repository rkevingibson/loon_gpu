#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"
#include "common/gpu_args.h"


using namespace loon::gpu;

class ManyCubes : public Example {
   public:
    ManyCubes(const WindowState& window_state);
    ~ManyCubes() override;

    void Update(const WindowState& window) override;

   private:
    void recreate_swapchain(uint32_t width, uint32_t height);

    Device   m_device;
    Queue    m_queue;
    Format   m_swapchain_format;
    uint32_t m_swapchain_width  = 0;
    uint32_t m_swapchain_height = 0;

    Handle<Pipeline>          m_render_pipeline;
    Handle<DepthStencilState> m_depth_stencil_state;
    Handle<Texture>           m_depth_texture{0};
    Handle<Texture>           m_color_texture;
    Handle<TextureHeap>       m_texture_heap;

    GpuPtr      m_vertex_ptr;
    TextureView m_color_view;
    Sampler     m_sampler;

    loon::RingBuffer m_ring_buffer;
    uint64_t         m_frame_idx;

    static constexpr int kFrameTimeWindow     = 300;
    int64_t                 m_frame_time_us[kFrameTimeWindow] = {0};
    float                   m_frame_time_ms[kFrameTimeWindow] = {0};
    int64_t m_frame_time_average = 0;

    int m_grid_width  = 32;
    int m_grid_height = 32;
};
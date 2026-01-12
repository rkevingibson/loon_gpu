#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"
using namespace loon::gpu;

class HelloCube : public Example {
   public:
    HelloCube(const WindowState& window_state);
    ~HelloCube() override;
    void Update(const WindowState& window) override;

   private:
    Device            m_device;
    Handle<Queue>     m_queue;
    FORMAT            m_swapchain_format;
    Handle<Pipeline>  m_render_pipeline;
    Handle<Semaphore> m_semaphore;
    uint64_t          m_frame_idx        = 0;
    uint32_t          m_swapchain_width  = 0;
    uint32_t          m_swapchain_height = 0;

    Handle<Buffer> m_geometry_buffer;
    GpuPtr         m_vertex_ptr;

    Handle<Buffer> m_constant_buffer;
};

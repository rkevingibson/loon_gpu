#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"
using namespace loon::gpu;

class HelloTriangle : public Example {
   public:
    HelloTriangle(const WindowState& window_state);
    ~HelloTriangle() override;
    void Update(const WindowState& window) override;

   private:
    void              recreate_swapchain(uint32_t width, uint32_t height);
    Device            m_device;
    Handle<Queue>     m_queue;
    FORMAT            m_swapchain_format;
    Handle<Pipeline>  m_render_pipeline;
    Handle<Semaphore> m_semaphore;
    uint64_t          m_frame_idx        = 0;
    uint32_t          m_swapchain_width  = 0;
    uint32_t          m_swapchain_height = 0;
};

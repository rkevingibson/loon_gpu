#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"
using namespace loon::gpu;

class Bunny : public Example {
   public:
    Bunny(const WindowState& window_state);
    ~Bunny() override;
    void Update(const WindowState& window) override;

   private:
    void             recreate_swapchain(uint32_t width, uint32_t height);
    Device           m_device;
    Queue            m_queue;
    Format           m_swapchain_format;
    Handle<Pipeline> m_render_pipeline;
    uint32_t         m_swapchain_width  = 0;
    uint32_t         m_swapchain_height = 0;
};

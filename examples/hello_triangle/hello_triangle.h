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
    Device           m_device;
    Handle<Queue>    m_queue;
    FORMAT           m_swapchain_format;
    Handle<Pipeline> m_render_pipeline;
};

#pragma once

#include <webgpu/webgpu.h>

#include "common/example.h"


class HelloTriangle : public Example {
   public:
    HelloTriangle(const WindowState& window_state);
    ~HelloTriangle() override;
    void Update(const WindowState& window) override;

   private:
    WGPUInstance       m_instance;
    WGPUAdapter        m_adapter;
    WGPUSurface        m_surface;
    WGPUDevice         m_device;
    WGPUQueue          m_queue;
    WGPUTextureFormat  m_swapchain_format;
    WGPURenderPipeline m_render_pipeline;
};

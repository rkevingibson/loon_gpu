#pragma once

#include <webgpu/webgpu.h>

#include "common/example.h"


class HelloCube : public Example {
   public:
    HelloCube(const WindowState& window_state);
    ~HelloCube() override;
    void Update(const WindowState& window) override;

   private:
    WGPUInstance       m_instance;
    WGPUAdapter        m_adapter;
    WGPUSurface        m_surface;
    WGPUDevice         m_device;
    WGPUQueue          m_queue;
    WGPUTextureFormat  m_swapchain_format;
    WGPURenderPipeline m_render_pipeline;

    WGPUBuffer m_vertex_buffer;
    WGPUBuffer m_index_buffer;

    WGPUBuffer    m_camera_data_buffer;
    WGPUBindGroup m_bind_group;

    WGPUBuffer m_constant_buffer;
};

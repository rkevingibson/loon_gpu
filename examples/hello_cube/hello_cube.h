#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"
using namespace loon::gpu;

class HelloCube : public Example {
   public:
    HelloCube(const WindowState& window_state);
    ~HelloCube() override = default;

   private:
    bool Update(const UpdateInfo& info) override;

    Handle<Pipeline> m_render_pipeline;
    uint64_t         m_frame_idx = 0;

    GpuPtr m_vertex_ptr;

    GpuPtr                    m_constant_buffer;
    Handle<DepthStencilState> m_depth_stencil_state;
};

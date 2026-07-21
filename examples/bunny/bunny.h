#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"
#include "gpu_args.h"
using namespace loon::gpu;

struct GpuMesh {
    GpuPtr positions;
    GpuPtr uvs;
    GpuPtr normals;
};

class Bunny : public Example {
   public:
    Bunny(const WindowState& window_state);
    ~Bunny() override;
    bool update(const UpdateInfo& window) override;

   private:
    Handle<Pipeline>          m_render_pipeline;
    Handle<DepthStencilState> m_depth_stencil_state;
    loon::RingBuffer          m_ring_buffer;
    GpuMesh                   m_mesh;
    GpuPtr                    m_mesh_indices;
    uint32_t                  m_num_indices;
    size_t                    m_frame_idx = 0;

    Handle<TextureHeap> m_texture_heap;
    Handle<Texture>     m_hdri_cubemap;

    // Debugging data:
    TextureView m_debug_cubemap_face;
};

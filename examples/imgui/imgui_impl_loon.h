// dear imgui: Renderer Backend for Loon GPU

#pragma once
#include "imgui.h"  // IMGUI_IMPL_API
#ifndef IMGUI_DISABLE
#    include <gpu/loon_gpu.h>

#    include "shaders.h"



namespace loon::imgui {
struct InitInfo {
    gpu::Device*                  device;
    gpu::Handle<gpu::Queue>       queue;
    gpu::Handle<gpu::TextureHeap> texture_heap;
    int                           num_frames_in_flight;
    loon::gpu::Format             render_target_format;       // RenderTarget format.
    loon::gpu::Format             depth_stencil_view_format;  // DepthStencilView format.
    ShaderLoader*                 shader_loader;
};

bool Init(const InitInfo& info);
void Shutdown();

void NewFrame();
void Render(gpu::CommandBuffer cmd_buffer);
}  // namespace loon::imgui

#endif  // #ifndef IMGUI_DISABLE

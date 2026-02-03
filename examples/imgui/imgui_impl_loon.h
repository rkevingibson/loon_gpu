// dear imgui: Renderer Backend for Loon GPU

#pragma once
#include "imgui.h"  // IMGUI_IMPL_API
#ifndef IMGUI_DISABLE
#    include <gpu/loon_gpu.h>


// Initialization data, for ImGui_ImplDX12_Init()
struct ImGui_ImplLoon_InitInfo {
    loon::gpu::Device*                  device;
    loon::gpu::Handle<loon::gpu::Queue> queue;  // Command queue used for queuing texture uploads.
    int                                 NumFramesInFlight;
    loon::gpu::Format                   RTVFormat;  // RenderTarget format.
    loon::gpu::Format                   DSVFormat;  // DepthStencilView format.
    void*                               UserData;

    // Allocating SRV descriptors for textures is up to the application, so we provide callbacks.
    // (current version of the backend will only allocate one descriptor, from 1.92 the backend will
    // need to allocate more)
    loon::gpu::Handle<loon::gpu::TextureHeap> TextureHeap;
};

// Follow "Getting Started" link and check examples/ folder to learn about using backends!
IMGUI_IMPL_API bool ImGui_ImplLoon_Init(ImGui_ImplLoon_InitInfo* info);
IMGUI_IMPL_API void ImGui_ImplLoon_Shutdown();
IMGUI_IMPL_API void ImGui_ImplLoon_NewFrame();
IMGUI_IMPL_API void ImGui_ImplLoon_RenderDrawData(ImDrawData*              draw_data,
                                                  loon::gpu::CommandBuffer cmd_buffer);

// Use if you want to reset your rendering device without losing Dear ImGui state.
IMGUI_IMPL_API bool ImGui_ImplLoon_CreateDeviceObjects();
IMGUI_IMPL_API void ImGui_ImplLoon_InvalidateDeviceObjects();

// (Advanced) Use e.g. if you need to precisely control the timing of texture updates (e.g. for
// staged rendering), by setting ImDrawData::Textures = nullptr to handle this manually.
IMGUI_IMPL_API void ImGui_ImplLoon_UpdateTexture(ImTextureData* tex);

// [BETA] Selected render state data shared with callbacks.
// This is temporarily stored in GetPlatformIO().Renderer_RenderState during the
// ImGui_ImplDX12_RenderDrawData() call. (Please open an issue if you feel you need access to more
// data)
struct ImGui_ImplLoon_RenderState {
    loon::gpu::CommandBuffer CmdBuffer;
};

#endif  // #ifndef IMGUI_DISABLE

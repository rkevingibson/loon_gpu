// dear imgui: Renderer Backend for Loon GPU
// This needs to be used along with a Platform Backend (e.g. Win32)

#include "imgui.h"
#ifndef IMGUI_DISABLE
#    include <gpu/loon_gpu.h>

#    include "imgui_impl_loon.h"

using namespace loon;
// Loon data
struct ImGui_ImplLoon_RenderBuffers;

struct ImGui_ImplLoon_Texture {
    gpu::Handle<gpu::Texture>     texture;
    gpu::Handle<gpu::TextureView> texture_view;
    uint32_t                      tex_heap_idx;
    ImGui_ImplLoon_Texture() { memset((void*)this, 0, sizeof(*this)); }
};

struct ImGui_ImplLoon_Data {
    ImGui_ImplLoon_InitInfo    InitInfo;
    gpu::Device*               device;
    gpu::Handle<gpu::Pipeline> pipelineState;

    gpu::Handle<gpu::Queue>       queue;
    gpu::Format                   RTVFormat;
    gpu::Format                   DSVFormat;
    gpu::Handle<gpu::TextureHeap> texture_heap;
    // ID3D12Fence*                  Fence;
    // UINT64                        FenceLastSignaledValue;
    // HANDLE                        FenceEvent;
    uint32_t numFramesInFlight;

    // ID3D12GraphicsCommandList* pTexCmdList;
    // ID3D12Resource*            pTexUploadBuffer;
    uint32_t pTexUploadBufferSize;
    void*    pTexUploadBufferMapped;

    ImGui_ImplLoon_RenderBuffers* pFrameResources;
    uint64_t                      frameIndex;

    ImGui_ImplLoon_Data() { memset((void*)this, 0, sizeof(*this)); }
};

// Backend data stored in io.BackendRendererUserData to allow support for multiple Dear ImGui
// contexts It is STRONGLY preferred that you use docking branch with multi-viewports (== single
// Dear ImGui context + multiple windows) instead of multiple Dear ImGui contexts.
static ImGui_ImplLoon_Data* ImGui_ImplLoon_GetBackendData() {
    return ImGui::GetCurrentContext() ? (ImGui_ImplLoon_Data*)ImGui::GetIO().BackendRendererUserData
                                      : nullptr;
}

// Buffers used during the rendering of a frame
struct ImGui_ImplLoon_RenderBuffers {
    gpu::Handle<gpu::Buffer> IndexBuffer;
    gpu::Handle<gpu::Buffer> VertexBuffer;
    int                      IndexBufferSize;
    int                      VertexBufferSize;
};

struct VERTEX_CONSTANT_BUFFER_DX12 {
    float mvp[4][4];
};

static void ImGui_ImplLoon_SetupRenderState(ImDrawData*                   draw_data,
                                            gpu::CommandBuffer            command_list,
                                            ImGui_ImplLoon_RenderBuffers* fr) {
    // Setup orthographic projection matrix into our constant buffer
    // Our visible imgui space lies from draw_data->DisplayPos (top left) to
    // draw_data->DisplayPos+data_data->DisplaySize (bottom right).
    VERTEX_CONSTANT_BUFFER_DX12 vertex_constant_buffer;
    {
        float L         = draw_data->DisplayPos.x;
        float R         = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
        float T         = draw_data->DisplayPos.y;
        float B         = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
        float mvp[4][4] = {
            {2.0f / (R - L), 0.0f, 0.0f, 0.0f},
            {0.0f, 2.0f / (T - B), 0.0f, 0.0f},
            {0.0f, 0.0f, 0.5f, 0.0f},
            {(R + L) / (L - R), (T + B) / (B - T), 0.5f, 1.0f},
        };
        memcpy(&vertex_constant_buffer.mvp, mvp, sizeof(mvp));
    }
    ImGui_ImplLoon_Data* bd = ImGui_ImplLoon_GetBackendData();
    // Setup viewport


    // Bind shader
    command_list.set_pipeline(bd->pipelineState);
}

template <typename T>
static inline void SafeRelease(T*& res) {
    if (res) res->Release();
    res = nullptr;
}

// Render function
void ImGui_ImplLoon_RenderDrawData(ImDrawData* draw_data, gpu::CommandBuffer cmd) {
    // Avoid rendering when minimized
    if (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f) return;

    // Catch up with texture updates. Most of the times, the list will have 1 element with an OK
    // status, aka nothing to do. (This almost always points to ImGui::GetPlatformIO().Textures[]
    // but is part of ImDrawData to allow overriding or disabling texture updates).
    if (draw_data->Textures != nullptr)
        for (ImTextureData* tex : *draw_data->Textures)
            if (tex->Status != ImTextureStatus_OK) ImGui_ImplLoon_UpdateTexture(tex);

    // FIXME: We are assuming that this only gets called once per frame!
    ImGui_ImplLoon_Data* bd          = ImGui_ImplLoon_GetBackendData();
    bd->frameIndex                   = bd->frameIndex + 1;
    ImGui_ImplLoon_RenderBuffers* fr = &bd->pFrameResources[bd->frameIndex % bd->numFramesInFlight];

    // Create and grow vertex/index buffers if needed
    if (fr->VertexBuffer || fr->VertexBufferSize < draw_data->TotalVtxCount) {
        // TODO:
    }
    if (fr->IndexBuffer || fr->IndexBufferSize < draw_data->TotalIdxCount) {
        // TODO:
    }

    // Upload vertex/index data into a single contiguous GPU buffer
    // During Map() we specify a null read range (as per DX12 API, this is informational and for
    // tooling only)
    void *vtx_resource, *idx_resource;
    vtx_resource        = bd->device->get_host_pointer(fr->VertexBuffer);
    idx_resource        = bd->device->get_host_pointer(fr->IndexBuffer);
    ImDrawVert* vtx_dst = (ImDrawVert*)vtx_resource;
    ImDrawIdx*  idx_dst = (ImDrawIdx*)idx_resource;
    for (const ImDrawList* draw_list : draw_data->CmdLists) {
        memcpy(vtx_dst, draw_list->VtxBuffer.Data, draw_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idx_dst, draw_list->IdxBuffer.Data, draw_list->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtx_dst += draw_list->VtxBuffer.Size;
        idx_dst += draw_list->IdxBuffer.Size;
    }

    // Setup desired state
    ImGui_ImplLoon_SetupRenderState(draw_data, cmd, fr);

    // Setup render state structure (for callbacks and custom texture bindings)
    ImGuiPlatformIO&           platform_io = ImGui::GetPlatformIO();
    ImGui_ImplLoon_RenderState render_state;
    render_state.CmdBuffer           = cmd;
    platform_io.Renderer_RenderState = &render_state;

    // Render command lists
    // (Because we merged all buffers into a single one, we maintain our own offset into them)
    int    global_vtx_offset = 0;
    int    global_idx_offset = 0;
    ImVec2 clip_off          = draw_data->DisplayPos;
    ImVec2 clip_scale        = draw_data->FramebufferScale;
    for (const ImDrawList* draw_list : draw_data->CmdLists) {
        for (int cmd_i = 0; cmd_i < draw_list->CmdBuffer.Size; cmd_i++) {
            const ImDrawCmd* pcmd = &draw_list->CmdBuffer[cmd_i];
            if (pcmd->UserCallback != nullptr) {
                // User callback, registered via ImDrawList::AddCallback()
                // (ImDrawCallback_ResetRenderState is a special callback value used by the user to
                // request the renderer to reset render state.)
                if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
                    ImGui_ImplLoon_SetupRenderState(draw_data, cmd, fr);
                else
                    pcmd->UserCallback(draw_list, pcmd);
            } else {
                // Project scissor/clipping rectangles into framebuffer space
                ImVec2 clip_min((pcmd->ClipRect.x - clip_off.x) * clip_scale.x,
                                (pcmd->ClipRect.y - clip_off.y) * clip_scale.y);
                ImVec2 clip_max((pcmd->ClipRect.z - clip_off.x) * clip_scale.x,
                                (pcmd->ClipRect.w - clip_off.y) * clip_scale.y);
                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y) continue;

                // Apply scissor/clipping rectangle
                // TODO: Add scissor support to loon_gpu.
                // const D3D12_RECT r
                //     = {(LONG)clip_min.x, (LONG)clip_min.y, (LONG)clip_max.x, (LONG)clip_max.y};
                // command_list->RSSetScissorRects(1, &r);

                // Bind texture, Draw
                auto tex_id = (uint64_t)pcmd->GetTexID();

                // pcmd->IdxOffset + global_idx_offset;
                // pcmd->VtxOffset + global_vtx_offset;
                gpu::GpuPtr draw_args_gpu;
                gpu::GpuPtr index_buf;
                cmd.draw_indexed_instanced(draw_args_gpu,
                                           draw_args_gpu,
                                           index_buf,
                                           pcmd->ElemCount,
                                           1);
            }
        }
        global_idx_offset += draw_list->IdxBuffer.Size;
        global_vtx_offset += draw_list->VtxBuffer.Size;
    }
    platform_io.Renderer_RenderState = nullptr;
}

static void ImGui_ImplLoon_DestroyTexture(ImTextureData* tex) {
    if (ImGui_ImplLoon_Texture* backend_tex = (ImGui_ImplLoon_Texture*)tex->BackendUserData) {
        IM_ASSERT(backend_tex->tex_heap_idx == (uint32_t)tex->TexID);
        ImGui_ImplLoon_Data* bd = ImGui_ImplLoon_GetBackendData();

        // TODO: Free the texture and remove it from the heap.
        bd->device->remove_texture_view_from_heap(bd->texture_heap, backend_tex->tex_heap_idx);
        bd->device->free(backend_tex->texture_view);
        bd->device->free(backend_tex->texture);
        IM_DELETE(backend_tex);

        // Clear identifiers and mark as destroyed (in order to allow e.g. calling
        // InvalidateDeviceObjects while running)
        tex->SetTexID(ImTextureID_Invalid);
        tex->BackendUserData = nullptr;
    }
    tex->SetStatus(ImTextureStatus_Destroyed);
}

void ImGui_ImplLoon_UpdateTexture(ImTextureData* tex) {
    ImGui_ImplLoon_Data* bd = ImGui_ImplLoon_GetBackendData();
    bool                 need_barrier_before_copy
        = true;  // Do we need a resource barrier before we copy new data in?

    if (tex->Status == ImTextureStatus_WantCreate) {
        // Create and upload new texture to graphics system
        // IMGUI_DEBUG_LOG("UpdateTexture #%03d: WantCreate %dx%d\n", tex->UniqueID, tex->Width,
        // tex->Height);
        IM_ASSERT(tex->TexID == ImTextureID_Invalid && tex->BackendUserData == nullptr);
        IM_ASSERT(tex->Format == ImTextureFormat_RGBA32);
        ImGui_ImplLoon_Texture* backend_tex = IM_NEW(ImGui_ImplLoon_Texture)();

        // Create a texture, texture view and add it to the texture heap. Use the texture heap index
        // as the tex id.

        // Store identifiers
        tex->SetTexID((ImTextureID)backend_tex->tex_heap_idx);
        tex->BackendUserData = backend_tex;
        need_barrier_before_copy
            = false;  // Because this is a newly-created texture it will be in
                      // D3D12_RESOURCE_STATE_COMMON and thus we don't need a barrier
        // We don't set tex->Status to ImTextureStatus_OK to let the code fallthrough below.
    }

    if (tex->Status == ImTextureStatus_WantCreate || tex->Status == ImTextureStatus_WantUpdates) {
        ImGui_ImplLoon_Texture* backend_tex = (ImGui_ImplLoon_Texture*)tex->BackendUserData;
        IM_ASSERT(tex->Format == ImTextureFormat_RGBA32);

        // Use the staging buffer to upload the texture data to the GPU.
        // Check the need_barrier_before_copy to decide on synchronization, and insert a barrier
        // after.

        // For simplicity, use a semamphore to make this update a blocking function, not async.

        tex->SetStatus(ImTextureStatus_OK);
    }

    if (tex->Status == ImTextureStatus_WantDestroy
        && tex->UnusedFrames >= (int)bd->numFramesInFlight)
        ImGui_ImplLoon_DestroyTexture(tex);
}

bool ImGui_ImplLoon_CreateDeviceObjects() {
    ImGui_ImplLoon_Data* bd = ImGui_ImplLoon_GetBackendData();
    if (bd->pipelineState) ImGui_ImplLoon_InvalidateDeviceObjects();

    // Create pipelines

    // Create depth-stencil State

    // Create CPU-mapped ring buffer for vertex, index, and texture upload data.
    return true;
}

void ImGui_ImplLoon_InvalidateDeviceObjects() {
    ImGui_ImplLoon_Data* bd = ImGui_ImplLoon_GetBackendData();
    if (!bd) return;

    // Destroy GPU resources, pipelines, etc.


    // Destroy all textures
    for (ImTextureData* tex : ImGui::GetPlatformIO().Textures)
        if (tex->RefCount == 1) ImGui_ImplLoon_DestroyTexture(tex);

    for (uint32_t i = 0; i < bd->numFramesInFlight; i++) {
        ImGui_ImplLoon_RenderBuffers* fr = &bd->pFrameResources[i];
    }
}

bool ImGui_ImplLoon_Init(ImGui_ImplLoon_InitInfo* init_info) {
    ImGuiIO& io = ImGui::GetIO();
    IMGUI_CHECKVERSION();
    IM_ASSERT(io.BackendRendererUserData == nullptr && "Already initialized a renderer backend!");

    // Setup backend capabilities flags
    ImGui_ImplLoon_Data* bd = IM_NEW(ImGui_ImplLoon_Data)();
    bd->InitInfo            = *init_info;  // Deep copy
    init_info               = &bd->InitInfo;

    bd->device = init_info->device;
    IM_ASSERT(init_info->queue);
    bd->queue             = init_info->queue;
    bd->RTVFormat         = init_info->RTVFormat;
    bd->DSVFormat         = init_info->DSVFormat;
    bd->numFramesInFlight = init_info->NumFramesInFlight;
    bd->texture_heap      = init_info->TextureHeap;

    io.BackendRendererUserData = (void*)bd;
    io.BackendRendererName     = "imgui_impl_loon";
    io.BackendFlags
        |= ImGuiBackendFlags_RendererHasVtxOffset;  // We can honor the ImDrawCmd::VtxOffset field,
                                                    // allowing for large meshes.
    io.BackendFlags
        |= ImGuiBackendFlags_RendererHasTextures;  // We can honor ImGuiPlatformIO::Textures[]
                                                   // requests during render.

    // Create buffers with a default size (they will later be grown as needed)
    bd->frameIndex      = 0;
    bd->pFrameResources = new ImGui_ImplLoon_RenderBuffers[bd->numFramesInFlight];
    for (int i = 0; i < (int)bd->numFramesInFlight; i++) {
        // TODO: Initialize these buffers
        ImGui_ImplLoon_RenderBuffers* fr = &bd->pFrameResources[i];
        fr->IndexBuffer                  = {0};
        fr->VertexBuffer                 = {0};
        fr->IndexBufferSize              = 10000;
        fr->VertexBufferSize             = 5000;
    }

    return true;
}

void ImGui_ImplLoon_Shutdown() {
    ImGui_ImplLoon_Data* bd = ImGui_ImplLoon_GetBackendData();
    IM_ASSERT(bd != nullptr && "No renderer backend to shutdown, or already shutdown?");
    ImGuiIO&         io          = ImGui::GetIO();
    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();

    ImGui_ImplLoon_InvalidateDeviceObjects();
    delete[] bd->pFrameResources;

    io.BackendRendererName     = nullptr;
    io.BackendRendererUserData = nullptr;
    io.BackendFlags
        &= ~(ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures);
    platform_io.ClearRendererHandlers();
    IM_DELETE(bd);
}

void ImGui_ImplLoon_NewFrame() {
    ImGui_ImplLoon_Data* bd = ImGui_ImplLoon_GetBackendData();
    IM_ASSERT(bd != nullptr
              && "Context or backend not initialized! Did you call ImGui_ImplLoon_Init()?");

    if (!bd->pipelineState)
        if (!ImGui_ImplLoon_CreateDeviceObjects())
            IM_ASSERT(0 && "ImGui_ImplLoon_CreateDeviceObjects() failed!");
}

//-----------------------------------------------------------------------------

#endif  // #ifndef IMGUI_DISABLE

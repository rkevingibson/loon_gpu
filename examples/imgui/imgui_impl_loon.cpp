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
    gpu::Handle<gpu::Texture> texture;
    uint32_t                  tex_heap_idx;
    ImGui_ImplLoon_Texture() { memset((void*)this, 0, sizeof(*this)); }
};

struct ImGui_ImplLoon_Data {
    gpu::Device                device;
    gpu::Handle<gpu::Pipeline> pipelineState;

    gpu::Queue                          queue;
    gpu::Format                         color_format;
    gpu::Format                         depth_format;
    gpu::Handle<gpu::TextureHeap>       texture_heap;
    gpu::Handle<gpu::DepthStencilState> depth_stencil_state;
    gpu::Sampler                        sampler;
    uint32_t                            num_frames_in_flight;
    ShaderLoader*                       shader_loader;

    ImGui_ImplLoon_RenderBuffers* pFrameResources;
    uint64_t                      frameIndex;

    ImGui_ImplLoon_Data() { memset((void*)this, 0, sizeof(*this)); }
};

// Backend data stored in io.BackendRendererUserData to allow support for
// multiple Dear ImGui contexts It is STRONGLY preferred that you use docking
// branch with multi-viewports (== single Dear ImGui context + multiple windows)
// instead of multiple Dear ImGui contexts.
static ImGui_ImplLoon_Data* GetBackendData() {
    return ImGui::GetCurrentContext() ? (ImGui_ImplLoon_Data*)ImGui::GetIO().BackendRendererUserData
                                      : nullptr;
}

// Buffers used during the rendering of a frame
struct ImGui_ImplLoon_RenderBuffers {
    gpu::GpuPtr buffer;
    size_t      buffer_size;
};

static void SetupRenderState(ImDrawData*                   draw_data,
                             gpu::CommandBuffer            command_list,
                             ImGui_ImplLoon_RenderBuffers* fr) {
    // Setup transforms

    ImGui_ImplLoon_Data* bd = GetBackendData();

    // Bind shader
    gpu::cmd_set_depth_stencil_state(command_list, bd->depth_stencil_state);
    gpu::cmd_set_pipeline(command_list, bd->pipelineState);
}

template <typename T>
static inline void SafeRelease(T*& res) {
    if (res) res->Release();
    res = nullptr;
}

static void DestroyTexture(ImTextureData* tex) {
    if (ImGui_ImplLoon_Texture* backend_tex = (ImGui_ImplLoon_Texture*)tex->BackendUserData) {
        IM_ASSERT(backend_tex->tex_heap_idx == (uint32_t)tex->TexID - 1);
        ImGui_ImplLoon_Data* bd = GetBackendData();

        // Free the texture and remove it from the heap.
        gpu::remove_texture_view_from_heap(bd->device, bd->texture_heap, backend_tex->tex_heap_idx);
        gpu::free(bd->device, backend_tex->texture);
        IM_DELETE(backend_tex);

        // Clear identifiers and mark as destroyed (in order to allow e.g. calling
        // InvalidateDeviceObjects while running)
        tex->SetTexID(ImTextureID_Invalid);
        tex->BackendUserData = nullptr;
    }
    tex->SetStatus(ImTextureStatus_Destroyed);
}

static void UpdateTexture(ImTextureData* tex, ImGui_ImplLoon_RenderBuffers* fb) {
    ImGui_ImplLoon_Data* bd = GetBackendData();
    bool                 need_barrier_before_copy
        = false;  // Do we need a resource barrier before we copy new data in?

    if (tex->Status == ImTextureStatus_WantCreate) {
        // Create and upload new texture to graphics system
        // IMGUI_DEBUG_LOG("UpdateTexture #%03d: WantCreate %dx%d\n", tex->UniqueID,
        // tex->Width, tex->Height);
        IM_ASSERT(tex->TexID == ImTextureID_Invalid && tex->BackendUserData == nullptr);
        IM_ASSERT(tex->Format == ImTextureFormat_RGBA32);
        ImGui_ImplLoon_Texture* backend_tex = IM_NEW(ImGui_ImplLoon_Texture)();

        // Create a texture, texture view and add it to the texture heap. Use the
        // texture heap index as the tex id.
        backend_tex->texture = gpu::create_texture(
            bd->device,
            {
                .dimensions = {.x = static_cast<uint32_t>(tex->Width),
                               .y = static_cast<uint32_t>(tex->Height),
                               .z = 1},
                .format     = loon::gpu::Format::RGBA8Unorm,
                .usage      = loon::gpu::UsageFlags::Sampled | loon::gpu::UsageFlags::TransferDst,
            });

        backend_tex->tex_heap_idx
            = gpu::add_texture_view_to_heap(bd->device,
                                            bd->texture_heap,
                                            {
                                                .texture = backend_tex->texture,
                                                .format  = loon::gpu::Format::RGBA8Unorm,
                                            });
<<<<<<< HEAD
=======

>>>>>>> 0eaa7ce (Switching the WIP metal backend over to the C-api. Need to fix the vulkan impl.)
        // Store identifiers
        // Because invalid tex id == 0, we add one here and subtract on retrieval.
        tex->SetTexID((ImTextureID)backend_tex->tex_heap_idx + 1);
        tex->BackendUserData     = backend_tex;
        need_barrier_before_copy = true;
    }

    if (tex->Status == ImTextureStatus_WantCreate || tex->Status == ImTextureStatus_WantUpdates) {
        ImGui_ImplLoon_Texture* backend_tex = (ImGui_ImplLoon_Texture*)tex->BackendUserData;
        IM_ASSERT(tex->Format == ImTextureFormat_RGBA32);

        // Use the staging buffer to upload the texture data to the GPU.
        // Check the need_barrier_before_copy to decide on synchronization, and
        // insert a barrier after.

        // For simplicity, use a semamphore to make this update a blocking function,
        // not async.
        if (fb->buffer_size < tex->GetSizeInBytes()) {
            if (fb->buffer) gpu::free(bd->device, fb->buffer);
            fb->buffer      = gpu::malloc(bd->device, tex->GetSizeInBytes());
            fb->buffer_size = tex->GetSizeInBytes();
        }

        void* dst = gpu::get_host_pointer(bd->device, fb->buffer);
        memcpy(dst, tex->GetPixels(), tex->GetSizeInBytes());

        auto cmd = gpu::queue_start_command_recording(bd->queue);

        if (need_barrier_before_copy) {
            gpu::cmd_barrier(cmd,
                             gpu::StageFlags::None,
                             gpu::StageFlags::Transfer,
                             gpu::TextureTransition{
                                 .texture    = backend_tex->texture,
                                 .old_layout = loon::gpu::Layout::DontCare,
                                 .new_layout = loon::gpu::Layout::General,
                             });
        }

        gpu::cmd_copy_to_texture(
            cmd,
            fb->buffer,
            backend_tex->texture,
            gpu::BufferToTextureCopyInfo{
                .buffer_image_size
                = {static_cast<uint32_t>(tex->Width), static_cast<uint32_t>(tex->Height)},
                .image_extent
                = {static_cast<uint32_t>(tex->Width), static_cast<uint32_t>(tex->Height), 1},
            });
        gpu::cmd_barrier(cmd, gpu::StageFlags::Transfer, gpu::StageFlags::PixelShader);
        gpu::cmd_finalize(cmd);
        auto copy_semaphore = gpu::create_semaphore(bd->device, 0);

        gpu::queue_submit(bd->queue,
                          cmd,
                          {},
                          gpu::SemaphoreInfo{
                              .semaphore = copy_semaphore,
                              .value     = 1,
                              .stage     = gpu::StageFlags::Transfer,
                          });
        gpu::wait_semaphore(bd->device, copy_semaphore, 1);
        gpu::free(bd->device, copy_semaphore);

        tex->SetStatus(ImTextureStatus_OK);
    }

    if (tex->Status == ImTextureStatus_WantDestroy
        && tex->UnusedFrames >= (int)bd->num_frames_in_flight)
        DestroyTexture(tex);
}

static void InvalidateDeviceObjects();

static bool CreateDeviceObjects() {
    ImGui_ImplLoon_Data* bd = GetBackendData();
    if (bd->pipelineState) InvalidateDeviceObjects();

    // Create pipelines
    using namespace gpu;
    ShaderModule shader         = bd->shader_loader->load_module("imgui.slang");
    const auto   vertex_spirv   = get_spirv(shader.get(), "vertex_main");
    const auto   fragment_spirv = get_spirv(shader.get(), "fragment_main");
    bd->pipelineState = gpu::create_graphics_pipeline(bd->device,
      {
          .spirv = Span(vertex_spirv.data(), vertex_spirv.size()).as_bytes(),
          .entry_point = "vertex_main"_sv,
      },
      {
          .spirv =
              Span(fragment_spirv.data(), fragment_spirv.size()).as_bytes(),
          .entry_point = "fragment_main"_sv,
      },
      RasterDesc{
          .cull = Cull::None,
          .depth_format = bd->depth_format,
          .color_targets = ColorTarget{
            .format = bd->color_format,
            .blendstate =
              BlendDesc{
                  .color_op = Blend::Add,
                  .src_color_factor = Factor::SrcAlpha,
                  .dst_color_factor = Factor::OneMinusSrcAlpha,
                  .src_alpha_factor = Factor::One,
                  .dst_alpha_factor = Factor::OneMinusSrcAlpha,
              },
        },
      });

    bd->sampler = gpu::add_sampler_to_heap(bd->device, bd->texture_heap, SamplerDesc{});

    // Create depth-stencil State
    bd->depth_stencil_state
        = gpu::create_depth_stencil_state(bd->device,
                                          DepthStencilDesc{
                                              .depth_mode = loon::gpu::DepthFlags::None,
                                              .depth_test = loon::gpu::Op::Always,
                                          });

    return true;
}

void InvalidateDeviceObjects() {
    ImGui_ImplLoon_Data* bd = GetBackendData();
    if (!bd) return;

    // Destroy GPU resources, pipelines, etc.
    gpu::free(bd->device, bd->pipelineState);
    bd->pipelineState.h = 0;
    gpu::free_depth_stencil_state(bd->device, bd->depth_stencil_state);
    bd->depth_stencil_state.h = 0;
    gpu::remove_sampler_from_heap(bd->device, bd->texture_heap, bd->sampler);

    // Destroy all textures
    for (ImTextureData* tex : ImGui::GetPlatformIO().Textures)
        if (tex->RefCount == 1) DestroyTexture(tex);

    for (uint32_t i = 0; i < bd->num_frames_in_flight; i++) {
        ImGui_ImplLoon_RenderBuffers* fr = &bd->pFrameResources[i];
        gpu::free(bd->device, fr->buffer);
    }
}

namespace loon::imgui {

bool Init(const InitInfo& info) {
    ImGuiIO& io = ImGui::GetIO();
    IMGUI_CHECKVERSION();
    IM_ASSERT(io.BackendRendererUserData == nullptr && "Already initialized a renderer backend!");

    // Setup backend capabilities flags
    ImGui_ImplLoon_Data* bd = IM_NEW(ImGui_ImplLoon_Data)();

    bd->device               = info.device;
    bd->queue                = info.queue;
    bd->color_format         = info.render_target_format;
    bd->depth_format         = info.depth_stencil_view_format;
    bd->num_frames_in_flight = info.num_frames_in_flight;
    bd->texture_heap         = info.texture_heap;
    bd->shader_loader        = info.shader_loader;

    io.BackendRendererUserData = (void*)bd;
    io.BackendRendererName     = "imgui_impl_loon";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;  // We can honor the
                                                                // ImDrawCmd::VtxOffset field,
                                                                // allowing for large meshes.
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;   // We can honor
                                                                // ImGuiPlatformIO::Textures[]
                                                                // requests during render.

    // Create buffers with a default size (they will later be grown as needed)
    bd->frameIndex      = 0;
    bd->pFrameResources = new ImGui_ImplLoon_RenderBuffers[bd->num_frames_in_flight];
    for (int i = 0; i < (int)bd->num_frames_in_flight; i++) {
        ImGui_ImplLoon_RenderBuffers* fr = &bd->pFrameResources[i];
        fr->buffer                       = {0};
        fr->buffer_size                  = 0;
    }

    return true;
}

void Shutdown() {
    ImGui_ImplLoon_Data* bd = GetBackendData();
    IM_ASSERT(bd != nullptr && "No renderer backend to shutdown, or already shutdown?");
    ImGuiIO&         io          = ImGui::GetIO();
    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();

    InvalidateDeviceObjects();
    delete[] bd->pFrameResources;

    io.BackendRendererName     = nullptr;
    io.BackendRendererUserData = nullptr;
    io.BackendFlags
        &= ~(ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures);
    platform_io.ClearRendererHandlers();
    IM_DELETE(bd);
}

void NewFrame() {
    ImGui_ImplLoon_Data* bd = GetBackendData();
    IM_ASSERT(bd != nullptr && "Context or backend not initialized! Did you call "
                             "ImGui_ImplLoon_Init()?");

    if (!bd->pipelineState)
        if (!CreateDeviceObjects()) IM_ASSERT(0 && "ImGui_ImplLoon_CreateDeviceObjects() failed!");

    ImGui::NewFrame();
}

struct alignas(64) VertexInput {
    ImVec2      scale;
    ImVec2      translate;
    ImVec2      padding;
    gpu::GpuPtr vertex_buffer;
};

struct alignas(64) FragmentInput {
    gpu::TextureView texture;
    gpu::Sampler     sampler;
};

struct DrawArgs {
    VertexInput   vert;
    FragmentInput frag;
};

static gpu::GpuPtr AlignAddress(gpu::GpuPtr size, gpu::GpuPtr alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

void Render(gpu::CommandBuffer cmd) {
    ImGui::Render();

    auto draw_data = ImGui::GetDrawData();

    // Avoid rendering when minimized
    if (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f) return;

    ImGui_ImplLoon_Data* bd = GetBackendData();
    bd->frameIndex          = bd->frameIndex + 1;
    ImGui_ImplLoon_RenderBuffers* fr
        = &bd->pFrameResources[bd->frameIndex % bd->num_frames_in_flight];

    // Catch up with texture updates. Most of the times, the list will have 1
    // element with an OK status, aka nothing to do. (This almost always points to
    // ImGui::GetPlatformIO().Textures[] but is part of ImDrawData to allow
    // overriding or disabling texture updates).
    if (draw_data->Textures != nullptr)
        for (ImTextureData* tex : *draw_data->Textures)
            if (tex->Status != ImTextureStatus_OK) UpdateTexture(tex, fr);

    // FIXME: We are assuming that this only gets called once per frame!

    // Create and grow buffers if needed
    const size_t vertex_data_size = AlignAddress(draw_data->TotalVtxCount * sizeof(ImDrawVert), 64);
    const size_t index_data_size  = AlignAddress(draw_data->TotalIdxCount * sizeof(ImDrawIdx), 64);

    size_t num_draws = 0;
    for (const ImDrawList* draw_list : draw_data->CmdLists) {
        num_draws += draw_list->CmdBuffer.size();
    }
    const size_t args_size            = AlignAddress(sizeof(DrawArgs) * num_draws, 64);
    const size_t required_buffer_size = vertex_data_size + index_data_size + args_size;

    if (required_buffer_size == 0) { return; }

    if (!fr->buffer || fr->buffer_size < required_buffer_size) {
        // Round up to some nice multiple to avoid reallocs frequently.
        const size_t buffer_size = ((required_buffer_size + 1023) / 1024) * 1024;

        if (fr->buffer) { gpu::free(bd->device, fr->buffer); }
        fr->buffer      = gpu::malloc(bd->device, required_buffer_size, gpu::Memory::Default);
        fr->buffer_size = required_buffer_size;
    }

    char*       buffer_host = (char*)gpu::get_host_pointer(bd->device, fr->buffer);
    ImDrawVert* vtx_dst     = (ImDrawVert*)buffer_host;
    ImDrawIdx*  idx_dst     = (ImDrawIdx*)(buffer_host + vertex_data_size);
    for (const ImDrawList* draw_list : draw_data->CmdLists) {
        memcpy(vtx_dst, draw_list->VtxBuffer.Data, draw_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idx_dst, draw_list->IdxBuffer.Data, draw_list->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtx_dst += draw_list->VtxBuffer.Size;
        idx_dst += draw_list->IdxBuffer.Size;
    }

    DrawArgs* draw_args = (DrawArgs*)(buffer_host + vertex_data_size + index_data_size);

    // Setup desired state

    SetupRenderState(draw_data, cmd, fr);

    // Setup render state structure (for callbacks and custom texture bindings)
    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
    // TODO: Support UserCallback render state if we want to.
    // ImGui_ImplLoon_RenderState render_state;
    // render_state.CmdBuffer           = cmd;
    // platform_io.Renderer_RenderState = &render_state;

    // Render command lists
    // (Because we merged all buffers into a single one, we maintain our own
    // offset into them)
    gpu::GpuPtr global_vtx_ptr = fr->buffer;
    gpu::GpuPtr global_idx_ptr = global_vtx_ptr + vertex_data_size;
    gpu::GpuPtr args_ptr       = global_idx_ptr + index_data_size;

    ImVec2 scale;
    scale[0] = 2.0f / draw_data->DisplaySize.x;
    scale[1] = 2.0f / draw_data->DisplaySize.y;
    ImVec2 translate;
    translate[0] = -1.0f - draw_data->DisplayPos.x * scale[0];
    translate[1] = -1.0f - draw_data->DisplayPos.y * scale[1];

    ImVec2 clip_off   = draw_data->DisplayPos;
    ImVec2 clip_scale = draw_data->FramebufferScale;

    for (const ImDrawList* draw_list : draw_data->CmdLists) {
        for (int cmd_i = 0; cmd_i < draw_list->CmdBuffer.Size; cmd_i++) {
            const ImDrawCmd* pcmd = &draw_list->CmdBuffer[cmd_i];
            if (pcmd->UserCallback != nullptr) {
                // User callback, registered via ImDrawList::AddCallback()
                // (ImDrawCallback_ResetRenderState is a special callback value used by
                // the user to request the renderer to reset render state.)
                if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
                    SetupRenderState(draw_data, cmd, fr);
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
                gpu::cmd_set_scissor_rect(cmd,
                                          {
                                              .offset_x = (uint32_t)clip_min.x,
                                              .offset_y = (uint32_t)clip_min.y,
                                              .width    = (uint32_t)(clip_max.x - clip_min.x),
                                              .height   = (uint32_t)(clip_max.y - clip_min.y),
                                          });

                // Bind texture, Draw
                auto        tex_id     = (uint64_t)pcmd->GetTexID() - 1;
                gpu::GpuPtr vertex_buf = global_vtx_ptr + (pcmd->VtxOffset * sizeof(ImDrawVert));
                gpu::GpuPtr index_buf  = global_idx_ptr + (pcmd->IdxOffset * sizeof(ImDrawIdx));

                *draw_args = DrawArgs{.vert{
                                  .scale         = scale,
                                  .translate     = translate,
                                  .padding       = {0, 0},
                                  .vertex_buffer = vertex_buf,
                              },
                              .frag = {
                                  .texture = tex_id,
                                  .sampler = bd->sampler,
                              },};

                gpu::cmd_draw_indexed_instanced(
                    cmd,
                    {
                        .vertexDataGpu   = args_ptr,
                        .fragmentDataGpu = args_ptr + offsetof(DrawArgs, frag),
                        .indicesGpu      = index_buf,
                        .indexCount      = pcmd->ElemCount,
                    });

                args_ptr += sizeof(DrawArgs);
                draw_args++;
            }
        }
        global_idx_ptr += draw_list->IdxBuffer.Size * sizeof(ImDrawIdx);
        global_vtx_ptr += draw_list->VtxBuffer.Size * sizeof(ImDrawVert);
    }
    platform_io.Renderer_RenderState = nullptr;
}

//-----------------------------------------------------------------------------

}  // namespace loon::imgui
#endif  // #ifndef IMGUI_DISABLE

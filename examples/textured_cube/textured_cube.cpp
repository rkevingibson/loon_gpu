/***
    Hello Triangle - A basic WGPU introduction

    This example shows setup, teardown, and basic rendering of a triangle.
    It shows everything in this file explicitly as a reference - other examples will use helpers for
   initialization.

*/

#include "textured_cube.h"

#include <gpu/loon_gpu.h>

#include <cassert>
#include <cstring>

#include "common/geometry.h"
#include "common/shaders.h"
#include "stb_image.h"


using namespace loon;
using namespace geometry;
namespace {
struct Cube {
    static constexpr float3 kPositions[] = {
        {-1, -1, -1}, {1, -1, -1}, {-1, 1, -1},  {1, 1, -1},  {-1, 1, -1}, {1, 1, -1},
        {-1, 1, 1},   {1, 1, 1},   {-1, -1, 1},  {1, -1, 1},  {-1, 1, 1},  {1, 1, 1},
        {-1, -1, -1}, {1, -1, -1}, {-1, -1, 1},  {1, -1, 1},  {1, -1, 1},  {1, -1, -1},
        {1, 1, 1},    {1, 1, -1},  {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
    };

    static constexpr float2 kUVs[] = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1},
        {0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 0}, {1, 0}, {0, 1}, {1, 1},
        {0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 0}, {1, 0}, {0, 1}, {1, 1},
    };
    static constexpr uint16_t kIndices[]
        = {0,  3,  1,  0,  2,  3,  4,  6,  7,  4,  7,  5,  8,  9,  11, 8,  11, 10,
           12, 13, 15, 12, 15, 14, 16, 17, 18, 18, 17, 19, 20, 21, 22, 22, 21, 23};
    static constexpr uint32_t kNumIndices = sizeof(kIndices) / sizeof(kIndices[0]);
    static constexpr size_t   kSize       = sizeof(kPositions) + sizeof(kUVs) + sizeof(kIndices);

    static void* write(void* ptr) {
        char* dst = (char*)ptr;
        memcpy(ptr, kPositions, sizeof(kPositions));
        dst += sizeof(kPositions);
        memcpy(dst, kUVs, sizeof(kUVs));
        dst += sizeof(kUVs);
        memcpy(dst, kIndices, sizeof(kIndices));
        return dst + sizeof(kIndices);
    }
};

}  // namespace
struct MeshGpu {
    GpuPtr   position;
    GpuPtr   color;
    float4x4 world_from_mesh;
};

struct CameraInfo {
    float4x4 projection        = {};
    float4x4 camera_from_world = {};
};

struct ShaderArgs {
    CameraInfo  camera;
    MeshGpu     mesh;
    TextureView texture;
    Sampler     sampler;
};

static Format select_surface_format(const loon::gpu::SurfaceCapabilities& surface_capabilities) {
    for (Format f : surface_capabilities.formats) {
        if (f == loon::gpu::Format::RGBA8UnormSrgb || f == loon::gpu::Format::BGRA8UnormSrgb) {
            // Choose 8 bit srgb if we have it
            return f;
        }
    }
    return surface_capabilities.formats[0];
}

TexturedCube::TexturedCube(const WindowState& window_state) {
    m_device = loon::gpu::create_device({
        .gpu_preference         = GpuPreference::Discrete,
        .native_window_handle   = window_state.native_window_handle,
        .native_instance_handle = window_state.native_instance_handle,
        .log_callback           = log_callback,
        .log_userdata           = nullptr,
        .log_level              = LogLevel::Debug,
        .alloc_callback         = nullptr,
        .alloc_userdata         = nullptr,
    });

    auto surface_capabilities = gpu::get_surface_capabilities(m_device);
    m_swapchain_format        = select_surface_format(surface_capabilities);

    recreate_swapchain(window_state.width, window_state.height);

    // Load shaders and create render pipeline
    ShaderModule shader         = window_state.shader_loader->load_module("textured_cube");
    const auto   vertex_spirv   = get_spirv(shader.get(), "vertex_main");
    const auto   fragment_spirv = get_spirv(shader.get(), "fragment_main");

    m_render_pipeline = gpu::create_graphics_pipeline(
        m_device,
        {
            .spirv       = Span(vertex_spirv.data(), vertex_spirv.size()).as_bytes(),
            .entry_point = "vertex_main"_sv,
        },
        {
            .spirv       = Span(fragment_spirv.data(), fragment_spirv.size()).as_bytes(),
            .entry_point = "fragment_main"_sv,
        },
        RasterDesc{
            .depth_format  = loon::gpu::Format::Depth32Float,
            .color_targets = {{.format = m_swapchain_format}},
        });

    assert(m_render_pipeline.h != 0);

    m_queue = gpu::get_queue(m_device);

    m_vertex_ptr = gpu::malloc(m_device, Cube::kSize, Memory::Gpu);

    m_constant_buffer = gpu::malloc(m_device, 1024ull * 1024);

    // Load the texture
    int            x = 0, y = 0, n = 0;
    unsigned char* image_data
        = stbi_load((window_state.file_paths.asset_directory + "uv-texture.png").c_str(),
                    &x,
                    &y,
                    &n,
                    4);

    m_color_texture = gpu::create_texture(
        m_device,
        TextureDesc{
            .dimensions = {(uint32_t)x, (uint32_t)y, 1},
            .format     = loon::gpu::Format::RGBA8UnormSrgb,
            .usage      = UsageFlags(UsageFlags::Sampled | UsageFlags::TransferDst),
        });

    m_texture_heap = gpu::create_texture_heap(m_device,
                                              {
                                                  .texture_count = 1024,
                                                  .sampler_count = 1,
                                              });
    m_color_view   = gpu::add_texture_view_to_heap(m_device,
                                                 m_texture_heap,
                                                   {
                                                       .texture = m_color_texture,
                                                       .format  = Format::RGBA8UnormSrgb,
                                                 });
    m_sampler      = gpu::add_sampler_to_heap(m_device, m_texture_heap, SamplerDesc{});

    // Copy over the geometry to the geometry buffer
    void* dst = gpu::get_host_pointer(m_device, m_constant_buffer);
    dst       = Cube::write(dst);

    memcpy(dst, image_data, (size_t)x * y * 4);
    stbi_image_free(image_data);

    auto cmd = gpu::queue_start_command_recording(m_queue);
    gpu::cmd_barrier(cmd,
                     StageFlags::None,
                     StageFlags::Transfer,
                     TextureTransition{
                         .texture    = m_color_texture,
                         .old_layout = loon::gpu::Layout::DontCare,
                         .new_layout = Layout::General,
                     });
    gpu::cmd_memcpy(cmd, m_vertex_ptr, m_constant_buffer, Cube::kSize);

    gpu::cmd_copy_to_texture(cmd,
                             m_constant_buffer + Cube::kSize,
                             m_color_texture,
                             BufferToTextureCopyInfo{
                                 .image_extent = {(uint32_t)x, (uint32_t)y, 1},
                             });
    //  A little excessive, but wait for the copy to be done before returning.
    gpu::cmd_barrier(cmd, StageFlags::Transfer, StageFlags::VertexShader);
    gpu::cmd_finalize(cmd);
    auto copy_semaphore = gpu::create_semaphore(m_device, 0);
    gpu::queue_submit(
        m_queue,
        cmd,
        {},
        SemaphoreInfo{.semaphore = copy_semaphore, .value = 1, .stage = StageFlags::Transfer});
    gpu::wait_semaphore(m_device, copy_semaphore, 1);
    gpu::free(m_device, copy_semaphore);

    m_depth_stencil_state = gpu::create_depth_stencil_state(
        m_device,
        DepthStencilDesc{
            .depth_mode = loon::gpu::DepthFlags::Write | loon::gpu::DepthFlags::Read,
            .depth_test = loon::gpu::Op::Greater,
        });
}

TexturedCube::~TexturedCube() {
    destroy_device(m_device);
}

void TexturedCube::recreate_swapchain(uint32_t width, uint32_t height) {
    gpu::device_wait_for_idle(m_device);
    gpu::unconfigure_surface(m_device);
    gpu::configure_surface(m_device,
                           {
                               .format       = m_swapchain_format,
                               .usages       = loon::gpu::UsageFlags::ColorAttachment,
                               .width        = width,
                               .height       = height,
                               .present_mode = PresentMode::Fifo,
                           });
    m_swapchain_width  = width;
    m_swapchain_height = height;

    // Recreate depth buffer as well
    if (m_depth_texture.h) { gpu::free(m_device, m_depth_texture); }
    m_depth_texture
        = gpu::create_texture(m_device,
                              {
                                  .type       = TextureType::Tex2D,
                                  .dimensions = {.x = width, .y = height, .z = 1},
                                  .format     = loon::gpu::Format::Depth32Float,
                                  .usage      = loon::gpu::UsageFlags::DepthStencilAttachment,
                              });
}

void TexturedCube::Update(const WindowState& window) {
    auto surface_texture = gpu::get_current_texture(m_device);
    if (surface_texture.status == SurfaceStatus::OutOfDate
        || surface_texture.status == SurfaceStatus::Suboptimal) {
        recreate_swapchain(window.width, window.height);
        return;
    } else if (surface_texture.status == SurfaceStatus::Error) {
        return;
    }

    // Update constant data
    auto args = reinterpret_cast<ShaderArgs*>(gpu::get_host_pointer(m_device, m_constant_buffer))
                + (m_frame_idx % 3);
    *args = ShaderArgs {
        .camera = CameraInfo{
            .projection        = projection({.view_width  = (float)window.width,
                                            .view_height = (float)window.height,
                                            .y_fov       = radians_from_degrees(30.f),
                                            .depth_far   = 0.5f}),
            .camera_from_world = transform3d::identity().translated({0, 0, -5}).to_matrix(),
        },
        .mesh = {
            .position        = m_vertex_ptr,
            .color           = m_vertex_ptr + sizeof(Cube::kPositions),
            .world_from_mesh = transform3d::identity()
                                .rotated_local(normalized({1, 0.5, 0}),
                                                radians_from_degrees((float)(m_frame_idx % 360)))
                                .to_matrix(),
        },
        .texture = m_color_view,
        .sampler = m_sampler,
    };

    auto cmd = gpu::queue_start_command_recording(m_queue);

    gpu::cmd_set_texture_heap(cmd, m_texture_heap);

    gpu::cmd_barrier(cmd,
                     StageFlags(StageFlags::Host | StageFlags::RasterColorOut),
                     StageFlags(StageFlags::PixelShader | StageFlags::RasterColorOut),
                     {TextureTransition{
                          .texture    = surface_texture.texture,
                          .old_layout = loon::gpu::Layout::DontCare,
                          .new_layout = Layout::Attachment,
                      },
                      TextureTransition{
                          .texture    = m_depth_texture,
                          .old_layout = loon::gpu::Layout::DontCare,
                          .new_layout = Layout::Attachment,
                      }});
    gpu::cmd_begin_render_pass(cmd,{
                                   .color_attachments = RenderAttachment{
                                       .texture = surface_texture.texture,
                                       .load_op      = loon::gpu::LoadOp::Clear,
                                       .store_op     = loon::gpu::StoreOp::Store,
                                       .clear_color  = Color(0, 0, 0, 0),
                                   }, 
                                   .depth_attachment = RenderAttachment{
                                    .texture = m_depth_texture,
                                    .load_op = loon::gpu::LoadOp::Clear,
                                    .store_op = loon::gpu::StoreOp::Discard,
                                    .clear_color = Color(0,0,0,0),
                                   }, 
                                   .render_area = {.width = m_swapchain_width, .height = m_swapchain_height},
                                });

    gpu::cmd_set_front_face(cmd, FrontFace::CW);
    gpu::cmd_set_cull_mode(cmd, Cull::Back);
    gpu::cmd_set_depth_stencil_state(cmd, m_depth_stencil_state);
    gpu::cmd_set_pipeline(cmd, m_render_pipeline);
    uint32_t args_offset = sizeof(ShaderArgs) * (m_frame_idx % 3);
    GpuPtr   argsGpu     = m_constant_buffer + args_offset;

    gpu::cmd_draw_indexed_instanced(
        cmd,
        {
            .vertexDataGpu   = argsGpu,
            .fragmentDataGpu = argsGpu + offsetof(ShaderArgs, texture),
            .indicesGpu      = m_vertex_ptr + sizeof(Cube::kPositions) + sizeof(Cube::kUVs),
            .indexCount      = Cube::kNumIndices,
        });

    gpu::cmd_end_render_pass(cmd);

    gpu::cmd_barrier(cmd,
                     StageFlags::RasterColorOut,
                     StageFlags::RasterColorOut,
                     TextureTransition{
                         .texture    = surface_texture.texture,
                         .old_layout = Layout::Attachment,
                         .new_layout = Layout::Present,
                     });
    gpu::cmd_finalize(cmd);
    gpu::queue_submit(m_queue, cmd);

    const auto status = gpu::present(m_device, m_queue);
    if (status == SurfaceStatus::OutOfDate || status == SurfaceStatus::Suboptimal) {
        recreate_swapchain(window.width, window.height);
    }

    gpu::queue_process_events(m_queue);

    m_frame_idx++;
}
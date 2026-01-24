/***
    Hello Triangle - A basic WGPU introduction

    This example shows setup, teardown, and basic rendering of a triangle.
    It shows everything in this file explicitly as a reference - other examples will use helpers for
   initialization.

*/

#include "textured_cube.h"

#include <gpu/loon_gpu.h>

#include <cassert>

#include "common/geometry.h"
#include "common/shaders.h"
#include "stb_image.h"

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
    CameraInfo camera;
    MeshGpu    mesh;
    uint64_t   texture;
    uint64_t   sampler;
};

static FORMAT select_surface_format(const loon::gpu::SurfaceCapabilities& surface_capabilities) {
    for (FORMAT f : surface_capabilities.formats) {
        if (f == loon::gpu::FORMAT_RGBA8UnormSrgb) {
            // Choose 8 bit srgb if we have it
            return f;
        }
    }
    return surface_capabilities.formats[0];
}

TexturedCube::TexturedCube(const WindowState& window_state) {
    m_device = loon::gpu::Device::create({
        .gpu_preference         = loon::gpu::GpuPreference_Discrete,
        .native_window_handle   = window_state.native_window_handle,
        .native_instance_handle = window_state.native_instance_handle,
        .log_callback           = log_callback,
        .log_userdata           = nullptr,
        .log_level              = LogLevel_Debug,
        .alloc_callback         = nullptr,
        .alloc_userdata         = nullptr,
    });

    auto surface_capabilities = m_device.get_surface_capabilities();
    m_swapchain_format        = select_surface_format(surface_capabilities);

    recreate_swapchain(window_state.width, window_state.height);

    // Load shaders and create render pipeline
    ShaderModule shader         = window_state.shader_loader->load_module("textured_cube.slang");
    const auto   vertex_spirv   = get_spirv(shader.get(), "vertexMain");
    const auto   fragment_spirv = get_spirv(shader.get(), "fragmentMain");

    m_render_pipeline = m_device.create_graphics_pipeline(
        {
            .spirv       = Span(vertex_spirv.data(), vertex_spirv.size()).as_bytes(),
            .entry_point = "vertexMain"_sv,
        },
        {
            .spirv       = Span(fragment_spirv.data(), fragment_spirv.size()).as_bytes(),
            .entry_point = "fragmentMain"_sv,
        },
        RasterDesc{
            .cull         = CULL_CW,
            .depthFormat  = loon::gpu::FORMAT_Depth32Float,
            .colorTargets = {{.format = m_swapchain_format}},
        });

    assert(m_render_pipeline.h != 0);

    m_queue = m_device.get_queue();

    m_geometry_buffer = m_device.malloc(Cube::kSize, MEMORY_GPU);
    m_vertex_ptr      = m_device.get_device_pointer(m_geometry_buffer);

    m_constant_buffer = m_device.malloc(1024ull * 1024);

    // Load the texture
    int            x = 0, y = 0, n = 0;
    unsigned char* image_data = stbi_load("E:\\loon_gpu\\assets\\uv-texture.png", &x, &y, &n, 4);

    m_color_texture = m_device.create_texture(TextureDesc{
        .dimensions = {(uint32_t)x, (uint32_t)y, 1},
        .format     = loon::gpu::FORMAT_RGBA8UnormSrgb,
        .usage      = USAGE_FLAGS(USAGE_SAMPLED | USAGE_TRANSFER_DST),
    });

    m_color_view = m_device.create_texture_view(m_color_texture, {.format = FORMAT_RGBA8UnormSrgb});

    m_texture_heap = m_device.create_texture_heap(1024);
    m_texture_id   = m_device.add_texture_view_to_heap(m_texture_heap, m_color_view);


    // Copy over the geometry to the geometry buffer
    void* dst = m_device.get_host_pointer(m_constant_buffer);
    dst       = Cube::write(dst);

    memcpy(dst, image_data, (size_t)x * y * 4);
    stbi_image_free(image_data);

    auto cmd = m_device.start_command_recording(m_queue);
    cmd.barrier(loon::gpu::STAGE_NONE,
                loon::gpu::STAGE_TRANSFER,
                TextureTransition{
                    .texture    = m_color_texture,
                    .old_layout = loon::gpu::LAYOUT_DONT_CARE,
                    .new_layout = LAYOUT_GENERAL,
                });
    cmd.memcpy(m_vertex_ptr, m_device.get_device_pointer(m_constant_buffer), Cube::kSize);

    cmd.copy_to_texture(m_device.get_device_pointer(m_constant_buffer) + Cube::kSize,
                        m_color_texture,
                        BufferToTextureCopyInfo{
                            .buffer_image_size = {(uint32_t)x, (uint32_t)y},
                            .image_extent      = {(uint32_t)x, (uint32_t)y, 1},
                        });
    //  A little excessive, but wait for the copy to be done before returning.
    cmd.barrier(loon::gpu::STAGE_TRANSFER, loon::gpu::STAGE_VERTEX_SHADER);
    auto copy_semaphore = m_device.create_semaphore(0);
    m_device.submit(
        m_queue,
        cmd,
        {},
        SemaphoreInfo{.semaphore = copy_semaphore, .value = 1, .stage = STAGE_TRANSFER});
    m_device.wait_semaphore(copy_semaphore, 1);
    m_device.free(copy_semaphore);

    m_depth_stencil_state = m_device.create_depth_stencil_state(DepthStencilDesc{
        .depthMode = DEPTH_FLAGS(loon::gpu::DEPTH_WRITE | loon::gpu::DEPTH_READ),
        .depthTest = loon::gpu::OP_GREATER,
    });
}


TexturedCube::~TexturedCube() {
    m_device.wait_for_device_idle();
    m_device.free(m_render_pipeline);
    m_device.unconfigure_surface();
}

void TexturedCube::recreate_swapchain(uint32_t width, uint32_t height) {
    m_device.wait_for_device_idle();
    m_device.unconfigure_surface();
    m_device.configure_surface({
        .format       = m_swapchain_format,
        .usages       = loon::gpu::USAGE_COLOR_ATTACHMENT,
        .width        = width,
        .height       = height,
        .present_mode = PRESENT_MODE_FIFO,
    });
    m_swapchain_width  = width;
    m_swapchain_height = height;

    // Recreate depth buffer as well
    if (m_depth_view.h) {
        m_device.free(m_depth_view);
        m_device.free(m_depth_texture);
    }
    m_depth_texture = m_device.create_texture({
        .type       = TEXTURE_2D,
        .dimensions = {.x = width, .y = height, .z = 1},
        .format     = loon::gpu::FORMAT_Depth32Float,
        .usage      = loon::gpu::USAGE_DEPTH_STENCIL_ATTACHMENT,
    });

    m_depth_view = m_device.create_texture_view(m_depth_texture,
                                                TextureViewDesc{
                                                    .format = loon::gpu::FORMAT_Depth32Float,
                                                });
}

void TexturedCube::Update(const WindowState& window) {
    auto surface_texture = m_device.get_current_texture();
    if (surface_texture.status == SURFACE_STATUS_OUT_OF_DATE
        || surface_texture.status == SURFACE_STATUS_SUBOPTIMAL) {
        recreate_swapchain(window.width, window.height);
        return;
    } else if (surface_texture.status == SURFACE_STATUS_ERROR) {
        return;
    }

    // Update constant data
    auto args = reinterpret_cast<ShaderArgs*>(m_device.get_host_pointer(m_constant_buffer))
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
        .texture = m_texture_id,
        .sampler = 0,
    };

    auto swapchain_view = m_device.create_texture_view(surface_texture.texture,
                                                       TextureViewDesc{
                                                           .format     = m_swapchain_format,
                                                           .baseMip    = 0,
                                                           .mipCount   = 1,
                                                           .baseLayer  = 0,
                                                           .layerCount = 1,
                                                       });

    auto commandBuffer = m_device.start_command_recording(m_queue);

    commandBuffer.set_texture_heap(m_texture_heap);

    commandBuffer.barrier(STAGE_FLAGS(STAGE_HOST | STAGE_RASTER_COLOR_OUT),
                          STAGE_FLAGS(STAGE_PIXEL_SHADER | STAGE_RASTER_COLOR_OUT),
                          {TextureTransition{
                               .texture    = surface_texture.texture,
                               .old_layout = loon::gpu::LAYOUT_DONT_CARE,
                               .new_layout = LAYOUT_ATTACHMENT,
                           },
                           TextureTransition{
                               .texture    = m_depth_texture,
                               .old_layout = loon::gpu::LAYOUT_DONT_CARE,
                               .new_layout = LAYOUT_ATTACHMENT,
                           }});
    commandBuffer.begin_render_pass({
                                   .color_attachments = RenderAttachment{
                                       .texture_view = swapchain_view,
                                       .load_op      = loon::gpu::LOAD_OP_CLEAR,
                                       .store_op     = loon::gpu::STORE_OP_STORE,
                                       .clear_color  = Color(0, 0, 0, 0),
                                   }, 
                                   .depth_attachment = RenderAttachment{
                                    .texture_view = m_depth_view,
                                    .load_op = loon::gpu::LOAD_OP_CLEAR,
                                    .store_op = loon::gpu::STORE_OP_DISCARD,
                                    .clear_color = Color(0,0,0,0),
                                   }, 
                                   .render_area = {.width = m_swapchain_width, .height = m_swapchain_height},
                                });
    commandBuffer.set_depth_stencil_State(m_depth_stencil_state);
    commandBuffer.set_pipeline(m_render_pipeline);
    uint32_t args_offset = sizeof(ShaderArgs) * (m_frame_idx % 3);
    GpuPtr   argsGpu     = m_device.get_device_pointer(m_constant_buffer) + args_offset;

    commandBuffer.draw_indexed_instanced(
        argsGpu,
        argsGpu + offsetof(ShaderArgs, texture),
        m_vertex_ptr + sizeof(Cube::kPositions) + sizeof(Cube::kUVs),
        Cube::kNumIndices,
        1);

    commandBuffer.end_render_pass();
    commandBuffer.barrier(STAGE_RASTER_COLOR_OUT,
                          STAGE_RASTER_COLOR_OUT,
                          TextureTransition{
                              .texture    = surface_texture.texture,
                              .old_layout = LAYOUT_ATTACHMENT,
                              .new_layout = LAYOUT_PRESENT,
                          });
    m_device.submit(m_queue,
                    commandBuffer,
                    SemaphoreInfo{
                        .semaphore = surface_texture.acquire_semaphore,
                        .stage     = loon::gpu::STAGE_PIXEL_SHADER,
                    });

    const auto status = m_device.present(m_queue);
    if (status == SURFACE_STATUS_OUT_OF_DATE || status == SURFACE_STATUS_SUBOPTIMAL) {
        recreate_swapchain(window.width, window.height);
    }

    m_device.on_submitted_work_completed(m_queue,
                                         [&, swapchain_view]() { m_device.free(swapchain_view); });
    m_device.process_events(m_queue);

    m_frame_idx++;
}
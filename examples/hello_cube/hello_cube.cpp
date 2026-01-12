/***
    Hello Triangle - A basic WGPU introduction

    This example shows setup, teardown, and basic rendering of a triangle.
    It shows everything in this file explicitly as a reference - other examples will use helpers for
   initialization.

*/

#include "hello_cube.h"

#include <gpu/loon_gpu.h>

#include <cassert>

#include "common/geometry.h"
#include "common/shaders.h"

struct Vertex {
    float3 position;
    float3 color;
};

constexpr Vertex kCubeVerts[] = {
    {.position = {1, 1, 1}, .color = {1, 0, 0}},
    {.position = {1, 1, -1}, .color = {0, 1, 0}},
    {.position = {1, -1, 1}, .color = {0, 0, 1}},
    {.position = {1, -1, -1}, .color = {1, 0, 1}},
    {.position = {-1, 1, 1}, .color = {0, 1, 1}},
    {.position = {-1, 1, -1}, .color = {1, 1, 0}},
    {.position = {-1, -1, 1}, .color = {0, 1, 0}},
    {.position = {-1, -1, -1}, .color = {0, 0, 1}},
};

constexpr uint16_t kCubeIndices[] = {0, 2, 1, 1, 2, 3, 0, 6, 2, 0, 4, 6, 4, 5, 6, 6, 5, 7,
                                     5, 1, 7, 7, 1, 3, 1, 0, 4, 1, 4, 5, 2, 6, 3, 6, 7, 3};

struct CameraInfo {
    float3   position                = {0, 0, 5.f};
    quatf    rotationCameraFromWorld = {};
    float4x4 projection              = {};
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

HelloCube::HelloCube(const WindowState& window_state) {
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

    m_device.configure_surface({
        .format       = m_swapchain_format,
        .usages       = loon::gpu::USAGE_COLOR_ATTACHMENT,
        .width        = window_state.width,
        .height       = window_state.height,
        .present_mode = PRESENT_MODE_FIFO,
    });
    m_swapchain_width  = window_state.width;
    m_swapchain_height = window_state.height;

    // Load shaders and create render pipeline
    ShaderModule shader         = window_state.shader_loader->load_module("hello_cube.slang");
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
        RasterDesc{.colorTargets = {{.format = m_swapchain_format}}});

    assert(m_render_pipeline.h != 0);

    m_queue     = m_device.get_queue();
    m_semaphore = m_device.create_semaphore(0);
}

HelloCube::~HelloCube() {
    m_device.wait_for_device_idle();
    m_device.free(m_render_pipeline);
    m_device.free(m_semaphore);
    m_device.unconfigure_surface();
}

void HelloCube::Update(const WindowState& window) {
    constexpr uint64_t kMaxFramesInFlight = 3;
    m_device.wait_semaphore(m_semaphore,
                           std::max(m_frame_idx, kMaxFramesInFlight) - kMaxFramesInFlight);
    auto surface_texture = m_device.get_current_texture();
    if (surface_texture.status == loon::gpu::SurfaceTextureInfo::STATUS_OUT_OF_DATE
        || surface_texture.status == loon::gpu::SurfaceTextureInfo::STATUS_SUBOPTIMAL) {
        m_device.unconfigure_surface();
        m_device.configure_surface({
            .format       = m_swapchain_format,
            .usages       = loon::gpu::USAGE_COLOR_ATTACHMENT,
            .width        = window.width,
            .height       = window.height,
            .present_mode = PRESENT_MODE_FIFO,
        });
        m_swapchain_width  = window.width;
        m_swapchain_height = window.height;
        return;
    } else if (surface_texture.status == loon::gpu::SurfaceTextureInfo::STATUS_ERROR) {
        return;
    }

    auto swapchain_view = m_device.create_texture_view(surface_texture.texture,
                                                     TextureViewDesc{
                                                         .format     = m_swapchain_format,
                                                         .baseMip    = 0,
                                                         .mipCount   = 1,
                                                         .baseLayer  = 0,
                                                         .layerCount = 1,
                                                     });

    auto commandBuffer = m_device.start_command_recording(m_queue);

    commandBuffer.make_surface_writable();
    commandBuffer.begin_render_pass({
                                   .color_attachments = RenderAttachment{
                                       .texture_view = swapchain_view,
                                       .load_op      = loon::gpu::LOAD_OP_CLEAR,
                                       .store_op     = loon::gpu::STORE_OP_STORE,
                                       .clear_color  = Color(0, 0, 0, 0),
                                   }, .render_area = {.width = m_swapchain_width, .height = m_swapchain_height},});

    commandBuffer.set_pipeline(m_render_pipeline);
    commandBuffer.draw(0, 0, 3, 1);

    commandBuffer.end_render_pass();
    commandBuffer.make_surface_presentable();
    m_device.submit(m_queue,
                    commandBuffer,
                    SemaphoreInfo{
                        .semaphore = surface_texture.acquire_semaphore,
                        .stage     = loon::gpu::STAGE_RASTER_COLOR_OUT,
                    },
                    SemaphoreInfo{
                        .semaphore = m_semaphore,
                        .value     = ++m_frame_idx,
                    });

    m_device.present(m_queue);

    m_device.on_submitted_work_completed(m_queue,
                                         [&, swapchain_view]() { m_device.free(swapchain_view); });
    m_device.process_events(m_queue);
}
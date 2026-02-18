/***
    Hello Triangle - A basic WGPU introduction

    This example shows setup, teardown, and basic rendering of a triangle.
    It shows everything in this file explicitly as a reference - other examples will use helpers for
   initialization.

*/

#include "hello_triangle.h"

#include <gpu/loon_gpu.h>

#include <cassert>

#include "common/shaders.h"

using namespace loon::gpu;

static Format select_surface_format(const loon::gpu::SurfaceCapabilities& surface_capabilities) {
    for (Format f : surface_capabilities.formats) {
        if (f == loon::gpu::Format::RGBA8UnormSrgb) {
            // Choose 8 bit srgb if we have it
            return f;
        }
    }
    return surface_capabilities.formats[0];
}


HelloTriangle::HelloTriangle(const WindowState& window_state) {
    m_device = loon::gpu::Device::create({
        .gpu_preference         = loon::gpu::GpuPreference::Discrete,
        .native_window_handle   = window_state.native_window_handle,
        .native_instance_handle = window_state.native_instance_handle,
        .log_callback           = log_callback,
        .log_userdata           = nullptr,
        .log_level              = LogLevel::Debug,
        .alloc_callback         = nullptr,
        .alloc_userdata         = nullptr,
    });

    auto surface_capabilities = m_device.get_surface_capabilities();
    m_swapchain_format        = select_surface_format(surface_capabilities);

    m_device.configure_surface({
        .format       = m_swapchain_format,
        .usages       = loon::gpu::UsageFlags::ColorAttachment,
        .width        = window_state.width,
        .height       = window_state.height,
        .present_mode = PresentMode::Fifo,
    });
    m_swapchain_width  = window_state.width;
    m_swapchain_height = window_state.height;

    // Load shaders and create render pipeline
    ShaderModule shader         = window_state.shader_loader->load_module("hello_triangle.slang");
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
        RasterDesc{.color_targets = {{.format = m_swapchain_format}}});

    assert(m_render_pipeline.h != 0);

    m_queue = m_device.get_queue();
}

HelloTriangle::~HelloTriangle() {}

void HelloTriangle::recreate_swapchain(uint32_t width, uint32_t height) {
    m_device.unconfigure_surface();
    m_device.configure_surface({
        .format       = m_swapchain_format,
        .usages       = loon::gpu::UsageFlags::ColorAttachment,
        .width        = width,
        .height       = height,
        .present_mode = PresentMode::Fifo,
    });
    m_swapchain_width  = width;
    m_swapchain_height = height;
}

void HelloTriangle::Update(const WindowState& window) {
    auto surface_texture = m_device.get_current_texture();
    if (surface_texture.status == SurfaceStatus::OutOfDate
        || surface_texture.status == SurfaceStatus::Suboptimal) {
        recreate_swapchain(window.width, window.height);
        return;
    } else if (surface_texture.status == SurfaceStatus::Error) {
        return;
    }

    auto command_buffer = m_queue.start_command_recording();

    command_buffer.barrier(StageFlags::RasterColorOut,
                           StageFlags::RasterColorOut,
                           TextureTransition{
                               .texture    = surface_texture.texture,
                               .old_layout = loon::gpu::Layout::DontCare,
                               .new_layout = Layout::Attachment,
                           });
    command_buffer.begin_render_pass({
                                   .color_attachments = RenderAttachment{
                                       .texture = surface_texture.texture,
                                       .load_op      = loon::gpu::LoadOp::Clear,
                                       .store_op     = loon::gpu::StoreOp::Store,
                                       .clear_color  = Color(0, 0, 0, 0),
                                   }, .render_area = {.width = m_swapchain_width, .height = m_swapchain_height},});

    command_buffer.set_pipeline(m_render_pipeline);
    command_buffer.draw(0, 0, 3, 1);

    command_buffer.end_render_pass();
    command_buffer.barrier(StageFlags::RasterColorOut,
                           StageFlags::RasterColorOut,
                           TextureTransition{
                               .texture    = surface_texture.texture,
                               .old_layout = Layout::Attachment,
                               .new_layout = Layout::Present,
                           });

    m_queue.submit(command_buffer,
                   SemaphoreInfo{
                       .semaphore = surface_texture.acquire_semaphore,
                       .stage     = StageFlags::RasterColorOut,
                   });

    const auto status = m_device.present(m_queue);
    if (status == SurfaceStatus::OutOfDate || status == SurfaceStatus::Suboptimal) {
        recreate_swapchain(window.width, window.height);
    }

    m_queue.process_events();
}
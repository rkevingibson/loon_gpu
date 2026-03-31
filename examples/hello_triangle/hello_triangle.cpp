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

using namespace loon;

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
    m_device = loon::gpu::create_device({
        .gpu_preference         = loon::gpu::GpuPreference::Discrete,
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

    gpu::configure_surface(m_device,
                           {
                               .format       = m_swapchain_format,
                               .usages       = loon::gpu::UsageFlags::ColorAttachment,
                               .width        = window_state.width,
                               .height       = window_state.height,
                               .present_mode = PresentMode::Fifo,
                           });
    m_swapchain_width  = window_state.width;
    m_swapchain_height = window_state.height;

    // Load shaders and create render pipeline
    ShaderModule shader         = window_state.shader_loader->load_module("hello_triangle");
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
        RasterDesc{.color_targets = {{.format = m_swapchain_format}}});

    assert(m_render_pipeline.h != 0);

    m_queue = gpu::get_queue(m_device);
}

HelloTriangle::~HelloTriangle() {
    destroy_device(m_device);
}

void HelloTriangle::recreate_swapchain(uint32_t width, uint32_t height) {
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
}

void HelloTriangle::Update(const WindowState& window) {
    auto surface_texture = gpu::get_current_texture(m_device);
    if (surface_texture.status == SurfaceStatus::OutOfDate
        || surface_texture.status == SurfaceStatus::Suboptimal) {
        recreate_swapchain(window.width, window.height);
        return;
    } else if (surface_texture.status == SurfaceStatus::Error) {
        return;
    }

    auto cmd = gpu::queue_start_command_recording(m_queue);

    gpu::cmd_barrier(cmd,
                     StageFlags::RasterColorOut,
                     StageFlags::RasterColorOut,
                     TextureTransition{
                         .texture    = surface_texture.texture,
                         .old_layout = loon::gpu::Layout::DontCare,
                         .new_layout = Layout::Attachment,
                     });
    gpu::cmd_begin_render_pass(cmd, {
                                   .color_attachments = RenderAttachment{
                                       .texture = surface_texture.texture,
                                       .load_op      = loon::gpu::LoadOp::Clear,
                                       .store_op     = loon::gpu::StoreOp::Store,
                                       .clear_color  = Color(0, 0, 0, 0),
                                   }, .render_area = {.width = m_swapchain_width, .height =
                                   m_swapchain_height},});

    gpu::cmd_set_pipeline(cmd, m_render_pipeline);
    gpu::cmd_draw(cmd, 0, 0, 3, 1);

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
}
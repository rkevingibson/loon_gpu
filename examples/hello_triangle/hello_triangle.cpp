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

HelloTriangle::HelloTriangle(const WindowState& window_state) : Example(window_state) {
    // Load shaders and create render pipeline
    ShaderModule shader         = window_state.shader_loader->load_module("hello_triangle");
    const auto   vertex_spirv   = get_spirv(shader.get(), "vertex_main");
    const auto   fragment_spirv = get_spirv(shader.get(), "fragment_main");

    m_render_pipeline = gpu::create_graphics_pipeline(
        m_device,
        {
            .source      = Span(vertex_spirv.data(), vertex_spirv.size()).as_bytes(),
            .entry_point = "vertex_main"_sv,
        },
        {
            .source      = Span(fragment_spirv.data(), fragment_spirv.size()).as_bytes(),
            .entry_point = "fragment_main"_sv,
        },
        RasterDesc{.color_targets = {{.format = m_swapchain_format}}});

    assert(m_render_pipeline.h != 0);
}

bool HelloTriangle::update(const UpdateInfo& info) {
    auto cmd = gpu::queue_start_command_recording(m_queue);

    gpu::cmd_wait_for_surface_texture(cmd);
    gpu::cmd_begin_render_pass(cmd,
                               {
                                   .color_attachments =
                                       RenderAttachment{
                                           .texture     = info.color_texture,
                                           .load_op     = loon::gpu::LoadOp::Clear,
                                           .store_op    = loon::gpu::StoreOp::Store,
                                           .clear_color = Color(0, 0, 0, 0),
                                       },
                                   .render_area =
                                       Rect2D{
                                           .width  = info.texture_size.x,
                                           .height = info.texture_size.y,
                                       },
                               });

    gpu::cmd_set_pipeline(cmd, m_render_pipeline);
    gpu::cmd_draw(cmd, 0, 0, 3, 1);

    gpu::cmd_end_render_pass(cmd);
    gpu::cmd_signal_surface_texture(cmd);
    gpu::cmd_finalize(cmd);

    gpu::queue_submit(m_queue, cmd);

    return true;
}
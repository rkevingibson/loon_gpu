/***
    Hello Triangle - A basic WGPU introduction

    This example shows setup, teardown, and basic rendering of a triangle.
    It shows everything in this file explicitly as a reference - other examples will use helpers for
   initialization.

*/

#include "hello_cube.h"

#include <gpu/loon_gpu.h>

#include <cassert>
#include <cstring>

#include "common/geometry.h"
#include "common/shaders.h"
#include "example.h"
using namespace geometry;
using namespace loon;
namespace {
struct Cube {
    static constexpr float3 kPositions[] = {
        {1, 1, 1},
        {1, 1, -1},
        {1, -1, 1},
        {1, -1, -1},
        {-1, 1, 1},
        {-1, 1, -1},
        {-1, -1, 1},
        {-1, -1, -1},
    };
    static constexpr float3 kColors[] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0},
        {0, 1, 0},
        {0, 0, 1},
    };
    static constexpr uint16_t kIndices[] = {0, 2, 1, 1, 2, 3, 0, 6, 2, 0, 4, 6, 4, 5, 6, 6, 5, 7,
                                            5, 1, 7, 7, 1, 3, 1, 0, 4, 1, 4, 5, 2, 6, 3, 6, 7, 3};
    static constexpr size_t   kSize      = sizeof(kPositions) + sizeof(kColors) + sizeof(kIndices);

    static void write(void* ptr) {
        char* dst = (char*)ptr;
        memcpy(ptr, kPositions, sizeof(kPositions));
        dst += sizeof(kPositions);
        memcpy(dst, kColors, sizeof(kColors));
        dst += sizeof(kColors);
        memcpy(dst, kIndices, sizeof(kIndices));
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

HelloCube::HelloCube(const WindowState& window_state) : Example(window_state) {
    // Load shaders and create render pipeline
    ShaderModule shader         = window_state.shader_loader->load_module("hello_cube");
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
        RasterDesc{
            .depth_format  = loon::gpu::Format::Depth32Float,
            .color_targets = {{.format = m_swapchain_format}},
        });

    assert(m_render_pipeline.h != 0);

    m_vertex_ptr = gpu::malloc(m_device, Cube::kSize, Memory::Gpu);

    m_constant_buffer = gpu::malloc(m_device, 1024ull * 1024);
    // Copy over the geometry to the geometry buffer
    void* dst = gpu::get_host_pointer(m_device, m_constant_buffer);
    Cube::write(dst);

    auto cmd = gpu::queue_start_command_recording(m_queue);
    gpu::cmd_memcpy(cmd, m_vertex_ptr, m_constant_buffer, Cube::kSize);

    // A little excessive, but wait for the copy to be done before returning.
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

    // Create a depth state
    m_depth_stencil_state = gpu::create_depth_stencil_state(
        m_device,
        DepthStencilDesc{
            .depth_mode = loon::gpu::DepthFlags::Write | loon::gpu::DepthFlags::Read,
            .depth_test = loon::gpu::Op::Greater,
        });
}

bool HelloCube::Update(const UpdateInfo& info) {
    // Update constant data
    auto args = reinterpret_cast<ShaderArgs*>(gpu::get_host_pointer(m_device, m_constant_buffer));
    args[m_frame_idx % 3].camera = CameraInfo{
        .projection        = projection({.view_width  = (float)info.texture_size.x,
                                         .view_height = (float)info.texture_size.y,
                                         .y_fov       = radians_from_degrees(30.f),
                                         .depth_far   = 0.5f}),
        .camera_from_world = transform3d::identity().translated({0, 0, -5}).to_matrix(),
    };
    args[m_frame_idx % 3].mesh = {
        .position        = m_vertex_ptr,
        .color           = m_vertex_ptr + sizeof(Cube::kPositions),
        .world_from_mesh = transform3d::identity()
                               .rotated_local(normalized({1, 0.5, 0}),
                                              radians_from_degrees((float)(m_frame_idx % 360)))
                               .to_matrix(),
    };

    auto cmd = gpu::queue_start_command_recording(m_queue);
    gpu::cmd_wait_for_surface_texture(cmd);
    gpu::cmd_barrier(cmd, StageFlags::FragmentTests, StageFlags::FragmentTests);
    gpu::cmd_begin_render_pass(cmd,
                               {
                                   .color_attachments =
                                       RenderAttachment{
                                           .texture     = info.color_texture,
                                           .load_op     = loon::gpu::LoadOp::Clear,
                                           .store_op    = loon::gpu::StoreOp::Store,
                                           .clear_color = Color(0, 0, 0, 0),
                                       },
                                   .depth_attachment =
                                       RenderAttachment{
                                           .texture     = info.depth_texture,
                                           .load_op     = loon::gpu::LoadOp::Clear,
                                           .store_op    = loon::gpu::StoreOp::Discard,
                                           .clear_color = Color(0, 0, 0, 0),
                                       },
                                   .render_area =
                                       {
                                           .width  = info.texture_size.x,
                                           .height = info.texture_size.y,
                                       },
                               });
    gpu::cmd_set_depth_stencil_state(cmd, m_depth_stencil_state);
    gpu::cmd_set_pipeline(cmd, m_render_pipeline);
    gpu::cmd_draw_indexed_instanced(
        cmd,
        {
            .vertexDataGpu   = m_constant_buffer + sizeof(ShaderArgs) * (m_frame_idx % 3),
            .fragmentDataGpu = 0,
            .indicesGpu      = m_vertex_ptr + sizeof(Cube::kPositions) + sizeof(Cube::kColors),
            .indexCount      = 36,
        });

    gpu::cmd_end_render_pass(cmd);
    gpu::cmd_signal_surface_texture(cmd);
    gpu::cmd_finalize(cmd);
    gpu::queue_submit(m_queue, cmd);

    ++m_frame_idx;
    return true;
}
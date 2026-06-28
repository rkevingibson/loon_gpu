#include "many_cubes.h"

#include <gpu/loon_gpu.h>

#include <cassert>
#include <cstring>

#include "common/geometry.h"
#include "common/shaders.h"
#include "common/timer.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_loon.h"
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
    static constexpr uint16_t kIndices[]  = {0,  3,  1,  0,  2,  3,  4,  6,  7,  4,  7,  5,
                                             8,  9,  11, 8,  11, 10, 12, 13, 15, 12, 15, 14,
                                             16, 17, 18, 18, 17, 19, 20, 21, 22, 22, 21, 23};
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

struct CameraData {
    geometry::float4x4 projection        = {};
    geometry::float4x4 camera_from_world = {};
};

struct VertArgs {
    float4x4 world_from_mesh;
    GpuPtr   camera;
    GpuPtr   position;
    GpuPtr   uvs;
};

struct FragArgs {
    TextureView texture;
    Sampler     sampler;
};


}  // namespace

static Format select_surface_format(const loon::gpu::SurfaceCapabilities& surface_capabilities) {
    for (Format f : surface_capabilities.formats) {
        if (f == loon::gpu::Format::RGBA8UnormSrgb || f == loon::gpu::Format::BGRA8UnormSrgb) {
            // Choose 8 bit srgb if we have it
            return f;
        }
    }
    return surface_capabilities.formats[0];
}

ManyCubes::ManyCubes(const WindowState& window_state) : Example(window_state) {
    // Load shaders and create render pipeline
    ShaderModule shader         = window_state.shader_loader->load_module("many_cubes");
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


    // Load the texture
    int            x = 0, y = 0, n = 0;
    unsigned char* image_data =
        stbi_load((window_state.file_paths.asset_directory + "uv-texture.png").c_str(),
                  &x,
                  &y,
                  &n,
                  4);

    m_color_texture =
        gpu::create_texture(m_device,
                            TextureDesc{
                                .dimensions = {(uint32_t)x, (uint32_t)y, 1},
                                .format     = loon::gpu::Format::RGBA8UnormSrgb,
                                .usage = UsageFlags(UsageFlags::Sampled | UsageFlags::TransferDst),
                            });

    m_texture_heap = gpu::create_texture_heap(m_device,
                                              {
                                                  .texture_count = 1024,
                                                  .sampler_count = 5,
                                              });
    m_color_view   = gpu::add_texture_view_to_heap(m_device,
                                                 m_texture_heap,
                                                   {
                                                       .texture = m_color_texture,
                                                       .format  = Format::RGBA8UnormSrgb,
                                                 });
    m_sampler      = gpu::add_sampler_to_heap(m_device, m_texture_heap, SamplerDesc{});

    // Copy texture and geometry into staging buffer:
    const size_t image_size     = (size_t)x * y * 4;
    auto         staging_buffer = gpu::malloc(m_device, Cube::kSize + image_size);
    void*        cube_dst       = gpu::get_host_pointer(m_device, staging_buffer);
    void*        image_dst      = Cube::write(cube_dst);
    memcpy(image_dst, image_data, image_size);
    stbi_image_free(image_data);

    // GPU-side copy, but block on the result for simplicity
    auto cmd = gpu::queue_start_command_recording(m_queue);
    gpu::cmd_memcpy(cmd, m_vertex_ptr, staging_buffer, Cube::kSize);

    gpu::cmd_copy_to_texture(cmd,
                             staging_buffer + Cube::kSize,
                             m_color_texture,
                             BufferTextureCopyInfo{
                                 .image_extent = {(uint32_t)x, (uint32_t)y, 1},
                             });
    gpu::cmd_barrier(cmd, StageFlags::Transfer, StageFlags::VertexShader);
    gpu::cmd_finalize(cmd);
    gpu::queue_submit(m_queue, cmd, {}, {});
    gpu::device_wait_for_idle(m_device);
    gpu::free(m_device, staging_buffer);

    m_depth_stencil_state = gpu::create_depth_stencil_state(
        m_device,
        DepthStencilDesc{
            .depth_mode = loon::gpu::DepthFlags::Write | loon::gpu::DepthFlags::Read,
            .depth_test = loon::gpu::Op::Greater,
        });


    loon::imgui::Init({
        .device                    = m_device,
        .queue                     = m_queue,
        .texture_heap              = m_texture_heap,
        .num_frames_in_flight      = 3,
        .render_target_format      = m_swapchain_format,
        .depth_stencil_view_format = loon::gpu::Format::Depth32Float,
        .shader_loader             = window_state.shader_loader.get(),
    });

    m_ring_buffer = loon::RingBuffer(m_device, 128 * 1024 * 1024, 3);
    m_frame_idx   = 0;
}

ManyCubes::~ManyCubes() {
    gpu::device_wait_for_idle(m_device);
    loon::imgui::Shutdown();
}


bool ManyCubes::Update(const UpdateInfo& info) {
    // Imgui:

    loon::imgui::NewFrame();

    ImGui::Begin("Cubes!");
    ImGui::SliderInt("Cubes x", &m_grid_width, 1, 500);
    ImGui::SliderInt("Cubes y", &m_grid_height, 1, 500);
    ImGui::Checkbox("Use instanced draw", &m_use_instanced_draws);

    ImGui::LabelText("Timing", "CPU: %lld us", m_frame_time_average);
    ImGui::PlotLines("Frame time",
                     m_frame_time_ms,
                     kFrameTimeWindow,
                     m_frame_idx % kFrameTimeWindow,
                     0,
                     0.0f,
                     1000.f / 60.f,
                     ImVec2(0, 80.f));

    ImGui::End();
    m_frame_idx++;

    // Set up global draw arguments, these are constant for all cubes drawn.
    GpuPtr camera = m_ring_buffer.append(
        m_frame_idx,
        CameraData{
            .projection        = geometry::projection({.view_width  = (float)info.texture_size.x,
                                                       .view_height = (float)info.texture_size.y,
                                                       .y_fov       = geometry::radians_from_degrees(30.f),
                                                       .depth_far   = 0.5f}),
            .camera_from_world = geometry::transform3d::identity()
                                     .translated({0, 2, -8})
                                     .rotated_local({1, 0, 0}, radians_from_degrees(-30))
                                     .to_matrix(),
        });

    GpuPtr frag = m_ring_buffer.append(m_frame_idx,
                                       FragArgs{
                                           .texture = m_color_view,
                                           .sampler = m_sampler,
                                       });

    // Render

    auto frame_start = loon::Instant::now();
    auto cmd         = gpu::queue_start_command_recording(m_queue);

    gpu::cmd_wait_for_surface_texture(cmd);
    // Barrier to prevent the depth buffer being cleared before the last frame was done.
    gpu::cmd_barrier(cmd, StageFlags::FragmentTests, StageFlags::FragmentTests);

    gpu::cmd_begin_render_pass(cmd,
                               {
                                   .color_attachments =
                                       RenderAttachment{
                                           .texture     = info.color_texture,
                                           .load_op     = LoadOp::Clear,
                                           .store_op    = StoreOp::Store,
                                           .clear_color = Color(0, 0, 0, 0),
                                       },
                                   .depth_attachment =
                                       RenderAttachment{
                                           .texture     = info.depth_texture,
                                           .load_op     = LoadOp::Clear,
                                           .store_op    = StoreOp::Discard,
                                           .clear_color = Color(0, 0, 0, 0),
                                       },
                                   .render_area =
                                       {
                                           .width  = info.texture_size.x,
                                           .height = info.texture_size.y,
                                       },
                               });
    gpu::cmd_set_front_face(cmd, FrontFace::CW);
    gpu::cmd_set_cull_mode(cmd, Cull::Back);
    gpu::cmd_set_depth_stencil_state(cmd, m_depth_stencil_state);
    gpu::cmd_set_pipeline(cmd, m_render_pipeline);
    gpu::cmd_set_texture_heap(cmd, m_texture_heap);

    const auto grid_transform = [&](int x, int y) {
        return transform3d::from_axis_angle_and_origin(
                   normalized({1, 0.5, 0}),
                   radians_from_degrees((float)((30 * x + 10 * y + m_frame_idx) % 360)),
                   {4.f * (float)(x - m_grid_width / 2), 0, -4.f * (float)y})
            .to_matrix();
    };

    if (m_use_instanced_draws) {
        GpuPtr   vert_args = ~0;
        uint32_t num_cubes = 0;
        for (int x = 0; x < m_grid_width; ++x) {
            for (int y = 0; y < m_grid_height; ++y) {
                GpuPtr tx = m_ring_buffer.append(m_frame_idx,
                                                 VertArgs{
                                                     .world_from_mesh = grid_transform(x, y),
                                                     .camera          = camera,
                                                     .position        = m_vertex_ptr,
                                                     .uvs = m_vertex_ptr + sizeof(Cube::kPositions),
                                                 });

                if (x == 0 && y == 0) { vert_args = tx; }

                // We've wrapped around the ring buffer end, so submit a draw with whatever we've
                // recorded so far.
                if (tx < vert_args) {
                    gpu::cmd_draw_indexed_instanced(
                        cmd,
                        {
                            .vertexDataGpu   = vert_args,
                            .fragmentDataGpu = frag,
                            .indicesGpu =
                                m_vertex_ptr + sizeof(Cube::kPositions) + sizeof(Cube::kUVs),
                            .indexCount    = Cube::kNumIndices,
                            .instanceCount = num_cubes,
                        });

                    num_cubes = 0;
                    vert_args = tx;
                }
                ++num_cubes;
            }
        }
        if (num_cubes) {
            gpu::cmd_draw_indexed_instanced(
                cmd,
                {
                    .vertexDataGpu   = vert_args,
                    .fragmentDataGpu = frag,
                    .indicesGpu      = m_vertex_ptr + sizeof(Cube::kPositions) + sizeof(Cube::kUVs),
                    .indexCount      = Cube::kNumIndices,
                    .instanceCount   = num_cubes,
                });
        }

    } else {
        for (int x = 0; x < m_grid_width; ++x) {
            for (int y = 0; y < m_grid_height; ++y) {
                GpuPtr vert_args =
                    m_ring_buffer.append(m_frame_idx,
                                         VertArgs{
                                             .world_from_mesh = grid_transform(x, y),
                                             .camera          = camera,
                                             .position        = m_vertex_ptr,
                                             .uvs = m_vertex_ptr + sizeof(Cube::kPositions),
                                         });

                gpu::cmd_draw_indexed_instanced(
                    cmd,
                    {
                        .vertexDataGpu   = vert_args,
                        .fragmentDataGpu = frag,
                        .indicesGpu = m_vertex_ptr + sizeof(Cube::kPositions) + sizeof(Cube::kUVs),
                        .indexCount = Cube::kNumIndices,
                    });
            }
        }
    }

    loon::imgui::Render(cmd);
    gpu::cmd_end_render_pass(cmd);
    gpu::cmd_signal_surface_texture(cmd);
    gpu::cmd_finalize(cmd);
    gpu::queue_submit(m_queue, cmd);

    auto frame_duration                = frame_start.elapsed();
    m_frame_time_us[m_frame_idx % 300] = (int64_t)frame_duration.as_microseconds();
    m_frame_time_ms[m_frame_idx % 300] = frame_duration.as_millseconds_f32();

    m_frame_time_average +=
        (m_frame_time_us[m_frame_idx % 300] - m_frame_time_us[(m_frame_idx + 1) % 300]) / 300;
    return true;
}
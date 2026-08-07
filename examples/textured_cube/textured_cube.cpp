/***
    Textured Cube -

    A basic spinning cube, textured. No mipmaps
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
        {1, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 0}, {0, 1},
        {1, 0}, {0, 0}, {1, 1}, {0, 1}, {1, 0}, {0, 0}, {1, 1}, {0, 1},
        {1, 0}, {0, 0}, {1, 1}, {0, 1}, {1, 0}, {0, 0}, {1, 1}, {0, 1},
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

TexturedCube::TexturedCube(const WindowState& window_state) : Example(window_state) {
    // Load shaders and create render pipeline
    ShaderModule shader         = window_state.shader_loader->load_module("textured_cube");
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
                                .usage      = UsageFlags::Sampled | UsageFlags::TransferDst,
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
    gpu::cmd_memcpy(cmd, m_vertex_ptr, m_constant_buffer, Cube::kSize);

    gpu::cmd_copy_to_texture(cmd,
                             m_constant_buffer + Cube::kSize,
                             m_color_texture,
                             BufferTextureCopyInfo{
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

TexturedCube::~TexturedCube() = default;

bool TexturedCube::update(const UpdateInfo& info) {
    // Update constant data
    auto args = reinterpret_cast<ShaderArgs*>(gpu::get_host_pointer(m_device, m_constant_buffer)) +
                (m_frame_idx % 3);
    *args = ShaderArgs{
        .camera =
            CameraInfo{
                .projection        = projection({.view_width  = (float)info.texture_size.x,
                                                 .view_height = (float)info.texture_size.y,
                                                 .y_fov       = radians_from_degrees(30.f),
                                                 .depth_far   = 0.5f}),
                .camera_from_world = transform3d::identity().translated({0, 0, -5}).to_matrix(),
            },
        .mesh =
            {
                .position = m_vertex_ptr,
                .color    = m_vertex_ptr + sizeof(Cube::kPositions),
                .world_from_mesh =
                    transform3d::identity()
                        .rotated_local(normalized({1, 0.2, 0}),
                                       radians_from_degrees((float)(m_frame_idx % 360)))
                        .to_matrix(),
            },
        .texture = m_color_view,
        .sampler = m_sampler,
    };

    auto cmd = gpu::queue_start_command_recording(m_queue);

    gpu::cmd_set_texture_heap(cmd, m_texture_heap);
    gpu::cmd_wait_for_surface_texture(cmd);
    gpu::cmd_barrier(
        cmd,
        StageFlags::FragmentTests,
        StageFlags::FragmentTests);  // We only have one depth buffer so need to stall here.
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

    gpu::cmd_set_front_face(cmd, FrontFace::CCW);
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

    gpu::cmd_signal_surface_texture(cmd);
    gpu::cmd_finalize(cmd);
    gpu::queue_submit(m_queue, cmd);

    m_frame_idx++;
    return true;
}
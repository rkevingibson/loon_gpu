/***
    Bunny

    This example shows mesh rendering with image-based lighting.
    This includes some one-off compute work to preconvolve the cube map and convert it from an
   equirectangular HDRI image.
*/

#include "bunny.h"

#include <gpu/loon_gpu.h>

#include <cassert>

#include "common/shaders.h"
#include "example.h"
#include "geometry.h"
#include "imgui/imgui_impl_loon.h"
#include "obj_parser.h"
#include "stb_image.h"

using namespace loon;
using namespace geometry;

struct CameraInfo {
    float4x4 projection        = {};
    float4x4 camera_from_world = {};
};

struct VertexArgs {
    CameraInfo camera;
    GpuMesh    mesh;
};

Bunny::Bunny(const WindowState& window_state) : Example(window_state) {
    // Load shaders and create render pipeline
    ShaderModule shader         = window_state.shader_loader->load_module("bunny");
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


    m_texture_heap = create_texture_heap(m_device,
                                         {
                                             .texture_count    = 1024,
                                             .rw_texture_count = 1024,
                                             .sampler_count    = 5,
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

    // Load model

    auto stream =
        LineReader::from_file((window_state.file_paths.asset_directory + "bunny.obj").c_str());
    Box<ObjMesh> obj_file = loon::obj_parse(stream);
    auto         mesh     = loon::cleanup_mesh(std::move(obj_file));

    // Center the model by subtracting the center of the bbox.

    float3 bbox_min = float3::infinity();
    float3 bbox_max = float3::negative_infinity();
    for (auto& p : mesh->positions) {
        bbox_min = float3::min(bbox_min, p);
        bbox_max = float3::max(bbox_max, p);
    }
    const float3 center = 0.5f * (bbox_min + bbox_max);
    for (auto& p : mesh->positions) { p = p - center; }

    m_ring_buffer = RingBuffer(m_device, 32 * 1024 * 1024, 3);

    // Copy the mesh to GPU memory, via a staging buffer.
    GpuPtr pos_staging     = m_ring_buffer.append(0, mesh->positions);
    GpuPtr uv_staging      = m_ring_buffer.append(0, mesh->texcoords);
    GpuPtr normal_staging  = m_ring_buffer.append(0, mesh->normals);
    GpuPtr indices_staging = m_ring_buffer.append(0, mesh->indices);
    // Staging buffer should have room for everything.
    assert(pos_staging && uv_staging && normal_staging && indices_staging);

    const size_t pos_size    = mesh->positions.size() * sizeof(mesh->positions[0]);
    const size_t uv_size     = mesh->texcoords.size() * sizeof(mesh->texcoords[0]);
    const size_t normal_size = mesh->normals.size() * sizeof(mesh->normals[0]);
    const size_t index_size  = mesh->indices.size() * sizeof(mesh->indices[0]);
    const size_t mesh_size   = pos_size + uv_size + normal_size + index_size;

    GpuPtr mesh_buffer = gpu::malloc(m_device, mesh_size, Memory::Gpu);
    m_mesh             = GpuMesh{
                    .positions = mesh_buffer,
                    .uvs       = mesh_buffer + pos_size,
                    .normals   = mesh_buffer + pos_size + uv_size,
    };
    m_mesh_indices = m_mesh.normals + normal_size;
    m_num_indices  = mesh->indices.size();

    auto cmd = gpu::queue_start_command_recording(m_queue);

    gpu::cmd_memcpy(cmd, mesh_buffer, pos_staging, pos_size);
    gpu::cmd_memcpy(cmd, m_mesh.uvs, uv_staging, uv_size);
    gpu::cmd_memcpy(cmd, m_mesh.normals, normal_staging, normal_size);
    gpu::cmd_memcpy(cmd, m_mesh_indices, indices_staging, index_size);
    gpu::cmd_barrier(cmd, StageFlags::Transfer, StageFlags::VertexShader);
    gpu::cmd_finalize(cmd);
    gpu::queue_submit(m_queue, cmd);

    m_depth_stencil_state = gpu::create_depth_stencil_state(
        m_device,
        DepthStencilDesc{
            .depth_mode = loon::gpu::DepthFlags::Write | loon::gpu::DepthFlags::Read,
            .depth_test = loon::gpu::Op::Greater,
        });

    // Prepare an environment map for IBL rendering:

    int    x = 0, y = 0, n = 0;
    float* image_data =
        stbi_loadf((window_state.file_paths.asset_directory + "sunny_rose_garden_2k.hdr").c_str(),
                   &x,
                   &y,
                   &n,
                   4);

    Handle<Texture> equirectangular_hdr_map =
        gpu::create_texture(m_device,
                            TextureDesc{
                                .dimensions = {(uint32_t)x, (uint32_t)y, 1},
                                .format     = Format::RGBA32Float,
                                .usage      = UsageFlags::Sampled | UsageFlags::TransferDst,
                            });

    auto color_view = gpu::add_texture_view_to_heap(m_device,
                                                    m_texture_heap,
                                                    {
                                                        .texture = equirectangular_hdr_map,
                                                        .format  = Format::RGBA32Float,
                                                    });

    m_hdri_cubemap = gpu::create_texture(m_device,
                                         TextureDesc{
                                             .type       = TextureType::TexCube,
                                             .dimensions = {512, 512, 1},
                                             .format     = Format::RGBA16Float,
                                             .usage = UsageFlags::Storage | UsageFlags::Sampled,
                                         });

    auto test_view       = gpu::add_rw_texture_view_to_heap(m_device,
                                                      m_texture_heap,
                                                            {
                                                                .texture     = m_hdri_cubemap,
                                                                .type        = TextureType::Tex2DArray,
                                                                .format      = Format::RGBA16Float,
                                                                .base_layer  = 0,
                                                                .layer_count = 6,
                                                      });
    m_debug_cubemap_face = gpu::add_texture_view_to_heap(m_device,
                                                         m_texture_heap,
                                                         {
                                                             .texture    = m_hdri_cubemap,
                                                             .type       = TextureType::Tex2D,
                                                             .format     = Format::RGBA16Float,
                                                             .base_layer = 0,
                                                         });

    auto equirectangular_to_cube_shader = window_state.shader_loader->load_module("cubemap");
    auto equirectangular_to_cube_spirv =
        get_spirv(equirectangular_to_cube_shader.get(), "equirectangular_to_cubemap");

    auto equirectangular_to_cube_pipeline =
        gpu::create_compute_pipeline(m_device,
                                     {
                                         .source      = Span(equirectangular_to_cube_spirv.data(),
                                                        equirectangular_to_cube_spirv.size()),
                                         .entry_point = "equirectangular_to_cubemap",
                                     });

    assert(equirectangular_to_cube_pipeline);

    auto sampler = gpu::add_sampler_to_heap(m_device,
                                            m_texture_heap,
                                            {
                                                .coord = SamplerCoords::Normalized,
                                            });

    cmd = gpu::queue_start_command_recording(m_queue);
    gpu::cmd_set_pipeline(cmd, equirectangular_to_cube_pipeline);

    struct alignas(64) ConvertArgs {
        uint32_t         dim_x, dim_y, pad1, pad2;
        gpu::TextureView input_texture;
        gpu::Sampler     sampler;
        gpu::TextureView output_texture;
    };

    auto args = m_ring_buffer.append(1,
                                     ConvertArgs{
                                         .dim_x          = 512,
                                         .dim_y          = 512,
                                         .input_texture  = color_view,
                                         .sampler        = sampler,
                                         .output_texture = test_view,
                                     });
    gpu::cmd_set_texture_heap(cmd, m_texture_heap);
    gpu::cmd_dispatch(cmd, args, {512 / 8, 512 / 8, 1});
    gpu::cmd_finalize(cmd);
    gpu::queue_submit(m_queue, cmd);
}

Bunny::~Bunny() {
    gpu::device_wait_for_idle(m_device);
    imgui::Shutdown();
}

bool Bunny::update(const UpdateInfo& info) {
    loon::imgui::NewFrame();
    m_frame_idx++;

    static int face_selection = 0;
    if (ImGui::SliderInt("Cubemap Face", &face_selection, 0, 5)) {
        gpu::device_wait_for_idle(m_device);

        gpu::remove_texture_view_from_heap(m_device, m_texture_heap, m_debug_cubemap_face);
        m_debug_cubemap_face =
            gpu::add_texture_view_to_heap(m_device,
                                          m_texture_heap,
                                          {
                                              .texture    = m_hdri_cubemap,
                                              .type       = TextureType::Tex2D,
                                              .format     = Format::RGBA16Float,
                                              .base_layer = (uint16_t)face_selection,
                                          });
    }
    ImGui::Image(ImTextureRef(m_debug_cubemap_face), ImVec2(512, 512));

    auto args = m_ring_buffer.append(
        m_frame_idx,
        VertexArgs{
            .camera =
                CameraInfo{
                    .projection =
                        geometry::projection({.view_width  = (float)info.texture_size.x,
                                              .view_height = (float)info.texture_size.y,
                                              .y_fov       = geometry::radians_from_degrees(30.f),
                                              .depth_far   = 0.5f}),
                    .camera_from_world = geometry::transform3d::identity()
                                             .translated({0, 0, -3})
                                             .rotated_local({0, 0, 1}, radians_from_degrees(180))
                                             .to_matrix(),
                },
            .mesh = m_mesh,
        });

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
                                   .depth_attachment =
                                       RenderAttachment{
                                           .texture     = info.depth_texture,
                                           .load_op     = loon::gpu::LoadOp::Clear,
                                           .store_op    = loon::gpu::StoreOp::Discard,
                                           .clear_color = Color(0, 0, 0, 0),
                                       },
                                   .render_area =
                                       Rect2D{
                                           .width  = info.texture_size.x,
                                           .height = info.texture_size.y,
                                       },
                               });

    gpu::cmd_set_pipeline(cmd, m_render_pipeline);
    gpu::cmd_set_front_face(cmd, FrontFace::CW);
    gpu::cmd_set_cull_mode(cmd, Cull::Back);
    gpu::cmd_set_depth_stencil_state(cmd, m_depth_stencil_state);

    gpu::cmd_draw_indexed_instanced(cmd,
                                    {
                                        .vertexDataGpu   = args,
                                        .fragmentDataGpu = 0,
                                        .indicesGpu      = m_mesh_indices,
                                        .indexCount      = m_num_indices,
                                        .type            = IndexType::UInt32,
                                    });

    loon::imgui::Render(cmd);
    gpu::cmd_end_render_pass(cmd);
    gpu::cmd_signal_surface_texture(cmd);
    gpu::cmd_finalize(cmd);

    gpu::queue_submit(m_queue, cmd);
    return true;
}
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
#include "imgui/imgui.h"
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

struct SkyboxArgs {
    float4x4 world_from_camera  = {};
    float4x4 inverse_projection = {};
    GpuPtr   skybox;
    GpuPtr   sampler;
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

    m_ring_buffer = RingBuffer(m_device, 64 * 1024 * 1024, 3);

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
                    .world_from_mesh =
            geometry::transform3d::from_axis_angle_and_origin({0, 0, 1},
                                                              geometry::radians_from_degrees(180),
                                                              float3{0, 0, 0})
                .to_matrix(),
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
            .depth_test = loon::gpu::Op::GreaterEqual,
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
                                             .dimensions = {1024, 1024, 1},
                                             .format     = Format::RGBA16Float,
                                             .usage = UsageFlags::Storage | UsageFlags::Sampled,
                                         });

    auto test_view = gpu::add_rw_texture_view_to_heap(m_device,
                                                      m_texture_heap,
                                                      {
                                                          .texture     = m_hdri_cubemap,
                                                          .type        = TextureType::Tex2DArray,
                                                          .format      = Format::RGBA16Float,
                                                          .base_layer  = 0,
                                                          .layer_count = 6,
                                                      });
    m_skybox_view  = gpu::add_texture_view_to_heap(m_device,
                                                  m_texture_heap,
                                                   {
                                                       .texture     = m_hdri_cubemap,
                                                       .type        = TextureType::TexCube,
                                                       .format      = Format::RGBA16Float,
                                                       .layer_count = 6,
                                                  });

    for (int i = 0; i < 6; ++i) {
        m_debug_cubemap_faces[i] = gpu::add_texture_view_to_heap(m_device,
                                                                 m_texture_heap,
                                                                 {
                                                                     .texture = m_hdri_cubemap,
                                                                     .type    = TextureType::Tex2D,
                                                                     .format  = Format::RGBA16Float,
                                                                     .base_layer = (uint16_t)i,
                                                                 });
    }

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

    const auto skybox_vertex_spirv =
        get_spirv(equirectangular_to_cube_shader.get(), "skybox_vertex");
    const auto skybox_fragment_spirv =
        get_spirv(equirectangular_to_cube_shader.get(), "skybox_fragment");

    m_skybox_pipeline = gpu::create_graphics_pipeline(
        m_device,
        {
            .source      = Span(skybox_vertex_spirv.data(), skybox_vertex_spirv.size()).as_bytes(),
            .entry_point = "skybox_vertex"_sv,
        },
        {
            .source = Span(skybox_fragment_spirv.data(), skybox_fragment_spirv.size()).as_bytes(),
            .entry_point = "skybox_fragment"_sv,
        },
        RasterDesc{
            .depth_format  = loon::gpu::Format::Depth32Float,
            .color_targets = {{.format = m_swapchain_format}},
        });
    assert(m_skybox_pipeline);

    m_sampler = gpu::add_sampler_to_heap(m_device,
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

    auto hdri_gpu_ptr =
        m_ring_buffer.append_raw(1, image_data, size_t(x) * y * 4 * sizeof(float), 16);
    stbi_image_free(image_data);
    auto args = m_ring_buffer.append(1,
                                     ConvertArgs{
                                         .dim_x          = 1024,
                                         .dim_y          = 1024,
                                         .input_texture  = color_view,
                                         .sampler        = m_sampler,
                                         .output_texture = test_view,
                                     });

    gpu::cmd_copy_to_texture(cmd,
                             hdri_gpu_ptr,
                             equirectangular_hdr_map,
                             {
                                 .image_extent = {.x = (uint32_t)x, .y = (uint32_t)y, .z = 1},
                             });
    gpu::cmd_barrier(cmd, StageFlags::Transfer, StageFlags::Compute);
    gpu::cmd_set_texture_heap(cmd, m_texture_heap);
    gpu::cmd_dispatch(cmd, args, {1024 / 8, 1024 / 8, 1});
    gpu::cmd_finalize(cmd);
    gpu::queue_submit(m_queue, cmd);
}

Bunny::~Bunny() {
    gpu::device_wait_for_idle(m_device);
    imgui::Shutdown();
}

static void show_cubemap(TextureView* face_views, int size) {
    ImVec2 dim = ImVec2((float)size, (float)size);
    ImGui::PushStyleVarY(ImGuiStyleVar_ItemSpacing, 0);
    ImGui::Dummy(dim);
    ImGui::SameLine(0, 0);
    ImGui::Image(ImTextureRef(face_views[2]), dim);
    ImGui::Image(ImTextureRef(face_views[1]), dim);
    ImGui::SameLine(0, 0);
    ImGui::Image(ImTextureRef(face_views[4]), dim);
    ImGui::SameLine(0, 0);
    ImGui::Image(ImTextureRef(face_views[0]), dim);
    ImGui::SameLine(0, 0);
    ImGui::Image(ImTextureRef(face_views[5]), dim);
    ImGui::Dummy(dim);
    ImGui::SameLine(0, 0);
    ImGui::Image(ImTextureRef(face_views[3]), dim);
    ImGui::PopStyleVar();
}

bool Bunny::update(const UpdateInfo& info) {
    loon::imgui::NewFrame();
    m_frame_idx++;

    static float rotations[3] = {0, 0, 0};
    ImGui::SliderFloat3("Camera Rotations XYZ", rotations, 0.0f, 180.0);

    const auto camera_from_world =
        geometry::transform3d::identity()
            .translated({0, 0, -3})
            .rotated_local({1, 0, 0}, radians_from_degrees(rotations[0]))
            .rotated_local({0, 1, 0}, radians_from_degrees(rotations[1]))
            .rotated_local({0, 0, 1}, radians_from_degrees(rotations[2]));
    const auto world_from_camera = camera_from_world.inverse();

    const auto projection = geometry::projection({.view_width  = (float)info.texture_size.x,
                                                  .view_height = (float)info.texture_size.y,
                                                  .y_fov     = geometry::radians_from_degrees(30.f),
                                                  .depth_far = 1.f});
    const auto inverse_projection =
        projection.inverse();  // NOTE: This doesn't work - projection matrix isn't invertible.


    auto args = m_ring_buffer.append(m_frame_idx,
                                     VertexArgs{
                                         .camera =
                                             CameraInfo{
                                                 .projection        = projection,
                                                 .camera_from_world = camera_from_world.to_matrix(),
                                             },
                                         .mesh = m_mesh,
                                     });

    auto skybox_args = m_ring_buffer.append(m_frame_idx,
                                            SkyboxArgs{
                                                .world_from_camera  = world_from_camera.to_matrix(),
                                                .inverse_projection = inverse_projection,
                                                .skybox             = m_skybox_view,
                                                .sampler            = m_sampler,
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
    gpu::cmd_set_texture_heap(cmd, m_texture_heap);
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

    // Render skybox
    gpu::cmd_set_pipeline(cmd, m_skybox_pipeline);
    gpu::cmd_set_cull_mode(cmd, Cull::None);
    gpu::cmd_draw(cmd, skybox_args, 0, 3, 1);

    loon::imgui::Render(cmd);
    gpu::cmd_end_render_pass(cmd);
    gpu::cmd_signal_surface_texture(cmd);
    gpu::cmd_finalize(cmd);

    gpu::queue_submit(m_queue, cmd);
    return true;
}
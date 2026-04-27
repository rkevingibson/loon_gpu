#include "particle_emitter.h"

#include <cassert>

#include "geometry.h"
#include "gpu/loon_gpu.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_loon.h"
#include "shaders.h"

using namespace loon;
struct Particle {
    geometry::float3 position;
    float            life;
    geometry::float3 velocity;
    float            size;
    geometry::float4 color;
};

struct DeadList {
    GpuPtr indices;
};

struct SimGpu {
    ParticleSimOptions options;
    DeadList           dead_list;
    GpuPtr             particles;
    GpuPtr             indirect_args;
    GpuPtr             alive_list;
};

struct CameraDataGpu {
    geometry::float4x4 projection        = {};
    geometry::float4x4 camera_from_world = {};
};

struct DrawSimArgs {
    CameraDataGpu    camera;
    geometry::float3 camera_right_worldspace;
    geometry::float3 camera_up_worldspace;
    GpuPtr           particles;
    GpuPtr           alive_list;
};

static constexpr uint32_t kMaxNumParticles = 512;

static Format select_surface_format(const loon::gpu::SurfaceCapabilities& surface_capabilities) {
    for (Format f : surface_capabilities.formats) {
        if (f == loon::gpu::Format::RGBA8UnormSrgb || f == loon::gpu::Format::BGRA8UnormSrgb) {
            // Choose 8 bit srgb if we have it
            return f;
        }
    }
    return surface_capabilities.formats[0];
}

ParticleEmitter::ParticleEmitter(const WindowState& window_state) {
    m_device = loon::gpu::create_device({
        .gpu_preference         = GpuPreference::Discrete,
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

    recreate_swapchain(window_state.width, window_state.height);

    // Load shaders and create render pipeline
    ShaderModule shader               = window_state.shader_loader->load_module("particle_emitter");
    const auto   get_compute_pipeline = [&](loon::gpu::Span<const char> name) -> Handle<Pipeline> {
        const auto spirv = get_spirv(shader.get(), name.data());
        return gpu::create_compute_pipeline(
            m_device,
            {
                  .spirv       = Span(spirv.data(), spirv.size()).as_bytes(),
                  .entry_point = name,
            });
    };

    m_reset_sim_pipeline  = get_compute_pipeline("reset_sim"_sv);
    m_emitter_pipeline    = get_compute_pipeline("emitter"_sv);
    m_update_sim_pipeline = get_compute_pipeline("update_particle_sim"_sv);

    auto vertex_spirv   = get_spirv(shader.get(), "vertex_main");
    auto fragment_spirv = get_spirv(shader.get(), "fragment_main");
    m_render_particle_pipeline = gpu::create_graphics_pipeline(m_device,
        {
            .spirv       = Span(vertex_spirv.data(), vertex_spirv.size()),
            .entry_point = "vertex_main"_sv,
        },
        {
            .spirv       = Span(fragment_spirv.data(), fragment_spirv.size()),
            .entry_point = "fragment_main"_sv,
        },
        RasterDesc{
            .depth_format  = loon::gpu::Format::Depth32Float,
            .color_targets = ColorTarget{
                .format = m_swapchain_format,
                .blendstate = BlendDesc{
                    .color_op = Blend::Add,
                    .src_color_factor = Factor::SrcAlpha,
                    .dst_color_factor = Factor::OneMinusSrcAlpha,
                    .src_alpha_factor = Factor::One,
                    .dst_alpha_factor = Factor::OneMinusSrcAlpha,
                },
            },
        });
    assert(m_update_sim_pipeline.h != 0);
    assert(m_render_particle_pipeline.h != 0);

    m_queue = gpu::get_queue(m_device);

    m_depth_stencil_state
        = gpu::create_depth_stencil_state(m_device,
                                          DepthStencilDesc{
                                              .depth_mode = loon::gpu::DepthFlags::Read,
                                              .depth_test = Op::Greater,
                                          });

    // Particle sim initialization:
    m_sim.particle_buffer = gpu::malloc(m_device, sizeof(Particle) * kMaxNumParticles, Memory::Gpu);
    m_sim.dead_list
        = gpu::malloc(m_device, sizeof(uint32_t) * kMaxNumParticles + sizeof(int32_t), Memory::Gpu);
    m_sim.alive_list = gpu::malloc(m_device, sizeof(uint32_t) * kMaxNumParticles, Memory::Gpu);
    m_sim.options    = {
           .spawn_pos         = geometry::float3(0, 0, 0),
           .spawn_radius      = 0.5f,
           .lifetime          = 1.f,
           .particle_size     = 0.1f,
           .delta_t           = 1.0 / 60.f,
           .max_num_particles = kMaxNumParticles,
           .particles_to_emit = 10,
           .rng_seed          = (uint32_t)rand(),
    };

    m_ring_buffer = loon::RingBuffer(m_device, 16 * 1024 * 1024, 3);
    m_frame_idx   = 0;

    // Imgui setup

    m_texture_heap = gpu::create_texture_heap(m_device,
                                              {
                                                  .texture_count = 1024,
                                                  .sampler_count = 1,
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


    // Reset the particle sim
    auto cmd = gpu::queue_start_command_recording(m_queue);
    gpu::cmd_set_pipeline(cmd, m_reset_sim_pipeline);
    GpuPtr args = m_ring_buffer.append(m_frame_idx,
                         SimGpu{.options   = m_sim.options,
                                .dead_list = {
                                    .indices = m_sim.dead_list,
                                },
                                .particles =m_sim.particle_buffer,
                            });

    gpu::cmd_dispatch(cmd, args, {kMaxNumParticles / 64, 1, 1});
    gpu::cmd_barrier(cmd, StageFlags::Compute, StageFlags::Compute);
    gpu::cmd_finalize(cmd);
    gpu::queue_submit(m_queue, cmd);

    gpu::device_wait_for_idle(m_device);
}

ParticleEmitter::~ParticleEmitter() {
    gpu::device_wait_for_idle(m_device);
    loon::imgui::Shutdown();
    m_ring_buffer = RingBuffer();
    destroy_device(m_device);
}


void ParticleEmitter::recreate_swapchain(uint32_t width, uint32_t height) {
    gpu::device_wait_for_idle(m_device);
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

    // Recreate depth buffer as well
    if (m_depth_texture) { gpu::free(m_device, m_depth_texture); }
    m_depth_texture
        = gpu::create_texture(m_device,
                              {
                                  .type       = TextureType::Tex2D,
                                  .dimensions = {.x = width, .y = height, .z = 1},
                                  .format     = loon::gpu::Format::Depth32Float,
                                  .usage      = loon::gpu::UsageFlags::DepthStencilAttachment,
                              });
}

void ParticleEmitter::Update(const WindowState& window) {
    auto surface_texture = gpu::get_current_texture(m_device);
    if (surface_texture.status == SurfaceStatus::OutOfDate
        || surface_texture.status == SurfaceStatus::Suboptimal) {
        recreate_swapchain(window.width, window.height);
        return;
    } else if (surface_texture.status == SurfaceStatus::Error) {
        return;
    }

    // CPU-side update, construct GPU arguments.

    loon::imgui::NewFrame();

    ImGui::Begin("Simulation Parameters");
    ImGui::SliderFloat3("Spawn Position", &m_sim.options.spawn_pos.x, -1.f, 1.f);
    ImGui::SliderFloat("Spawn radius", &m_sim.options.spawn_radius, 0.f, 1.f);
    ImGui::SliderFloat("Particle size", &m_sim.options.particle_size, 0.f, 1.f);
    ImGui::SliderFloat("Particle lifetime", &m_sim.options.lifetime, 0.0f, 1.f);

    ImGui::End();
    m_frame_idx++;

    m_sim.options.rng_seed = rand();
    GpuPtr indirect_args   = m_ring_buffer.append(m_frame_idx,
                                                loon::gpu::DrawIndexedIndirectGpuArgs{
                                                      .index_count    = 6,
                                                      .instance_count = 0,
                                                      .first_index    = 0,
                                                      .vertex_offset  = 0,
                                                      .first_instance = 0,
                                                });

    GpuPtr sim_args = m_ring_buffer.append(m_frame_idx,
                         SimGpu{.options   = m_sim.options,
                                .dead_list = {
                                    .indices = m_sim.dead_list,
                                },
                                .particles = m_sim.particle_buffer,
                                .indirect_args = indirect_args,
                                .alive_list = m_sim.alive_list,
                            });

    GpuPtr vertex_args = m_ring_buffer.append(m_frame_idx, DrawSimArgs {
        .camera = {
            .projection =  geometry::projection({.view_width  = (float)window.width,
                                            .view_height = (float)window.height,
                                            .y_fov       = geometry::radians_from_degrees(30.f),
                                            .depth_far   = 0.5f}),
            .camera_from_world = geometry::transform3d::identity().translated({0, 0, -5}).to_matrix(),              
        },
        .camera_right_worldspace = {1,0,0},
        .camera_up_worldspace = {0,1,0},
        .particles = m_sim.particle_buffer,
        .alive_list = m_sim.alive_list,
    });

    uint16_t indices[]   = {0, 1, 2, 2, 1, 3};
    GpuPtr   indices_ptr = m_ring_buffer.append(m_frame_idx, indices);

    // Rendering here.
    auto cmd = gpu::queue_start_command_recording(m_queue);


    gpu::cmd_set_pipeline(cmd, m_emitter_pipeline);
    gpu::cmd_dispatch(cmd, sim_args, Dimension3D{.x = kMaxNumParticles / 64, .y = 1, .z = 1});
    gpu::cmd_barrier(cmd, StageFlags::Compute, StageFlags::Compute);

    gpu::cmd_set_pipeline(cmd, m_update_sim_pipeline);
    gpu::cmd_dispatch(cmd, sim_args, Dimension3D{.x = kMaxNumParticles / 64, .y = 1, .z = 1});
    gpu::cmd_barrier(cmd, StageFlags::Compute, StageFlags::IndirectArguments);

    // Render particles
    gpu::cmd_wait_for_surface_texture(cmd);
    // We only have one depth buffer so we need to stall here
    gpu::cmd_barrier(cmd, StageFlags::FragmentTests, StageFlags::FragmentTests);

    gpu::cmd_begin_render_pass(cmd,{
        .color_attachments = RenderAttachment {
            .texture = surface_texture.texture,
            .load_op = LoadOp::Clear,
            .store_op = StoreOp::Store,
            .clear_color = Color(0,0,0,0),
        },
        .depth_attachment = RenderAttachment {
            .texture = m_depth_texture,
            .load_op = LoadOp::Clear,
            .store_op = StoreOp::Discard,
            .clear_color = Color(0,0,0,0),
        },
        .render_area = {.width = m_swapchain_width, .height = m_swapchain_height,},
    });
    gpu::cmd_set_depth_stencil_state(cmd, m_depth_stencil_state);
    gpu::cmd_set_pipeline(cmd, m_render_particle_pipeline);
    gpu::cmd_draw_indexed_instanced_indirect(cmd,
                                             {
                                                 .vertexDataGpu   = vertex_args,
                                                 .fragmentDataGpu = 0,
                                                 .indicesGpu      = indices_ptr,
                                                 .argsGpu         = indirect_args,
                                             });

    gpu::cmd_set_texture_heap(cmd, m_texture_heap);
    loon::imgui::Render(cmd);
    gpu::cmd_end_render_pass(cmd);
    gpu::cmd_signal_surface_texture(cmd);
    gpu::cmd_finalize(cmd);
    gpu::queue_submit(m_queue, cmd);

    const auto status = gpu::present(m_device, m_queue);
    if (status == SurfaceStatus::OutOfDate || status == SurfaceStatus::Suboptimal) {
        recreate_swapchain(window.width, window.height);
    }

    gpu::queue_process_events(m_queue);
}
#include "particle_emitter.h"

#include <cassert>

#include "common/gpu_args.h"
#include "geometry.h"
#include "gpu/loon_gpu.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_loon.h"
#include "shaders.h"

struct Particle {
    geometry::float3 position;
    float            life;
    geometry::float3 velocity;
    float            size;
    geometry::float4 color;
};

struct DeadList {
    int32_t size;
    GpuPtr  indices;
};

static constexpr uint32_t kMaxNumParticles = 128;

static Format select_surface_format(const loon::gpu::SurfaceCapabilities& surface_capabilities) {
    for (Format f : surface_capabilities.formats) {
        if (f == loon::gpu::Format::RGBA8UnormSrgb) {
            // Choose 8 bit srgb if we have it
            return f;
        }
    }
    return surface_capabilities.formats[0];
}

ParticleEmitter::ParticleEmitter(const WindowState& window_state) {
    m_device = loon::gpu::Device::create({
        .gpu_preference         = GpuPreference::Discrete,
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

    recreate_swapchain(window_state.width, window_state.height);

    // Load shaders and create render pipeline
    ShaderModule shader = window_state.shader_loader->load_module("particle_emitter.slang");
    const auto   get_compute_pipeline = [&](loon::gpu::Span<const char> name) -> Handle<Pipeline> {
        const auto spirv = get_spirv(shader.get(), name.data());
        return m_device.create_compute_pipeline({
              .spirv       = Span(spirv.data(), spirv.size()).as_bytes(),
              .entry_point = name,
        });
    };

    m_reset_sim_pipeline  = get_compute_pipeline("reset_sim"_sv);
    m_emitter_pipeline    = get_compute_pipeline("emitter"_sv);
    m_update_sim_pipeline = get_compute_pipeline("update_particle_sim"_sv);

    auto vertex_spirv          = get_spirv(shader.get(), "vertex_main");
    auto fragment_spirv        = get_spirv(shader.get(), "fragment_main");
    m_render_particle_pipeline = m_device.create_graphics_pipeline(
        {
            .spirv       = Span(vertex_spirv.data(), vertex_spirv.size()).as_bytes(),
            .entry_point = "vertex_main"_sv,
        },
        {
            .spirv       = Span(fragment_spirv.data(), fragment_spirv.size()).as_bytes(),
            .entry_point = "fragment_main"_sv,
        },
        RasterDesc{
            .cull          = Cull::CW,
            .depth_format  = loon::gpu::Format::Depth32Float,
            .color_targets = ColorTarget{.format = m_swapchain_format},
        });
    assert(m_update_sim_pipeline.h != 0);
    assert(m_render_particle_pipeline.h != 0);

    m_queue = m_device.get_queue();

    m_depth_stencil_state = m_device.create_depth_stencil_state(DepthStencilDesc{
        .depth_mode = loon::gpu::DepthFlags::Write | loon::gpu::DepthFlags::Read,
        .depth_test = Op::Greater,
    });

    // Particle sim initialization:
    m_sim.particle_buffer = m_device.malloc(sizeof(Particle) * kMaxNumParticles, Memory::Gpu);
    m_sim.dead_list       = m_device.malloc(sizeof(uint32_t) * kMaxNumParticles, Memory::Gpu);
    m_sim.options         = {
                .spawn_pos         = geometry::float3(0, 0, 0),
                .spawn_radius      = 0.5f,
                .lifetime          = 1.f,
                .particle_size     = 0.1f,
                .delta_t           = 0,
                .max_num_particles = kMaxNumParticles,
                .particles_to_emit = 10,
    };

    // Imgui setup

    m_texture_heap = m_device.create_texture_heap(1024);
    loon::imgui::Init({
        .device                    = &m_device,
        .queue                     = m_queue,
        .texture_heap              = m_texture_heap,
        .num_frames_in_flight      = 3,
        .render_target_format      = m_swapchain_format,
        .depth_stencil_view_format = loon::gpu::Format::Depth32Float,
        .shader_loader             = window_state.shader_loader.get(),
    });


    // Reset the particle sim
    auto cmd = m_device.start_command_recording(m_queue);
    cmd.set_pipeline(m_reset_sim_pipeline);
    // cmd.dispatch(, );
}

ParticleEmitter::~ParticleEmitter() {
    m_device.wait_for_device_idle();
    loon::imgui::Shutdown();
}


void ParticleEmitter::recreate_swapchain(uint32_t width, uint32_t height) {
    m_device.wait_for_device_idle();
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

    // Recreate depth buffer as well
    if (m_depth_view.h) {
        m_device.free(m_depth_view);
        m_device.free(m_depth_texture);
    }
    m_depth_texture = m_device.create_texture({
        .type       = TextureType::Tex2D,
        .dimensions = {.x = width, .y = height, .z = 1},
        .format     = loon::gpu::Format::Depth32Float,
        .usage      = loon::gpu::UsageFlags::DepthStencilAttachment,
    });

    m_depth_view = m_device.create_texture_view(m_depth_texture,
                                                TextureViewDesc{
                                                    .format = loon::gpu::Format::Depth32Float,
                                                });
}

void ParticleEmitter::Update(const WindowState& window) {
    auto surface_texture = m_device.get_current_texture();
    if (surface_texture.status == SurfaceStatus::OutOfDate
        || surface_texture.status == SurfaceStatus::Suboptimal) {
        recreate_swapchain(window.width, window.height);
        return;
    } else if (surface_texture.status == SurfaceStatus::Error) {
        return;
    }

    auto swapchain_view
        = m_device.create_texture_view(surface_texture.texture,
                                       TextureViewDesc{.format = m_swapchain_format});


    // CPU-side update, construct GPU arguments.

    loon::imgui::NewFrame();


    ImGui::Begin("Simulation Parameters");
    ImGui::SliderFloat3("Spawn Position", &m_sim.options.spawn_pos.x, -1.f, 1.f);
    ImGui::SliderFloat("Spawn radius", &m_sim.options.spawn_radius, 0.f, 1.f);

    ImGui::End();

    // Rendering here.
    auto cmd = m_device.start_command_recording(m_queue);
    cmd.barrier(RasterColorOut,
                Compute | PixelShader | RasterColorOut,
                {TextureTransition{
                     .texture    = surface_texture.texture,
                     .old_layout = Layout::DontCare,
                     .new_layout = Layout::Attachment,
                 },
                 TextureTransition{
                     .texture    = m_depth_texture,
                     .old_layout = Layout::DontCare,
                     .new_layout = Layout::Attachment,
                 }});

    cmd.set_pipeline(m_update_sim_pipeline);
    // cmd.dispatch(0, Dimension3D{.x = 1, .y = 1, .z = 1});
    cmd.barrier(Compute, VertexShader);

    // Render particles
    cmd.begin_render_pass({
        .color_attachments = RenderAttachment {
            .texture_view = swapchain_view,
            .load_op = LoadOp::Clear,
            .store_op = StoreOp::Store,
            .clear_color = Color(0,0,0,0),
        },
        .depth_attachment = RenderAttachment {
            .texture_view = m_depth_view,
            .load_op = LoadOp::Clear,
            .store_op = StoreOp::Discard,
            .clear_color = Color(0,0,0,0),
        },
        .render_area = {.width = m_swapchain_width, .height = m_swapchain_height,},
    });
    cmd.set_depth_stencil_State(m_depth_stencil_state);
    cmd.set_pipeline(m_render_particle_pipeline);

    cmd.set_texture_heap(m_texture_heap);
    loon::imgui::Render(cmd);
    cmd.end_render_pass();

    cmd.barrier(RasterColorOut,
                RasterColorOut,
                TextureTransition{
                    .texture    = surface_texture.texture,
                    .old_layout = Layout::Attachment,
                    .new_layout = Layout::Present,
                });
    m_device.submit(m_queue,
                    cmd,
                    SemaphoreInfo{
                        .semaphore = surface_texture.acquire_semaphore,
                        .stage     = PixelShader,
                    });

    const auto status = m_device.present(m_queue);
    if (status == SurfaceStatus::OutOfDate || status == SurfaceStatus::Suboptimal) {
        recreate_swapchain(window.width, window.height);
    }

    m_device.on_submitted_work_completed(m_queue,
                                         [&, swapchain_view]() { m_device.free(swapchain_view); });
    m_device.process_events(m_queue);
}
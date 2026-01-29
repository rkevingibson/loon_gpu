#include "particle_emitter.h"

#include <cassert>

#include "gpu/loon_gpu.h"
#include "shaders.h"

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
    ShaderModule shader         = window_state.shader_loader->load_module("particle_emitter.slang");
    const auto   sim_spirv      = get_spirv(shader.get(), "update_particle_sim");
    const auto   vertex_spirv   = get_spirv(shader.get(), "vertex_main");
    const auto   fragment_spirv = get_spirv(shader.get(), "fragment_main");

    m_update_sim_pipeline = m_device.create_compute_pipeline({
        .spirv       = Span(sim_spirv.data(), sim_spirv.size()).as_bytes(),
        .entry_point = "update_particle_sim"_sv,
    });

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
}

ParticleEmitter::~ParticleEmitter() {}

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



    // Rendering here.
    auto cmd = m_device.start_command_recording(m_queue);
    cmd.barrier(StageFlags(Host | RasterColorOut),
                StageFlags(Compute | PixelShader | RasterColorOut),
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
    cmd.dispatch(0, Dimension3D{.x = 1, .y = 1, .z = 1});
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
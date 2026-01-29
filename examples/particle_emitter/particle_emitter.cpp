#include "particle_emitter.h"

#include <cassert>

#include "gpu/loon_gpu.h"
#include "shaders.h"

static FORMAT select_surface_format(const loon::gpu::SurfaceCapabilities& surface_capabilities) {
    for (FORMAT f : surface_capabilities.formats) {
        if (f == loon::gpu::FORMAT_RGBA8UnormSrgb) {
            // Choose 8 bit srgb if we have it
            return f;
        }
    }
    return surface_capabilities.formats[0];
}

ParticleEmitter::ParticleEmitter(const WindowState& window_state) {
    m_device = loon::gpu::Device::create({
        .gpu_preference         = loon::gpu::GpuPreference_Discrete,
        .native_window_handle   = window_state.native_window_handle,
        .native_instance_handle = window_state.native_instance_handle,
        .log_callback           = log_callback,
        .log_userdata           = nullptr,
        .log_level              = LogLevel_Debug,
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
            .cull          = CULL_CW,
            .depth_format  = loon::gpu::FORMAT_Depth32Float,
            .color_targets = ColorTarget{.format = m_swapchain_format},
        });
    assert(m_update_sim_pipeline.h != 0);
    assert(m_render_particle_pipeline.h != 0);

    m_queue = m_device.get_queue();

    m_depth_stencil_state = m_device.create_depth_stencil_state(DepthStencilDesc{
        .depth_mode = DEPTH_FLAGS(loon::gpu::DEPTH_WRITE | loon::gpu::DEPTH_READ),
        .depth_test = loon::gpu::OP_GREATER,
    });
}

ParticleEmitter::~ParticleEmitter() {}

void ParticleEmitter::recreate_swapchain(uint32_t width, uint32_t height) {
    m_device.wait_for_device_idle();
    m_device.unconfigure_surface();
    m_device.configure_surface({
        .format       = m_swapchain_format,
        .usages       = loon::gpu::USAGE_COLOR_ATTACHMENT,
        .width        = width,
        .height       = height,
        .present_mode = PRESENT_MODE_FIFO,
    });
    m_swapchain_width  = width;
    m_swapchain_height = height;

    // Recreate depth buffer as well
    if (m_depth_view.h) {
        m_device.free(m_depth_view);
        m_device.free(m_depth_texture);
    }
    m_depth_texture = m_device.create_texture({
        .type       = TEXTURE_2D,
        .dimensions = {.x = width, .y = height, .z = 1},
        .format     = loon::gpu::FORMAT_Depth32Float,
        .usage      = loon::gpu::USAGE_DEPTH_STENCIL_ATTACHMENT,
    });

    m_depth_view = m_device.create_texture_view(m_depth_texture,
                                                TextureViewDesc{
                                                    .format = loon::gpu::FORMAT_Depth32Float,
                                                });
}

void ParticleEmitter::Update(const WindowState& window) {
    auto surface_texture = m_device.get_current_texture();
    if (surface_texture.status == SURFACE_STATUS_OUT_OF_DATE
        || surface_texture.status == SURFACE_STATUS_SUBOPTIMAL) {
        recreate_swapchain(window.width, window.height);
        return;
    } else if (surface_texture.status == SURFACE_STATUS_ERROR) {
        return;
    }

    auto swapchain_view
        = m_device.create_texture_view(surface_texture.texture,
                                       TextureViewDesc{.format = m_swapchain_format});


    // CPU-side update, construct GPU arguments.



    // Rendering here.
    auto cmd = m_device.start_command_recording(m_queue);
    cmd.barrier(STAGE_FLAGS(STAGE_HOST | STAGE_RASTER_COLOR_OUT),
                STAGE_FLAGS(STAGE_COMPUTE | STAGE_PIXEL_SHADER | STAGE_RASTER_COLOR_OUT),
                {TextureTransition{
                     .texture    = surface_texture.texture,
                     .old_layout = LAYOUT_DONT_CARE,
                     .new_layout = LAYOUT_ATTACHMENT,
                 },
                 TextureTransition{
                     .texture    = m_depth_texture,
                     .old_layout = LAYOUT_DONT_CARE,
                     .new_layout = LAYOUT_ATTACHMENT,
                 }});

    cmd.set_pipeline(m_update_sim_pipeline);
    cmd.dispatch(0, Dimension3D{.x = 1, .y = 1, .z = 1});
    cmd.barrier(STAGE_COMPUTE, STAGE_VERTEX_SHADER);

    // Render particles
    cmd.begin_render_pass({
        .color_attachments = RenderAttachment {
            .texture_view = swapchain_view,
            .load_op = LOAD_OP_CLEAR,
            .store_op = STORE_OP_STORE,
            .clear_color = Color(0,0,0,0),
        },
        .depth_attachment = RenderAttachment {
            .texture_view = m_depth_view,
            .load_op = LOAD_OP_CLEAR,
            .store_op = STORE_OP_DISCARD,
            .clear_color = Color(0,0,0,0),
        },
        .render_area = {.width = m_swapchain_width, .height = m_swapchain_height,},
    });
    cmd.set_depth_stencil_State(m_depth_stencil_state);
    cmd.set_pipeline(m_render_particle_pipeline);

    cmd.end_render_pass();

    cmd.barrier(STAGE_RASTER_COLOR_OUT,
                STAGE_RASTER_COLOR_OUT,
                TextureTransition{
                    .texture    = surface_texture.texture,
                    .old_layout = LAYOUT_ATTACHMENT,
                    .new_layout = LAYOUT_PRESENT,
                });
    m_device.submit(m_queue,
                    cmd,
                    SemaphoreInfo{
                        .semaphore = surface_texture.acquire_semaphore,
                        .stage     = STAGE_PIXEL_SHADER,
                    });

    const auto status = m_device.present(m_queue);
    if (status == SURFACE_STATUS_OUT_OF_DATE || status == SURFACE_STATUS_SUBOPTIMAL) {
        recreate_swapchain(window.width, window.height);
    }

    m_device.on_submitted_work_completed(m_queue,
                                         [&, swapchain_view]() { m_device.free(swapchain_view); });
    m_device.process_events(m_queue);
}
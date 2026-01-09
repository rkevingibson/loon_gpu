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

using namespace loon::gpu;

static void LoonLogCallback(loon::gpu::LogLevel         lvl,
                            loon::gpu::Span<const char> message,
                            void*                       userdata) {
    fprintf(stderr, "%.*s", static_cast<int>(message.size()), message.data());
}

static FORMAT select_surface_format(const loon::gpu::SurfaceCapabilities& surface_capabilities) {
    for (FORMAT f : surface_capabilities.formats) {
        if (f == loon::gpu::FORMAT_RGBA8UnormSrgb) {
            // Choose 8 bit srgb if we have it
            return f;
        }
    }
    return surface_capabilities.formats[0];
}


HelloTriangle::HelloTriangle(const WindowState& window_state) {
    m_device = loon::gpu::Device::create({
        .gpu_preference         = loon::gpu::GpuPreference_Discrete,
        .native_window_handle   = window_state.native_window_handle,
        .native_instance_handle = window_state.native_instance_handle,
        .log_callback           = LoonLogCallback,
        .log_userdata           = nullptr,
        .log_level              = LogLevel_Debug,
        .alloc_callback         = nullptr,
        .alloc_userdata         = nullptr,
    });

    // TODO: Surface creation/configuration:
    auto surface_capabilities = m_device.get_surface_capabilities();
    m_swapchain_format        = select_surface_format(surface_capabilities);

    m_device.configure_surface({
        .format       = m_swapchain_format,
        .usages       = loon::gpu::USAGE_COLOR_ATTACHMENT,
        .width        = window_state.width,
        .height       = window_state.height,
        .present_mode = PRESENT_MODE_FIFO,
    });

    // Load shaders and create render pipeline
    ShaderModule shader         = window_state.shader_loader->load_module("hello_triangle.slang");
    const auto   vertex_spirv   = get_spirv(shader.get(), "vertexMain");
    const auto   fragment_spirv = get_spirv(shader.get(), "fragmentMain");

    m_render_pipeline = m_device.createGraphicsPipeline(
        {
            .spirv       = Span(vertex_spirv.data(), vertex_spirv.size()).as_bytes(),
            .entry_point = "vertexMain"_sv,
        },
        {
            .spirv       = Span(fragment_spirv.data(), fragment_spirv.size()).as_bytes(),
            .entry_point = "fragmentMain"_sv,
        },
        RasterDesc{.colorTargets = {{.format = m_swapchain_format}}});

    assert(m_render_pipeline.h != 0);

    m_queue = m_device.getQueue();
}

HelloTriangle::~HelloTriangle() {
    m_device.freePipeline(m_render_pipeline);
    m_device.unconfigure_surface();
}

void HelloTriangle::Update(const WindowState& window) {
    auto surface_texture = m_device.get_current_texture();
    if (surface_texture.status == loon::gpu::SurfaceTextureInfo::STATUS_OUT_OF_DATE) {
        m_device.unconfigure_surface();
        m_device.configure_surface({
            .format       = m_swapchain_format,
            .usages       = loon::gpu::USAGE_COLOR_ATTACHMENT,
            .width        = window.width,
            .height       = window.height,
            .present_mode = PRESENT_MODE_FIFO,
        });

        return;
    } else if (surface_texture.status == loon::gpu::SurfaceTextureInfo::STATUS_ERROR) {
        return;
    }

    auto swapchain_view = m_device.createTextureView(m_texture_heap,
                                                     surface_texture.texture,
                                                     TextureViewDesc{
                                                         .format     = m_swapchain_format,
                                                         .baseMip    = 0,
                                                         .mipCount   = 1,
                                                         .baseLayer  = 0,
                                                         .layerCount = 1,
                                                     });

    auto commandBuffer = m_device.startCommandRecording(m_queue);

    // WGPURenderPassColorAttachment color_attachment{
    //     .nextInChain   = nullptr,
    //     .view          = swapchain_view,
    //     .depthSlice    = WGPU_DEPTH_SLICE_UNDEFINED,
    //     .resolveTarget = nullptr,
    //     .loadOp        = WGPULoadOp_Clear,
    //     .storeOp       = WGPUStoreOp_Store,
    //     .clearValue    = WGPUColor{.r = 0.f, .g = 0.f, .b = 0.f, .a = 1.f},
    // };
    // WGPURenderPassDescriptor render_pass_descriptor{
    //     .nextInChain            = nullptr,
    //     .label                  = "Render pass"_wsv,
    //     .colorAttachmentCount   = 1,
    //     .colorAttachments       = &color_attachment,
    //     .depthStencilAttachment = nullptr,
    //     .occlusionQuerySet      = nullptr,
    //     .timestampWrites        = nullptr,
    // };

    commandBuffer.beginRenderPass({
        // TODO: Need to fill in stuff here.
    });

    commandBuffer.setPipeline(m_render_pipeline);
    commandBuffer.draw(0, 0, 3, 1);

    commandBuffer.endRenderPass();
    m_device.submit(m_queue,
                    commandBuffer,
                    SemaphoreInfo{
                        .semaphore = surface_texture.acquire_semaphore,
                        .stage     = loon::gpu::STAGE_RASTER_COLOR_OUT,
                    });

    m_device.present();


    m_device.freeTextureView(m_texture_heap, swapchain_view);
    // wgpuTextureRelease(surface_texture.texture);
}
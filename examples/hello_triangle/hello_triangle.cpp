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


// static WGPUSurface create_surface(WGPUInstance instance, const WindowState& window_state) {
// #if _WIN32

//     WGPUSurfaceSourceWindowsHWND surface_source = WGPU_SURFACE_SOURCE_WINDOWS_HWND_INIT;
//     surface_source.hwnd      = reinterpret_cast<void*>(window_state.native_window_handle);
//     surface_source.hinstance = reinterpret_cast<void*>(window_state.native_instance_handle);
// #elif __linux__
// #    error "Unsupported platform currently"
// #elif __APPLE__
//     WGPUSurfaceSourceMetalLayer surface_source = WGPU_SURFACE_SOURCE_METAL_LAYER_INIT;
//     surface_source.layer = reinterpret_cast<void*>(window_state.native_window_handle);
// #endif


//     WGPUSurfaceDescriptor surface_descriptor{
//         .nextInChain = &surface_source.chain,
//         .label       = "Hello Triangle Surface"_wsv,
//     };

//     WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surface_descriptor);
//     return surface;
// }

// static WGPUTextureFormat select_surface_format(
//     const WGPUSurfaceCapabilities& surface_capabilities) {
//     for (size_t format_idx = 0; format_idx < surface_capabilities.formatCount; ++format_idx) {
//         if (surface_capabilities.formats[format_idx] == WGPUTextureFormat_BGRA8UnormSrgb) {
//             // Choose 8 bit srgb if we have it
//             return surface_capabilities.formats[format_idx];
//         }
//     }
//     return surface_capabilities.formats[0];
// }


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

    // WGPUSurfaceCapabilities surface_capabilities = WGPU_SURFACE_CAPABILITIES_INIT;
    // wgpuSurfaceGetCapabilities(m_surface, m_adapter, &surface_capabilities);
    // WGPUTextureFormat surface_format = select_surface_format(surface_capabilities);
    // m_swapchain_format               = surface_format;
    // wgpuSurfaceCapabilitiesFreeMembers(surface_capabilities);

    // WGPUSurfaceConfiguration surface_config{
    //     .nextInChain     = nullptr,
    //     .device          = m_device,
    //     .format          = m_swapchain_format,
    //     .usage           = WGPUTextureUsage_RenderAttachment,
    //     .width           = window_state.width,
    //     .height          = window_state.height,
    //     .viewFormatCount = 0,
    //     .viewFormats     = nullptr,
    //     .alphaMode       = WGPUCompositeAlphaMode_Auto,
    //     .presentMode     = WGPUPresentMode_Fifo,
    // };
    // wgpuSurfaceConfigure(m_surface, &surface_config);

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
    // wgpuSurfaceUnconfigure(m_surface);
    // wgpuSurfaceRelease(m_surface);

    // wgpuQueueRelease(m_queue);
}

void HelloTriangle::Update(const WindowState& window) {
    // WGPUSurfaceTexture surface_texture = WGPU_SURFACE_TEXTURE_INIT;
    // wgpuSurfaceGetCurrentTexture(m_surface, &surface_texture);

    // if (surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal
    //     || surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Outdated) {
    //     wgpuSurfaceUnconfigure(m_surface);
    //     WGPUSurfaceConfiguration surface_config{
    //         .nextInChain     = nullptr,
    //         .device          = m_device,
    //         .format          = m_swapchain_format,
    //         .usage           = WGPUTextureUsage_RenderAttachment,
    //         .width           = window.width,
    //         .height          = window.height,
    //         .viewFormatCount = 0,
    //         .viewFormats     = nullptr,
    //         .alphaMode       = WGPUCompositeAlphaMode_Auto,
    //         .presentMode     = WGPUPresentMode_Fifo,
    //     };
    //     wgpuSurfaceConfigure(m_surface, &surface_config);
    //     return;
    // } else if (surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Error) {
    //     // Error, continue?
    //     fprintf(stderr, "Failed to get Surface texture\n");
    //     return;
    // }

    // WGPUTextureViewDescriptor view_descriptor = {
    //     .nextInChain     = nullptr,
    //     .label           = "Swapchain view"_wsv,
    //     .format          = m_swapchain_format,
    //     .dimension       = WGPUTextureViewDimension_2D,
    //     .baseMipLevel    = 0,
    //     .mipLevelCount   = 1,
    //     .baseArrayLayer  = 0,
    //     .arrayLayerCount = 1,
    //     .aspect          = WGPUTextureAspect_Undefined,
    //     .usage           = WGPUTextureUsage_RenderAttachment,
    // };
    // WGPUTextureView swapchain_view
    //     = wgpuTextureCreateView(surface_texture.texture, &view_descriptor);

    // WGPUCommandEncoderDescriptor encoder_descriptor{
    //     .nextInChain = nullptr,
    //     .label       = "CommandEncoder"_wsv,
    // };
    // WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &encoder_descriptor);
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
    m_device.submit(m_queue, {commandBuffer});



    // wgpuSurfacePresent(m_surface);

    // wgpuTextureViewRelease(swapchain_view);
    // wgpuTextureRelease(surface_texture.texture);
}
/***
    Hello Triangle - A basic WGPU introduction

    This example shows setup, teardown, and basic rendering of a triangle.
    It shows everything in this file explicitly as a reference - other examples will use helpers for
   initialization.

*/

#include "hello_triangle.h"

#include <webgpu/webgpu_loon.h>

#include <cassert>

#include "common/shaders.h"


static void WGPUErrorCallback(WGPUDevice const*,
                              WGPUErrorType  type,
                              WGPUStringView message,
                              WGPU_NULLABLE void*,
                              WGPU_NULLABLE void*) {
    switch (type) {
        case WGPUErrorType_Validation:
            fprintf(stderr,
                    "Validation Error: %.*s",
                    static_cast<int>(message.length),
                    message.data);
            break;
        case WGPUErrorType_Internal:
            fprintf(stderr,
                    "Internal webgpu Error: %.*s",
                    static_cast<int>(message.length),
                    message.data);
            break;
        case WGPUErrorType_OutOfMemory:
            fprintf(stderr,
                    "Webgpu out of memory error: %.*s",
                    static_cast<int>(message.length),
                    message.data);
            break;
        default: break;
    }
}

static void LoonLogCallback(WGPULoonLogLevel lvl, WGPUStringView message, void* userdata) {
    fprintf(stderr, "%.*s", static_cast<int>(message.length), message.data);
}

static WGPUInstance create_instance() {
    WGPULoonInstanceConfiguration loon_config {
        .chain = { .next = nullptr, .sType = static_cast<WGPUSType>(WGPUSType_LoonInstanceConfiguration),
           },
        .alloc = nullptr,
        .alloc_userdata = nullptr,
        .log_level = WGPULoonLogLevel_Debug,
        .log = LoonLogCallback,
        .log_userdata = nullptr,
    };

    WGPUInstanceDescriptor instance_descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    instance_descriptor.nextInChain            = &loon_config.chain;
    WGPUInstance instance                      = wgpuCreateInstance(&instance_descriptor);
    return instance;
}

static WGPUAdapter get_default_adapter(WGPUInstance instance) {
    WGPURequestAdapterOptions adapter_options{
        .nextInChain          = nullptr,
        .featureLevel         = WGPUFeatureLevel_Core,
        .powerPreference      = WGPUPowerPreference_HighPerformance,
        .forceFallbackAdapter = false,
        .backendType          = WGPUBackendType_Undefined,
        .compatibleSurface    = nullptr,
    };

    WGPUAdapter adapter = nullptr;
    auto        future  = wgpuInstanceRequestAdapter(
        instance,
        &adapter_options,
        {
                    .nextInChain = nullptr,
                    .mode        = WGPUCallbackMode_AllowSpontaneous,
                    .callback    = [](WGPURequestAdapterStatus,
                           WGPUAdapter adapter,
                           WGPUStringView,
                           void* userdata1,
                           void*) { *reinterpret_cast<WGPUAdapter*>(userdata1) = adapter; },
                    .userdata1   = (void*)&adapter,
                    .userdata2   = nullptr,
        });

    WGPUFutureWaitInfo wait_info{
        .future    = future,
        .completed = false,
    };
    const WGPUWaitStatus wait_status = wgpuInstanceWaitAny(instance, 1, &wait_info, UINT64_MAX);
    assert(wait_status == WGPUWaitStatus_Success);
    assert(wait_info.completed);

    return adapter;
}

static WGPUSurface create_surface(WGPUInstance instance, const WindowState& window_state) {
#if _WIN32

    WGPUSurfaceSourceWindowsHWND surface_source = WGPU_SURFACE_SOURCE_WINDOWS_HWND_INIT;
    surface_source.hwnd      = reinterpret_cast<void*>(window_state.native_window_handle);
    surface_source.hinstance = reinterpret_cast<void*>(window_state.native_instance_handle);
#elif __linux__
#    error "Unsupported platform currently"
#elif __APPLE__
    WGPUSurfaceSourceMetalLayer surface_source = WGPU_SURFACE_SOURCE_METAL_LAYER_INIT;
    surface_source.layer = reinterpret_cast<void*>(window_state.native_window_handle);
#endif


    WGPUSurfaceDescriptor surface_descriptor{
        .nextInChain = &surface_source.chain,
        .label       = "Hello Triangle Surface"_wsv,
    };

    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surface_descriptor);
    return surface;
}

static WGPUDevice get_default_device(WGPUInstance instance, WGPUAdapter adapter) {
    WGPUDevice device = nullptr;
    WGPUDeviceDescriptor device_descriptor{
        .nextInChain                 = nullptr,
        .label                       = WGPU_STRING_VIEW_INIT,
        .requiredFeatureCount        = 0,
        .requiredFeatures            = nullptr,
        .requiredLimits              = nullptr,
        .defaultQueue                = WGPU_QUEUE_DESCRIPTOR_INIT,
        .deviceLostCallbackInfo      = WGPU_DEVICE_LOST_CALLBACK_INFO_INIT,
        .uncapturedErrorCallbackInfo = {
            .nextInChain = nullptr,
            .callback = WGPUErrorCallback,
            .userdata1 = nullptr,
            .userdata2 = nullptr,
        },
    };

    const auto device_future = wgpuAdapterRequestDevice(
        adapter,
        &device_descriptor,
        {
            .nextInChain = nullptr,
            .mode        = WGPUCallbackMode_AllowSpontaneous,
            .callback    = [](WGPURequestDeviceStatus,
                           WGPUDevice device,
                           WGPUStringView,
                           void* userdata1,
                           void*) { *reinterpret_cast<WGPUDevice*>(userdata1) = device; },
            .userdata1   = (void*)&device,
            .userdata2   = nullptr,
        });

    WGPUFutureWaitInfo wait_info{
        .future    = device_future,
        .completed = false,
    };

    const WGPUWaitStatus wait_status = wgpuInstanceWaitAny(instance, 1, &wait_info, UINT64_MAX);
    assert(wait_status == WGPUWaitStatus_Success);
    assert(wait_info.completed);
    return device;
}

static WGPUTextureFormat select_surface_format(
    const WGPUSurfaceCapabilities& surface_capabilities) {
    for (size_t format_idx = 0; format_idx < surface_capabilities.formatCount; ++format_idx) {
        if (surface_capabilities.formats[format_idx] == WGPUTextureFormat_BGRA8UnormSrgb) {
            // Choose 8 bit srgb if we have it
            return surface_capabilities.formats[format_idx];
        }
    }
    return surface_capabilities.formats[0];
}

static WGPUShaderModule load_shader_module(WGPUDevice        device,
                                           ShaderModuleImpl* module,
                                           const char*       entry_point) {
    auto                  source = get_spirv(module, entry_point);
    WGPUShaderSourceSPIRV spirv  = WGPU_SHADER_SOURCE_SPIRV_INIT;
    spirv.code                   = source.data();
    spirv.codeSize               = source.size() * sizeof(uint32_t);

    WGPUShaderModuleDescriptor descriptor{
        .nextInChain = &spirv.chain,
        .label       = WGPU_STRING_VIEW_INIT,
    };
    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &descriptor);
    return shader;
}

static WGPURenderPipeline create_render_pipeline(WGPUDevice         device,
                                                 WGPUTextureFormat  surface_format,
                                                 WGPUPipelineLayout pipeline_layout,
                                                 WGPUShaderModule   vertex_shader,
                                                 WGPUShaderModule   fragment_shader) {
    WGPUBlendState blend_state = {
        .color = {.operation = WGPUBlendOperation_Add,
                  .srcFactor = WGPUBlendFactor_SrcAlpha,
                  .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,},
        .alpha = {.operation = WGPUBlendOperation_Add, 
            .srcFactor = WGPUBlendFactor_One, 
            .dstFactor = WGPUBlendFactor_Zero,},
    };

    WGPUColorTargetState color_target{
        .nextInChain = nullptr,
        .format      = surface_format,
        .blend       = &blend_state,
        .writeMask   = WGPUColorWriteMask_All,
    };

    WGPUFragmentState fragment_state{
        .nextInChain   = nullptr,
        .module        = fragment_shader,
        .entryPoint    = "fragmentMain"_wsv,
        .constantCount = 0,
        .constants     = nullptr,
        .targetCount   = 1,
        .targets       = &color_target,
    };

    WGPURenderPipelineDescriptor descriptor{
        .nextInChain  = nullptr,
        .label        = WGPUStringView{.data = "hello_triangle", .length = WGPU_STRLEN},
        .layout       = pipeline_layout,
        .vertex       = {
            .nextInChain = nullptr,
            .module = vertex_shader,
            .entryPoint  ="vertexMain"_wsv,
            .constantCount = 0,
            .constants = nullptr,
            .bufferCount = 0,
            .buffers = 0,
        },
        .primitive    = {
            .nextInChain = nullptr,
            .topology = WGPUPrimitiveTopology_TriangleList,
            .stripIndexFormat = WGPUIndexFormat_Uint16,
            .frontFace = WGPUFrontFace_CCW,
            .cullMode = WGPUCullMode_None,
            .unclippedDepth = false,
        },
        .depthStencil = nullptr,
        .multisample  = WGPU_MULTISAMPLE_STATE_INIT,
        .fragment     = &fragment_state,
    };

    return wgpuDeviceCreateRenderPipeline(device, &descriptor);
}

HelloTriangle::HelloTriangle(const WindowState& window_state) {
    m_instance = create_instance();
    m_adapter  = get_default_adapter(m_instance);
    m_surface  = create_surface(m_instance, window_state);
    m_device   = get_default_device(m_instance, m_adapter);

    WGPUSurfaceCapabilities surface_capabilities = WGPU_SURFACE_CAPABILITIES_INIT;
    wgpuSurfaceGetCapabilities(m_surface, m_adapter, &surface_capabilities);
    WGPUTextureFormat surface_format = select_surface_format(surface_capabilities);
    m_swapchain_format               = surface_format;
    wgpuSurfaceCapabilitiesFreeMembers(surface_capabilities);

    WGPUSurfaceConfiguration surface_config{
        .nextInChain     = nullptr,
        .device          = m_device,
        .format          = m_swapchain_format,
        .usage           = WGPUTextureUsage_RenderAttachment,
        .width           = window_state.width,
        .height          = window_state.height,
        .viewFormatCount = 0,
        .viewFormats     = nullptr,
        .alphaMode       = WGPUCompositeAlphaMode_Auto,
        .presentMode     = WGPUPresentMode_Fifo,
    };
    wgpuSurfaceConfigure(m_surface, &surface_config);

    // Load shaders and create render pipeline

    ShaderModule     shader = window_state.shader_loader->load_module("hello_triangle.slang");
    WGPUShaderModule vertex_shader   = load_shader_module(m_device, shader.get(), "vertexMain");
    WGPUShaderModule fragment_shader = load_shader_module(m_device, shader.get(), "fragmentMain");

    WGPUPipelineLayoutDescriptor pipeline_layout_descriptor{
        .nextInChain          = nullptr,
        .label                = "Empty pipeline descriptor"_wsv,
        .bindGroupLayoutCount = 0,
        .bindGroupLayouts     = nullptr,
    };
    WGPUPipelineLayout pipeline_layout
        = wgpuDeviceCreatePipelineLayout(m_device, &pipeline_layout_descriptor);
    assert(pipeline_layout != nullptr);

    m_render_pipeline = create_render_pipeline(m_device,
                                               surface_format,
                                               pipeline_layout,
                                               vertex_shader,
                                               fragment_shader);
    assert(m_render_pipeline != nullptr);

    m_queue = wgpuDeviceGetQueue(m_device);

    wgpuShaderModuleRelease(vertex_shader);
    wgpuShaderModuleRelease(fragment_shader);
    wgpuPipelineLayoutRelease(pipeline_layout);
}

HelloTriangle::~HelloTriangle() {
    wgpuRenderPipelineRelease(m_render_pipeline);
    wgpuSurfaceUnconfigure(m_surface);
    wgpuSurfaceRelease(m_surface);
    wgpuQueueRelease(m_queue);
    wgpuDeviceRelease(m_device);
    wgpuAdapterRelease(m_adapter);
    wgpuInstanceRelease(m_instance);
}

void HelloTriangle::Update(const WindowState& window) {
    WGPUSurfaceTexture surface_texture = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(m_surface, &surface_texture);

    if (surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal
        || surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Outdated) {
        wgpuSurfaceUnconfigure(m_surface);
        WGPUSurfaceConfiguration surface_config{
            .nextInChain     = nullptr,
            .device          = m_device,
            .format          = m_swapchain_format,
            .usage           = WGPUTextureUsage_RenderAttachment,
            .width           = window.width,
            .height          = window.height,
            .viewFormatCount = 0,
            .viewFormats     = nullptr,
            .alphaMode       = WGPUCompositeAlphaMode_Auto,
            .presentMode     = WGPUPresentMode_Fifo,
        };
        wgpuSurfaceConfigure(m_surface, &surface_config);
        return;
    } else if (surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Error) {
        // Error, continue?
        fprintf(stderr, "Failed to get Surface texture\n");
        return;
    }

    WGPUTextureViewDescriptor view_descriptor = {
        .nextInChain     = nullptr,
        .label           = "Swapchain view"_wsv,
        .format          = m_swapchain_format,
        .dimension       = WGPUTextureViewDimension_2D,
        .baseMipLevel    = 0,
        .mipLevelCount   = 1,
        .baseArrayLayer  = 0,
        .arrayLayerCount = 1,
        .aspect          = WGPUTextureAspect_Undefined,
        .usage           = WGPUTextureUsage_RenderAttachment,
    };
    WGPUTextureView swapchain_view
        = wgpuTextureCreateView(surface_texture.texture, &view_descriptor);

    WGPUCommandEncoderDescriptor encoder_descriptor{
        .nextInChain = nullptr,
        .label       = "CommandEncoder"_wsv,
    };
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &encoder_descriptor);

    WGPURenderPassColorAttachment color_attachment{
        .nextInChain   = nullptr,
        .view          = swapchain_view,
        .depthSlice    = WGPU_DEPTH_SLICE_UNDEFINED,
        .resolveTarget = nullptr,
        .loadOp        = WGPULoadOp_Clear,
        .storeOp       = WGPUStoreOp_Store,
        .clearValue    = WGPUColor{.r = 0.f, .g = 0.f, .b = 0.f, .a = 1.f},
    };
    WGPURenderPassDescriptor render_pass_descriptor{
        .nextInChain            = nullptr,
        .label                  = "Render pass"_wsv,
        .colorAttachmentCount   = 1,
        .colorAttachments       = &color_attachment,
        .depthStencilAttachment = nullptr,
        .occlusionQuerySet      = nullptr,
        .timestampWrites        = nullptr,
    };

    WGPURenderPassEncoder render_pass
        = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(render_pass, m_render_pipeline);
    wgpuRenderPassEncoderSetViewport(render_pass, 0, 0, window.width, window.height, 0.f, 1.f);
    wgpuRenderPassEncoderDraw(render_pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(render_pass);
    wgpuRenderPassEncoderRelease(render_pass);

    WGPUCommandBufferDescriptor buffer_descriptor{
        .nextInChain = nullptr,
        .label       = "CommandBuffer"_wsv,
    };
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, &buffer_descriptor);

    wgpuQueueSubmit(m_queue, 1, &command_buffer);

    wgpuCommandBufferRelease(command_buffer);
    wgpuCommandEncoderRelease(encoder);

    wgpuSurfacePresent(m_surface);

    wgpuTextureViewRelease(swapchain_view);
    wgpuTextureRelease(surface_texture.texture);
}
/***
    Hello Triangle - A basic WGPU introduction

    This example shows setup, teardown, and basic rendering of a triangle.
    It shows everything in this file explicitly as a reference - other examples will use helpers for
   initialization.

*/

#include "hello_cube.h"

#include <webgpu/webgpu_loon.h>

#include <cassert>

#include "common/geometry.h"
#include "common/initialization.h"
#include "common/shaders.h"
#include "webgpu/webgpu.h"

struct Vertex {
    float3 position;
    float3 color;
};

constexpr Vertex kCubeVerts[] = {
    {.position = {1, 1, 1}, .color = {1, 0, 0}},
    {.position = {1, 1, -1}, .color = {0, 1, 0}},
    {.position = {1, -1, 1}, .color = {0, 0, 1}},
    {.position = {1, -1, -1}, .color = {1, 0, 1}},
    {.position = {-1, 1, 1}, .color = {0, 1, 1}},
    {.position = {-1, 1, -1}, .color = {1, 1, 0}},
    {.position = {-1, -1, 1}, .color = {0, 1, 0}},
    {.position = {-1, -1, -1}, .color = {0, 0, 1}},
};

constexpr uint16_t kCubeIndices[] = {0, 2, 1, 1, 2, 3, 0, 6, 2, 0, 4, 6, 4, 5, 6, 6, 5, 7,
                                     5, 1, 7, 7, 1, 3, 1, 0, 4, 1, 4, 5, 2, 6, 3, 6, 7, 3};

struct CameraInfo {
    float3   position                = {0, 0, 5.f};
    quatf    rotationCameraFromWorld = {};
    float4x4 projection              = {};
};

constexpr size_t kNumFramesInFlight = 3;

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

    WGPUVertexAttribute vertex_attrs[2] = {
        {
            .nextInChain    = nullptr,
            .format         = WGPUVertexFormat_Float32x3,
            .offset         = offsetof(Vertex, position),
            .shaderLocation = 0,
        },
        {
            .nextInChain    = nullptr,
            .format         = WGPUVertexFormat_Float32x3,
            .offset         = offsetof(Vertex, color),
            .shaderLocation = 1,
        },
    };

    WGPUVertexBufferLayout vertex_layout{
        .nextInChain    = nullptr,
        .stepMode       = WGPUVertexStepMode_Vertex,
        .arrayStride    = sizeof(Vertex),
        .attributeCount = 2,
        .attributes     = vertex_attrs,
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
            .bufferCount = 1,
            .buffers = &vertex_layout,
        },
        .primitive    = {
            .nextInChain = nullptr,
            .topology = WGPUPrimitiveTopology_TriangleList,
            .stripIndexFormat = WGPUIndexFormat_Uint16,
            .frontFace = WGPUFrontFace_CCW,
            .cullMode = WGPUCullMode_Back,
            .unclippedDepth = false,
        },
        .depthStencil = nullptr,
        .multisample  = WGPU_MULTISAMPLE_STATE_INIT,
        .fragment     = &fragment_state,
    };

    return wgpuDeviceCreateRenderPipeline(device, &descriptor);
}

HelloCube::HelloCube(const WindowState& window_state) {
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

    ShaderModule     shader          = window_state.shader_loader->load_module("hello_cube.slang");
    WGPUShaderModule vertex_shader   = load_shader_module(m_device, shader.get(), "vertexMain");
    WGPUShaderModule fragment_shader = load_shader_module(m_device, shader.get(), "fragmentMain");

    // SKETCH: How do I want this to look?
    WGPUBindGroupLayout camera_data_layout
        = create_bind_group_layout_builder(shader.get(), "cameraData")
              .dynamic_offsets(true)
              .build(m_device);

    WGPUPipelineLayoutDescriptor pipeline_layout_descriptor{
        .nextInChain          = nullptr,
        .label                = "Empty pipeline descriptor"_wsv,
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &camera_data_layout,
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

    WGPUBufferDescriptor buffer_descriptor{
        .nextInChain      = nullptr,
        .label            = "Vertex Buffer"_wsv,
        .usage            = WGPUBufferUsage_Vertex,
        .size             = sizeof(kCubeVerts),
        .mappedAtCreation = true,
    };

    m_vertex_buffer = wgpuDeviceCreateBuffer(m_device, &buffer_descriptor);
    wgpuBufferWriteMappedRange(m_vertex_buffer, 0, kCubeVerts, sizeof(kCubeVerts));
    wgpuBufferUnmap(m_vertex_buffer);

    WGPUBufferDescriptor index_buffer_descriptor{
        .nextInChain      = nullptr,
        .label            = "Index buffer"_wsv,
        .usage            = WGPUBufferUsage_Index,
        .size             = sizeof(kCubeIndices),
        .mappedAtCreation = true,
    };
    m_index_buffer = wgpuDeviceCreateBuffer(m_device, &index_buffer_descriptor);
    wgpuBufferWriteMappedRange(m_index_buffer, 0, kCubeIndices, sizeof(kCubeIndices));
    wgpuBufferUnmap(m_index_buffer);

    WGPUBufferDescriptor camera_data_buffer_descriptor{
        .nextInChain      = nullptr,
        .label            = "Camera Data"_wsv,
        .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size             = sizeof(CameraInfo),
        .mappedAtCreation = true,
    };

    m_camera_data_buffer = wgpuDeviceCreateBuffer(m_device, &camera_data_buffer_descriptor);
    // wgpuBufferWriteMappedRange(m_camera_data_buffer, 0, const void* data, size_t size);
    wgpuBufferUnmap(m_camera_data_buffer);

    WGPUBindGroupEntry camera_data_entry{
        .nextInChain = nullptr,
        .binding     = 0,
        .buffer      = m_camera_data_buffer,
        .offset      = 0,
        .size        = WGPU_WHOLE_SIZE,
        .sampler     = nullptr,
        .textureView = nullptr,
    };

    WGPUBindGroupDescriptor camera_data_descriptor{
        .nextInChain = nullptr,
        .label       = ""_wsv,
        .layout      = camera_data_layout,
        .entryCount  = 1,
        .entries     = &camera_data_entry,
    };

    m_bind_group = wgpuDeviceCreateBindGroup(m_device, &camera_data_descriptor);

    wgpuShaderModuleRelease(vertex_shader);
    wgpuShaderModuleRelease(fragment_shader);
    wgpuPipelineLayoutRelease(pipeline_layout);
}

HelloCube::~HelloCube() {
    wgpuRenderPipelineRelease(m_render_pipeline);
    wgpuSurfaceUnconfigure(m_surface);
    wgpuSurfaceRelease(m_surface);
    wgpuQueueRelease(m_queue);
    wgpuDeviceRelease(m_device);
    wgpuAdapterRelease(m_adapter);
    wgpuInstanceRelease(m_instance);
}

void HelloCube::Update(const WindowState& window) {
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



    // wgpuQueueWriteBuffer()
    wgpuRenderPassEncoderSetPipeline(render_pass, m_render_pipeline);
    wgpuRenderPassEncoderSetViewport(render_pass, 0, 0, window.width, window.height, 0.f, 1.f);
    wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, m_vertex_buffer, 0, sizeof(kCubeVerts));
    wgpuRenderPassEncoderSetIndexBuffer(render_pass,
                                        m_index_buffer,
                                        WGPUIndexFormat_Uint16,
                                        0,
                                        sizeof(kCubeIndices));
    wgpuRenderPassEncoderDrawIndexed(render_pass,
                                     sizeof(kCubeIndices) / sizeof(kCubeIndices[0]),
                                     1,
                                     0,
                                     0,
                                     0);
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
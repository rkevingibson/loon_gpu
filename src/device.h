#pragma once

#include <cstdint>

#include "platform_utils.h"
#include "queue.h"
#include "utilities.h"
#include "volk.h"
#include "webgpu/webgpu.h"


VK_DEFINE_HANDLE(VmaAllocator);

namespace webgpu {
struct CommandsMixin;
}

namespace webgpu {
}
#ifndef WGPU_VK_CHECK
#    define WGPU_VK_CHECK(fn_call)                                                                 \
        do {                                                                                       \
            VkResult _vk_result = vk_api.fn_call;                                                  \
            if (_vk_result != VK_SUCCESS) {                                                        \
                device->vulkan_error(_vk_result, __FILE__, __LINE__);                              \
            }                                                                                      \
        } while (false)
#endif

#ifndef WGPU_VMA_CHECK
#    define WGPU_VMA_CHECK(fn_call)                                                                \
        do {                                                                                       \
            VkResult _vk_result = fn_call;                                                         \
            if (_vk_result != VK_SUCCESS) {                                                        \
                device->vulkan_error(_vk_result, __FILE__, __LINE__);                              \
            }                                                                                      \
        } while (false)
#endif

struct WGPUDeviceImpl {
   public:
    WGPUDeviceImpl();
    ~WGPUDeviceImpl();

    WGPURequestDeviceStatus initialize(WGPUAdapter adapter, const WGPUDeviceDescriptor* descriptor);

    void add_ref();
    void release();
    void destroy();

    // TODO: Handle device lost errors
    bool is_lost() const { return false; }

    void       vulkan_error(VkResult res, const char* file, int line);
    void       error(WGPUErrorType type, WGPUStringView msg);
    void       push_error_scope(WGPUErrorFilter filter);
    WGPUFuture pop_error_scope(WGPUPopErrorScopeCallbackInfo callback_info);

    // Resource management:
    WGPUBuffer          create_buffer(WGPUBufferDescriptor const* descriptor);
    WGPUBindGroupLayout create_bind_group_layout(WGPUBindGroupLayoutDescriptor const* descriptor);
    WGPUBindGroup       create_bind_group(WGPUBindGroupDescriptor const* descriptor);
    WGPUPipelineLayout  create_pipeline_layout(WGPUPipelineLayoutDescriptor const* descriptor);

    // Shaders + Pipelines:
    WGPUShaderModule    create_shader_module(WGPUShaderModuleDescriptor const* descriptor);
    WGPUComputePipeline create_compute_pipeline(WGPUComputePipelineDescriptor const* descriptor);
    WGPURenderPipeline  create_render_pipeline(WGPURenderPipelineDescriptor const* descriptor);

    // Commands:
    WGPUCommandEncoder create_command_encoder(WGPUCommandEncoderDescriptor const* descriptor);

    void free(WGPUShaderModule);
    void free(WGPURenderPipeline);
    void free(WGPUComputePipeline);
    void free(WGPUBindGroupLayout);
    void free(WGPUPipelineLayout);
    void free(WGPUTexture);
    void free(WGPUTextureView);
    void free(WGPUBuffer);
    void free(WGPUCommandEncoder);
    void free(WGPURenderPassEncoder);
    void free(WGPUComputePassEncoder);
    void free(WGPUCommandBuffer);
    void free(WGPUBindGroup);
    void free(WGPUQuerySet);
    void free(WGPUQueue queue) { /*NOOP - queue is by-value part of the device.*/ }

    WGPUCommandBuffer allocate_command_buffer();

    loon::gpu::Allocator& get_allocator() { return m_allocator; }
    loon::gpu::Arena*     get_thread_local_arena();

    WGPUInstance get_instance() const { return m_instance; }
    WGPUAdapter  get_adapter() const { return m_adapter; }
    uint32_t     get_queue_family() const;

    VkDevice        vk_device = VK_NULL_HANDLE;
    VolkDeviceTable vk_api{};
    VmaAllocator    vk_allocator = VK_NULL_HANDLE;

    WGPUQueueImpl queue;

    loon::gpu::ObjectList<WGPUCommandBufferImpl> cmd_buffers;
    loon::gpu::Label                             label;

   private:
    friend class WGPUSurfaceImpl;
    friend class WGPUCommandEncoderImpl;
    friend class WGPUTextureImpl;  // For allocating texture views.

    struct ThreadLocalState;
    ThreadLocalState*    get_thread_local_state();
    const char*          get_temp_null_terminated_string(WGPUStringView msg);
    void                 free_temp_string(const char* str);
    loon::gpu::Allocator m_allocator;
    WGPUInstance         m_instance = nullptr;
    WGPUAdapter          m_adapter  = nullptr;

    loon::gpu::ReferenceCount                         m_refcount;
    loon::gpu::ObjectList<WGPUShaderModuleImpl>       m_shader_modules;
    loon::gpu::ObjectList<WGPURenderPipelineImpl>     m_render_pipelines;
    loon::gpu::ObjectList<WGPUComputePipelineImpl>    m_compute_pipelines;
    loon::gpu::ObjectList<WGPUBindGroupLayoutImpl>    m_bind_group_layouts;
    loon::gpu::ObjectList<WGPUBindGroupImpl>          m_bind_groups;
    loon::gpu::ObjectList<WGPUPipelineLayoutImpl>     m_pipeline_layouts;
    loon::gpu::ObjectList<WGPUTextureImpl>            m_textures;
    loon::gpu::ObjectList<WGPUTextureViewImpl>        m_texture_views;
    loon::gpu::ObjectList<WGPUBufferImpl>             m_buffers;
    loon::gpu::ObjectList<WGPUCommandEncoderImpl>     m_cmd_encoders;
    loon::gpu::ObjectList<WGPURenderPassEncoderImpl>  m_render_passes;
    loon::gpu::ObjectList<WGPUComputePassEncoderImpl> m_compute_passes;

    loon::gpu::DescriptorSetAllocator m_descriptor_set_allocator;

    WGPUUncapturedErrorCallbackInfo m_error_callback;
    loon::gpu::tls_key              m_tls_key;
};
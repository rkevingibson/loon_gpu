#include "gpu/loon_gpu.h"

#include "containers.h"
#include "gpu_to_vk.h"
#include "vma_usage.h"
#include "volk.h"
#include "vulkan/vulkan_core.h"


namespace loon::gpu {

static constexpr const char* kRequiredDeviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
};
static constexpr size_t kRequiredDeviceExtensionsCount
    = sizeof(kRequiredDeviceExtensions) / sizeof(kRequiredDeviceExtensions[0]);

struct Buffer {
    VkBuffer      vk_buffer;
    VmaAllocation vk_allocation;
};

struct PhysicalDeviceInfo {
    VkPhysicalDevice device       = VK_NULL_HANDLE;
    uint32_t         queue_family = 0;
};

struct Device::Impl {
    bool initialize(const DeviceDesc& desc);

    void shutdown();

    // Buffers:
    Handle<Buffer> malloc(size_t bytes, MEMORY memory = MEMORY_DEFAULT);
    Handle<Buffer> malloc(size_t bytes, size_t align, MEMORY memory = MEMORY_DEFAULT);
    void           free(Handle<Buffer> buffer);
    GpuPtr         getDevicePointer(Handle<Buffer> buffer);

    // Textures:
    Handle<Texture>     createTexture(const GpuTextureDesc& desc);
    Handle<TextureHeap> createTextureHeap(size_t size);
    uint32_t            createTextureView(Handle<TextureHeap> heap,
                                          Handle<Texture>     texture,
                                          TextureViewDesc     desc);
    uint32_t            createRWTextureView(Handle<TextureHeap> heap,
                                            Handle<Texture>     texture,
                                            TextureViewDesc     desc);
    void                free(Handle<Texture>);
    void                free(Handle<TextureHeap>);
    void                freeTextureView(Handle<TextureHeap> heap, uint32_t view);

    // Pipelines
    Handle<Pipeline> createComputePipeline(ShaderSource computeIR);
    Handle<Pipeline> createGraphicsPipeline(ShaderSource vertex,
                                            ShaderSource fragment,
                                            RasterDesc   desc);
    Handle<Pipeline> createGraphicsMeshletPipeline(ShaderSource meshletIR,
                                                   ShaderSource pixelIR,
                                                   RasterDesc   desc);
    void             freePipeline(Handle<Pipeline> pipeline);

    // State objects
    Handle<DepthStencilState> createDepthStencilState(GpuDepthStencilDesc desc);
    Handle<BlendState>        createBlendState(GpuBlendDesc desc);
    void                      freeDepthStencilState(Handle<DepthStencilState> state);
    void                      freeBlendState(Handle<BlendState> state);

    // Queue
    Handle<Queue>         createQueue(/* DEVICE & QUEUE CREATION DETAILS OMITTED */);
    Handle<CommandBuffer> startCommandRecording(Handle<Queue> queue);
    void                  submit(Handle<Queue> queue, Span<Handle<CommandBuffer>> commandBuffers);
    void                  cancel(Handle<Queue> queue, Span<Handle<CommandBuffer>> commandBuffers);

    // Semaphores
    Handle<Semaphore> createSemaphore(uint64_t initValue);
    void              waitSemaphore(Handle<Semaphore> sema, uint64_t value);
    void              destroySemaphore(Handle<Semaphore> sema);


   private:
    Allocator        m_allocator;
    VkInstance       m_instance;
    VkPhysicalDevice m_physical_device;
    VkDevice         m_device;
    VolkDeviceTable  m_api;
    VmaAllocator     vk_allocator = VK_NULL_HANDLE;

    VkPipelineLayout default_graphics_layout;

    ObjectPool<Buffer, kMaxNumBuffers> buffer_pool;

    void               log(LogLevel lvl, Span<const char> msg);
    void               chk(VkResult result);
    PhysicalDeviceInfo selectPhysicalDevice(GpuPreference preference);
    VkPipelineLayout   createDefaultGraphicsLayout();
};

// MARK: Initialization


PhysicalDeviceInfo Device::Impl::selectPhysicalDevice(GpuPreference preference) {
    constexpr uint32_t max_physical_devices = 8;
    VkPhysicalDevice   physical_devices[max_physical_devices];
    uint32_t           device_count = max_physical_devices;
    VkResult vkresult = vkEnumeratePhysicalDevices(m_instance, &device_count, physical_devices);

    if (vkresult == VK_INCOMPLETE) {
        log(LogLevel_Warning, "Too many vulkan physical devices returned some will be ignored");
    }
    if (vkresult < 0) { return PhysicalDeviceInfo{}; }

    const bool prefer_integrated = preference == GpuPreference_Integrated;
    const bool prefer_dedicated  = preference == GpuPreference_Discrete;

    VkPhysicalDevice best_device       = VK_NULL_HANDLE;
    uint32_t         best_queue_family = 0;

    device_count = device_count < max_physical_devices ? device_count : max_physical_devices;
    for (uint32_t device_idx = 0; device_idx < device_count; ++device_idx) {
        const VkPhysicalDevice     physical_device = physical_devices[device_idx];
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_device, &properties);

        constexpr uint32_t      max_queue_families                   = 16;
        VkQueueFamilyProperties queue_properties[max_queue_families] = {};
        uint32_t                queue_family_count                   = max_queue_families;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                                 &queue_family_count,
                                                 queue_properties);
        if (queue_family_count > max_queue_families) {
            log(LogLevel_Warning, "Too many queue families on physical device, some may be missed");
        }
        queue_family_count
            = queue_family_count < max_queue_families ? queue_family_count : max_queue_families;

        // For WebGPU, we only need one queue (might add extension to support async compute later).
        // We need compute and graphics support, and possibly presentation support
        uint32_t selected_queue_family = ~0;
        for (uint32_t queue_family = 0; queue_family < queue_family_count; ++queue_family) {
            const auto& props    = queue_properties[queue_family];
            bool        is_valid = (props.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                            && (props.queueFlags & VK_QUEUE_COMPUTE_BIT);
            // Need to find the graphics/presentation queue.
            // if (options->compatibleSurface) {
            //     const VkSurfaceKHR surface           = options->compatibleSurface->vk_surface;
            //     VkBool32           surface_supported = false;
            //     vkGetPhysicalDeviceSurfaceSupportKHR(physical_devices[device_idx],
            //                                          queue_family,
            //                                          surface,
            //                                          &surface_supported);

            //     is_valid = is_valid && surface_supported;
            // }

            if (is_valid) {
                selected_queue_family = queue_family;
                break;
            }
        }
        if (selected_queue_family == ~0u) { continue; }

        // Check device extensions - there's no reasonable upperlimit on the number of device
        // extensions, so need to allocate :(
        uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
        // TODO: Use the arena instead.
        auto extension_properties_block
            = m_allocator.alloc(sizeof(VkExtensionProperties) * extension_count);
        auto* extension_properties
            = reinterpret_cast<VkExtensionProperties*>(extension_properties_block.ptr);
        vkEnumerateDeviceExtensionProperties(physical_device,
                                             nullptr,
                                             &extension_count,
                                             extension_properties);
        // Need to check against the list of required extensions and make sure all required
        // extensions are available.
        for (uint32_t i = 0; i < extension_count; ++i) {
            printf("%s\n", extension_properties[i].extensionName);
        }
        bool all_extensions_supported = true;
        for (size_t required_ext_idx = 0;
             required_ext_idx < loon::gpu::kRequiredDeviceExtensionsCount;
             ++required_ext_idx) {
            const char* required_extension = loon::gpu::kRequiredDeviceExtensions[required_ext_idx];
            bool        extension_found    = false;
            for (size_t available_ext_idx = 0; available_ext_idx < extension_count;
                 ++available_ext_idx) {
                if (strcmp(extension_properties[available_ext_idx].extensionName,
                           required_extension)
                    == 0) {
                    extension_found = true;
                    break;
                }
            }

            if (!extension_found) {
                all_extensions_supported = false;
                break;
            }
        }
        m_allocator.free(extension_properties_block);

        if (!all_extensions_supported) {
            // Invalid device, doesn't support the extensions we need.
            continue;
        }

        // If we don't have a "best" device yet, let's use this one.
        if (best_device == VK_NULL_HANDLE) {
            best_device       = physical_device;
            best_queue_family = selected_queue_family;
        }

        const bool is_integrated = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        if (is_integrated && prefer_integrated) {
            best_device       = physical_device;
            best_queue_family = selected_queue_family;
        }

        const bool is_dedicated = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        if (is_dedicated && prefer_dedicated) {
            best_device       = physical_device;
            best_queue_family = selected_queue_family;
        }
    }

    return PhysicalDeviceInfo{
        .device       = best_device,
        .queue_family = best_queue_family,
    };
}

bool Device::Impl::initialize(const DeviceDesc& desc) {
    // Setup instance
    VkResult result = volkInitialize();
    if (result != VK_SUCCESS) return false;

    VkApplicationInfo app_info = {
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext              = nullptr,
        .pApplicationName   = nullptr,
        .applicationVersion = 0,
        .pEngineName        = "loon",
        .engineVersion      = 0,
        .apiVersion         = VK_API_VERSION_1_3,
    };

    const char* instance_extensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_WIN32_KHR
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#elif defined(VK_USE_PLATFORM_METAL_EXT)
        VK_EXT_METAL_SURFACE_EXTENSION_NAME,
#endif
    };

    const auto instance_info = VkInstanceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = 0,
        .pApplicationInfo        = &app_info,
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = sizeof(instance_extensions) / sizeof(instance_extensions[0]),
        .ppEnabledExtensionNames = instance_extensions,
    };

    result = vkCreateInstance(&instance_info, nullptr, &m_instance);
    if (result != VK_SUCCESS) return false;
    volkLoadInstanceOnly(m_instance);


    // Select physical device
    auto physical_device_info = selectPhysicalDevice(desc.gpu_preference);

    return true;
}

void Device::Impl::shutdown() {
    vkDestroyInstance(m_instance, nullptr);
    volkFinalize();
}

// MARK: Buffers

Handle<Buffer> Device::Impl::malloc(size_t bytes, MEMORY memory) {
    return malloc(bytes, 64, memory);
}

Handle<Buffer> Device::Impl::malloc(size_t bytes, size_t align, MEMORY memory) {
    constexpr VkBufferUsageFlags kDefaultUsages
        = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
          | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
          | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VkBufferCreateInfo create_info{
        .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .size                  = bytes,
        .usage                 = kDefaultUsages,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,  // TODO: Support multiple queues.
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
    };

    VmaAllocationCreateFlags flags = 0;
    switch (memory) {
        case MEMORY_DEFAULT:
            flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case MEMORY_GPU: flags = 0; break;
        case MEMORY_READBACK:
            flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            break;
    }

    VmaAllocationCreateInfo alloc_info{
        .flags          = flags,
        .usage          = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags  = 0,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool           = VK_NULL_HANDLE,
        .pUserData      = 0,
        .priority       = 0.,
    };

    VkBuffer      vk_buffer     = nullptr;
    VmaAllocation vk_allocation = nullptr;
    chk(vmaCreateBuffer(vk_allocator,
                        &create_info,
                        &alloc_info,
                        &vk_buffer,
                        &vk_allocation,
                        nullptr));

    const uint32_t buffer_idx = buffer_pool.get();
    buffer_pool[buffer_idx]   = {
          .vk_buffer     = vk_buffer,
          .vk_allocation = vk_allocation,
    };
    return {.h = buffer_idx};
}

void Device::Impl::free(Handle<Buffer> buffer) {
    auto& b = buffer_pool[buffer.h];
    vmaDestroyBuffer(vk_allocator, b.vk_buffer, b.vk_allocation);
}

GpuPtr Device::Impl::getDevicePointer(Handle<Buffer> buffer) {
    VkBufferDeviceAddressInfo addr_info{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = buffer_pool[buffer.h].vk_buffer,
    };
    return m_api.vkGetBufferDeviceAddress(m_device, &addr_info);
}


// MARK: Pipelines

VkPipelineLayout Device::Impl::createDefaultGraphicsLayout() {
    // TODO: Set layout for bindless textures
    // VkDescriptorSetLayout set_layout;

    // We create 2 push constants - for vertex and fragment data.
    VkPushConstantRange push_constant_ranges[2] = {
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset     = 0,
            .size       = sizeof(VkDeviceAddress),
        },
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset     = sizeof(VkDeviceAddress),
            .size       = sizeof(VkDeviceAddress),
        },
    };


    VkPipelineLayoutCreateInfo create_info{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0,
        .setLayoutCount         = 0,
        .pSetLayouts            = nullptr,
        .pushConstantRangeCount = 2,
        .pPushConstantRanges    = push_constant_ranges,
    };

    VkPipelineLayout pipeline_layout;
    chk(m_api.vkCreatePipelineLayout(m_device, &create_info, nullptr, &pipeline_layout));
    return pipeline_layout;
}

Handle<Pipeline> Device::Impl::createGraphicsPipeline(ShaderSource vertex,
                                                      ShaderSource fragment,
                                                      RasterDesc   desc) {
    VkShaderModuleCreateInfo vert_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = vertex.spirv.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(vertex.spirv.data()),
    };
    VkShaderModule vert_module;
    chk(m_api.vkCreateShaderModule(m_device, &vert_info, nullptr, &vert_module));

    VkShaderModuleCreateInfo frag_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = fragment.spirv.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(fragment.spirv.data()),
    };
    VkShaderModule frag_module;
    chk(m_api.vkCreateShaderModule(m_device, &frag_info, nullptr, &frag_module));

    VkPipelineShaderStageCreateInfo stages[] = {
        // Vertex stage info
        VkPipelineShaderStageCreateInfo{
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext               = nullptr,
            .flags               = 0,
            .stage               = VK_SHADER_STAGE_VERTEX_BIT,
            .module              = vert_module,
            .pName               = vertex.entry_point.data(),
            .pSpecializationInfo = nullptr,  // TODO: Support specialization constants.
        },
        // Fragment stage info
        VkPipelineShaderStageCreateInfo{
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext               = nullptr,
            .flags               = 0,
            .stage               = VK_SHADER_STAGE_VERTEX_BIT,
            .module              = frag_module,
            .pName               = fragment.entry_point.data(),
            .pSpecializationInfo = nullptr,  // TODO: Support specialization constants.
        },
    };

    VkPipelineVertexInputStateCreateInfo vertex_input_state{
        .sType                         = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext                         = nullptr,
        .flags                         = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions    = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions    = nullptr,
    };

    // Primitive state:
    VkPipelineInputAssemblyStateCreateInfo input_assembly_state{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0,
        .topology               = loon::gpu::bridge(desc.topology),
        .primitiveRestartEnable = false,
    };


    // Depth-stencil state:
    VkFormat depth_attachment_format   = VK_FORMAT_UNDEFINED;
    VkFormat stencil_attachment_format = VK_FORMAT_UNDEFINED;
    // TODO: Depth stencil formats

    // Color blend state
    // TODO: Color blend state:
    loon::gpu::Stack<VkPipelineColorBlendAttachmentState, loon::gpu::kMaxColorAttachments>
                                                                color_blend_attachment_states{};
    loon::gpu::Stack<VkFormat, loon::gpu::kMaxColorAttachments> color_attachment_formats{};

    for (auto& t : desc.colorTargets) {
        // const auto attachment_state = loon::gpu::bridge(target);
        // color_blend_attachment_states.push(attachment_state);
        color_attachment_formats.push(loon::gpu::bridge(t.format));
    }

    VkPipelineColorBlendStateCreateInfo color_blend_state{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0,
        .logicOpEnable   = false,
        .logicOp         = VK_LOGIC_OP_NO_OP,
        .attachmentCount = color_blend_attachment_states.size(),
        .pAttachments    = color_blend_attachment_states.data(),
        .blendConstants  = {1.f, 1.f, 1.f, 1.f},
    };

    // Rasterization state:
    VkCullModeFlags cull_mode  = VK_CULL_MODE_BACK_BIT;
    VkFrontFace     front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    switch (desc.cull) {
        case CULL_CCW: front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE; break;
        case CULL_CW: front_face = VK_FRONT_FACE_CLOCKWISE; break;
        case CULL_ALL: cull_mode = VK_CULL_MODE_FRONT_AND_BACK; break;
        case CULL_NONE: cull_mode = VK_CULL_MODE_NONE; break;
    }

    VkPipelineRasterizationStateCreateInfo rasterization_state{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = 0,
        .depthClampEnable        = false,
        .rasterizerDiscardEnable = false,
        .polygonMode             = VK_POLYGON_MODE_FILL,
        .cullMode                = cull_mode,
        .frontFace               = front_face,
        .depthBiasEnable         = false,
        .depthBiasConstantFactor = 0,
        .depthBiasClamp          = 0,
        .depthBiasSlopeFactor    = 0,
        .lineWidth               = 1.f,
    };

    // Multisample state:

    // TODO: Proper multisampling support
    VkPipelineMultisampleStateCreateInfo multisample_state{
        .sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable   = false,
        .minSampleShading      = 1.0f,
        .pSampleMask           = nullptr,
        .alphaToCoverageEnable = false,
        .alphaToOneEnable      = false,
    };

    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS,
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT,
        VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext             = nullptr,
        .flags             = 0,
        .dynamicStateCount = sizeof(dynamic_states) / sizeof(VkDynamicState),
        .pDynamicStates    = dynamic_states,
    };

    VkPipelineRenderingCreateInfo pipeline_create{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext                   = nullptr,
        .viewMask                = 0,  // No multiview rendering right now
        .colorAttachmentCount    = color_attachment_formats.size(),
        .pColorAttachmentFormats = color_attachment_formats.data(),
        .depthAttachmentFormat   = depth_attachment_format,
        .stencilAttachmentFormat = stencil_attachment_format,
    };

    VkGraphicsPipelineCreateInfo create_info{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &pipeline_create,
        .flags               = 0,
        .stageCount          = 2,
        .pStages             = stages,
        .pVertexInputState   = &vertex_input_state,
        .pInputAssemblyState = &input_assembly_state,
        .pTessellationState  = nullptr,
        .pViewportState      = nullptr,  // Viewport state is dynamic
        .pRasterizationState = &rasterization_state,
        .pMultisampleState   = &multisample_state,
        .pDepthStencilState  = nullptr,
        .pColorBlendState    = &color_blend_state,
        .pDynamicState       = &dynamic_state,
        .layout              = default_graphics_layout,
        .renderPass          = VK_NULL_HANDLE,
        .subpass             = 0,
        .basePipelineHandle  = VK_NULL_HANDLE,
        .basePipelineIndex   = 0,
    };
    VkPipeline vk_pipeline;
    chk(m_api.vkCreateGraphicsPipelines(m_device,
                                        VK_NULL_HANDLE,
                                        1,
                                        &create_info,
                                        nullptr,
                                        &vk_pipeline));



    m_api.vkDestroyShaderModule(m_device, vert_module, nullptr);
    m_api.vkDestroyShaderModule(m_device, frag_module, nullptr);

    return {.h = reinterpret_cast<uintptr_t>(vk_pipeline)};
}

void Device::Impl::freePipeline(Handle<Pipeline> pipeline) {
    m_api.vkDestroyPipeline(m_device, reinterpret_cast<VkPipeline>(pipeline.h), nullptr);
}

// MARK: Queue

Handle<Queue> Device::Impl::createQueue() {
    return {.h = 0};
}

// MARK: Sempahores

Handle<Semaphore> Device::Impl::createSemaphore(uint64_t initValue) {
    VkSemaphoreTypeCreateInfo semaphore_type{
        .sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext         = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue  = 0,
    };

    VkSemaphoreCreateInfo timeline_create_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &semaphore_type,
        .flags = 0,
    };

    VkSemaphore s = VK_NULL_HANDLE;
    chk(m_api.vkCreateSemaphore(m_device, &timeline_create_info, nullptr, &s));

    // For semaphores, we don't need anything beyond the handle, so we just return the vk handle
    // directly.

    return {.h = reinterpret_cast<uintptr_t>(s)};
}

void Device::Impl::waitSemaphore(Handle<Semaphore> sema, uint64_t value) {
    VkSemaphore         s = reinterpret_cast<VkSemaphore>(sema.h);
    VkSemaphoreWaitInfo wait_info{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext          = nullptr,
        .flags          = 0,
        .semaphoreCount = 1,
        .pSemaphores    = &s,
        .pValues        = &value,
    };
    chk(m_api.vkWaitSemaphores(m_device, &wait_info, UINT64_MAX));
}

void Device::Impl::destroySemaphore(Handle<Semaphore> sema) {
    m_api.vkDestroySemaphore(m_device, reinterpret_cast<VkSemaphore>(sema.h), nullptr);
}

void Device::Impl::chk(VkResult result) {
    if (result == VK_SUCCESS) { return; }

    // TODO: Report the error somehow.
}

Device Device::create(const DeviceDesc& desc) {
    Impl* impl = new Impl;
    if (impl->initialize(desc)) { return Device(impl); }

    return Device(nullptr);
}

Handle<Buffer> Device::malloc(size_t bytes, MEMORY memory) {
    return impl->malloc(bytes, memory);
}

Handle<Buffer> Device::malloc(size_t bytes, size_t align, MEMORY memory) {
    return impl->malloc(bytes, align, memory);
}

void Device::free(Handle<Buffer> buffer) {
    return impl->free(buffer);
}

GpuPtr Device::getDevicePointer(Handle<Buffer> buffer) {
    return impl->getDevicePointer(buffer);
}

Handle<Pipeline> Device::createGraphicsPipeline(ShaderSource      vertex,
                                                ShaderSource      fragment,
                                                const RasterDesc& desc) {
    return impl->createGraphicsPipeline(vertex, fragment, desc);
}

void Device::freePipeline(Handle<Pipeline> pipeline) {
    return impl->freePipeline(pipeline);
}

Handle<Queue> Device::createQueue(/* DEVICE & QUEUE CREATION DETAILS OMITTED */) {
    return impl->createQueue();
}

}  // namespace loon::gpu
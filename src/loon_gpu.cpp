#include "gpu/loon_gpu.h"

#include "containers.h"
#include "gpu_to_vk.h"
#include "utilities.h"
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
    VkPhysicalDevice device                     = VK_NULL_HANDLE;
    uint32_t         graphics_queue_family      = 0;
    uint32_t         transfer_queue_family      = 0;
    uint32_t         async_compute_queue_family = 0;

    // If something goes wrong, an error message will be in here.
    // It will be a string literal, so returning a span is fine - no freeing needed.
    Span<const char> error_string;
};
static PhysicalDeviceInfo    select_physical_device(VkInstance    instance,
                                                    VkSurfaceKHR  surface,
                                                    GpuPreference preference,
                                                    Arena         arena);
static VkDescriptorSetLayout create_default_descriptor_layout(VkDevice               device,
                                                              const VolkDeviceTable& api);
static VkPipelineLayout      create_default_graphics_layout(VkDevice               device,
                                                            const VolkDeviceTable& api,
                                                            VkDescriptorSetLayout  set_layout);

struct ThreadLocalState {
    constexpr static size_t kArenaSize = 256ll * 1024;
    loon::gpu::Allocator    allocator;
    MemoryBlock             arena_memory;
    loon::gpu::Arena        arena;
    ThreadLocalState(const loon::gpu::Allocator& alloc) :
        allocator{alloc},
        arena_memory{allocator.alloc(kArenaSize)},
        arena(arena_memory.ptr, arena_memory.len) {}
    ~ThreadLocalState() { allocator.free(arena_memory); }
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
    Handle<BlendState>        createBlendState(BlendDesc desc);
    void                      freeDepthStencilState(Handle<DepthStencilState> state);
    void                      freeBlendState(Handle<BlendState> state);

    // Queue
    Handle<Queue>         getQueue(QUEUE_TYPE type);
    Handle<CommandBuffer> startCommandRecording(Handle<Queue> queue);
    void                  submit(Handle<Queue> queue, Span<Handle<CommandBuffer>> commandBuffers);
    void                  cancel(Handle<Queue> queue, Span<Handle<CommandBuffer>> commandBuffers);

    // Semaphores
    Handle<Semaphore> createSemaphore(uint64_t initValue);
    void              waitSemaphore(Handle<Semaphore> sema, uint64_t value);
    void              destroySemaphore(Handle<Semaphore> sema);


   private:
    Allocator          m_allocator;
    ProcLogCallback    m_log_callback = nullptr;
    void*              m_log_userdata = nullptr;
    LogLevel           m_log_level    = LogLevel_Off;
    loon::gpu::tls_key m_tls_key;

    VkInstance       m_instance                   = VK_NULL_HANDLE;
    VkSurfaceKHR     m_surface                    = VK_NULL_HANDLE;
    VkPhysicalDevice m_physical_device            = VK_NULL_HANDLE;
    uint32_t         m_graphics_queue_family      = -1;
    uint32_t         m_transfer_queue_family      = -1;
    uint32_t         m_async_compute_queue_family = -1;

    VkDevice        m_device;
    VolkDeviceTable m_api;
    VmaAllocator    m_vma = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_default_descriptor_layout;
    VkPipelineLayout      m_default_graphics_layout;

    ObjectPool<Buffer, kMaxNumBuffers> buffer_pool;

    void              log(LogLevel lvl, Span<const char> msg);
    bool              chk(VkResult result);
    loon::gpu::Arena* get_thread_local_arena();
};

// MARK: Initialization


PhysicalDeviceInfo select_physical_device(VkInstance    instance,
                                          VkSurfaceKHR  surface,
                                          GpuPreference preference,
                                          Arena         arena) {
    uint32_t device_count = 0;
    VkResult vkresult     = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (vkresult != VK_SUCCESS) {
        return {
            .error_string = "Error enumerating physical devices"_sv,
        };
    }

    auto physical_devices = (VkPhysicalDevice*)arena.alloc(sizeof(VkPhysicalDevice) * device_count);
    if (physical_devices == nullptr) {
        return {
            .error_string = "Arena out of memory when enumerating physical devices"_sv,
        };
    }

    vkresult = vkEnumeratePhysicalDevices(instance, &device_count, physical_devices);
    if (vkresult != VK_SUCCESS) { device_count = 0; }

    const bool prefer_integrated = preference == GpuPreference_Integrated;
    const bool prefer_dedicated  = preference == GpuPreference_Discrete;

    PhysicalDeviceInfo best_device_info = {
        .device = VK_NULL_HANDLE,
    };

    for (uint32_t device_idx = 0; device_idx < device_count; ++device_idx) {
        const VkPhysicalDevice     physical_device = physical_devices[device_idx];
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_device, &properties);

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

        VkQueueFamilyProperties* queue_properties = (VkQueueFamilyProperties*)arena.alloc(
            sizeof(VkQueueFamilyProperties) * queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                                 &queue_family_count,
                                                 queue_properties);

        // Find the default queue, as well as any dedicated compute and transfer queue families.
        uint32_t default_queue_family      = ~0;
        uint32_t dedicated_compute_family  = ~0;
        uint32_t dedicated_transfer_family = ~0;
        for (uint32_t queue_family = 0; queue_family < queue_family_count; ++queue_family) {
            const auto& props    = queue_properties[queue_family];
            bool        is_valid = (props.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                            && (props.queueFlags & VK_QUEUE_COMPUTE_BIT);

            const bool is_dedicated_compute = (props.queueFlags & VK_QUEUE_COMPUTE_BIT)
                                              && !(props.queueFlags & VK_QUEUE_GRAPHICS_BIT);
            const bool is_dedicated_transfer = (props.queueFlags & VK_QUEUE_TRANSFER_BIT)
                                               && !(props.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                                               && !(props.queueFlags & VK_QUEUE_COMPUTE_BIT);

            // Need to find the graphics/presentation queue.
            if (surface) {
                VkBool32 surface_supported = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(physical_devices[device_idx],
                                                     queue_family,
                                                     surface,
                                                     &surface_supported);
                is_valid = is_valid && surface_supported;
            }

            if (is_valid) { default_queue_family = queue_family; }
            if (is_dedicated_compute) { dedicated_compute_family = queue_family; }
            if (is_dedicated_transfer) { dedicated_transfer_family = queue_family; }
        }
        arena.free(queue_properties, sizeof(VkQueueFamilyProperties) * queue_family_count);
        // No valid default queue family, so this device won't work.
        if (default_queue_family == ~0u) { continue; }

        // Check device extensions - there's no reasonable upperlimit on the number of device
        // extensions, so need to allocate :(
        uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);

        auto extension_properties = reinterpret_cast<VkExtensionProperties*>(
            arena.alloc(sizeof(VkExtensionProperties) * extension_count));

        vkEnumerateDeviceExtensionProperties(physical_device,
                                             nullptr,
                                             &extension_count,
                                             extension_properties);
        // Need to check against the list of required extensions and make sure all required
        // extensions are available.
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
        arena.free(extension_properties, sizeof(VkExtensionProperties) * extension_count);

        if (!all_extensions_supported) {
            // Invalid device, doesn't support the extensions we need.
            continue;
        }

        // If we don't have a "best" device yet, let's use this one.
        if (best_device_info.device == VK_NULL_HANDLE) {
            best_device_info = {
                .device                     = physical_device,
                .graphics_queue_family      = default_queue_family,
                .transfer_queue_family      = dedicated_transfer_family,
                .async_compute_queue_family = dedicated_compute_family,
            };
        }

        const bool is_integrated = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        if (is_integrated && prefer_integrated) {
            best_device_info = {
                .device                     = physical_device,
                .graphics_queue_family      = default_queue_family,
                .transfer_queue_family      = dedicated_transfer_family,
                .async_compute_queue_family = dedicated_compute_family,
            };
        }

        const bool is_dedicated = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        if (is_dedicated && prefer_dedicated) {
            best_device_info = {
                .device                     = physical_device,
                .graphics_queue_family      = default_queue_family,
                .transfer_queue_family      = dedicated_transfer_family,
                .async_compute_queue_family = dedicated_compute_family,
            };
        }
    }

    return best_device_info;
}

VkDescriptorSetLayout create_default_descriptor_layout(VkDevice               device,
                                                       const VolkDeviceTable& api) {
    VkDescriptorBindingFlags descVariableFlag
        = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
          | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
          | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount  = 1,
        .pBindingFlags = &descVariableFlag};

    VkDescriptorSetLayoutBinding binding{
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = kMaxTextureHeapSize,
        .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext        = &binding_flags,
        .flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = 1,
        .pBindings    = &binding};

    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkResult              result                = api.vkCreateDescriptorSetLayout(device,
                                                      &descriptor_set_layout_create_info,
                                                      nullptr,
                                                      &descriptor_set_layout);
    return result == VK_SUCCESS ? descriptor_set_layout : VK_NULL_HANDLE;
}

VkPipelineLayout create_default_graphics_layout(VkDevice               device,
                                                const VolkDeviceTable& api,
                                                VkDescriptorSetLayout  set_layout) {
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
        .setLayoutCount         = 1,
        .pSetLayouts            = &set_layout,
        .pushConstantRangeCount = 2,
        .pPushConstantRanges    = push_constant_ranges,
    };

    VkPipelineLayout pipeline_layout;
    VkResult result = api.vkCreatePipelineLayout(device, &create_info, nullptr, &pipeline_layout);
    return result == VK_SUCCESS ? pipeline_layout : VK_NULL_HANDLE;
}

bool Device::Impl::initialize(const DeviceDesc& desc) {
    // Setup basics: Logging, Allocator, TLS:
    if (desc.alloc_callback) { m_allocator = Allocator(desc.alloc_callback, desc.alloc_userdata); }
    if (desc.log_callback) {
        m_log_callback = desc.log_callback;
        m_log_userdata = desc.log_userdata;
        m_log_level    = desc.log_level;
    }
    m_tls_key = loon::gpu::tls_alloc([](void* data) {
        auto state = reinterpret_cast<ThreadLocalState*>(data);
        state->~ThreadLocalState();
    });

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

    if (!chk(vkCreateInstance(&instance_info, nullptr, &m_instance))) {
        log(LogLevel_Error, "Failed to create vulkan instance");
        return false;
    }
    volkLoadInstanceOnly(m_instance);

    // Create surface if requested:
    if (desc.native_instance_handle != 0 || desc.native_window_handle != 0) {
#if defined(VK_USE_PLATFORM_WIN32_KHR)
        const VkWin32SurfaceCreateInfoKHR surface_info = {
            .sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            .pNext     = nullptr,
            .flags     = 0,
            .hinstance = (HINSTANCE)desc.native_instance_handle,
            .hwnd      = (HWND)desc.native_window_handle,
        };
        if (!chk(vkCreateWin32SurfaceKHR(m_instance, &surface_info, nullptr, &m_surface))) {
            return false;
        }
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
        const VkXlibSurfaceCreateInfoKHR surface_info{
            .sType  = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
            .pNext  = nullptr,
            .flags  = 0,
            .dpy    = reinterpret_cast<Display*>(desc.native_instance_handle),
            .window = desc.native_window_handle,
        };
        if (!chk(vkCreateXlibSurfaceKHR(m_instance, &surface_info, nullptr, &m_surface))) {
            return false;
        }
#elif defined(VK_USE_PLATFORM_METAL_EXT)
        const VkMetalSurfaceCreateInfoEXT surface_info{
            .sType  = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
            .pNext  = nullptr,
            .flags  = 0,
            .pLayer = reinterpret_cast<CAMetalLayer*>(desc.native_window_handle),
        };
        VkResult result = vkCreateMetalSurfaceEXT(m_instance, &surface_info, nullptr, &m_surface);
        if (result != VK_SUCCESS) return false;
#else
        default: log(LogLevel_Error, "Unsupported surface source"); return false;
#endif
    }

    auto physical_device_info = select_physical_device(m_instance,
                                                       m_surface,
                                                       desc.gpu_preference,
                                                       *get_thread_local_arena());
    if (physical_device_info.device == VK_NULL_HANDLE) {
        log(LogLevel_Error, physical_device_info.error_string);
        return false;
    }
    m_physical_device            = physical_device_info.device;
    m_graphics_queue_family      = physical_device_info.graphics_queue_family;
    m_transfer_queue_family      = physical_device_info.transfer_queue_family;
    m_async_compute_queue_family = physical_device_info.async_compute_queue_family;

    // Create logical device and request queues.:

    float                             queue_priority = 1.0f;
    Stack<VkDeviceQueueCreateInfo, 3> queue_create_infos;
    queue_create_infos.push(VkDeviceQueueCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = 0,
        .queueFamilyIndex = m_graphics_queue_family,
        .queueCount       = 1,
        .pQueuePriorities = &queue_priority,
    });

    if (m_async_compute_queue_family != -1
        && m_async_compute_queue_family != m_graphics_queue_family) {
        queue_create_infos.push({
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext            = nullptr,
            .flags            = 0,
            .queueFamilyIndex = m_async_compute_queue_family,
            .queueCount       = 1,
            .pQueuePriorities = &queue_priority,
        });
    }

    if (m_transfer_queue_family != -1 && m_transfer_queue_family != m_async_compute_queue_family
        && m_transfer_queue_family != m_graphics_queue_family) {
        queue_create_infos.push({
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext            = nullptr,
            .flags            = 0,
            .queueFamilyIndex = m_transfer_queue_family,
            .queueCount       = 1,
            .pQueuePriorities = &queue_priority,
        });
    }

    // TODO: Check required limits against our limits.
    VkPhysicalDeviceVulkan13Features vulkan_13_features{};
    vulkan_13_features.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan_13_features.pNext            = nullptr;
    vulkan_13_features.dynamicRendering = true;
    vulkan_13_features.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features vulkan_12_features{};
    vulkan_12_features.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan_12_features.pNext               = &vulkan_13_features;
    vulkan_12_features.timelineSemaphore   = true;
    vulkan_12_features.bufferDeviceAddress = true;
    vulkan_12_features.descriptorIndexing  = true;
    vulkan_12_features.descriptorBindingSampledImageUpdateAfterBind = true;
    vulkan_12_features.descriptorBindingPartiallyBound              = true;
    vulkan_12_features.descriptorBindingUpdateUnusedWhilePending    = true;
    vulkan_12_features.descriptorBindingVariableDescriptorCount     = true;

    VkPhysicalDeviceVulkan11Features vulkan_11_features{};
    vulkan_11_features.sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan_11_features.pNext                = &vulkan_12_features;
    vulkan_11_features.shaderDrawParameters = true;

    VkPhysicalDeviceFeatures2 device_features{
        .sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext    = &vulkan_11_features,
        .features = {},
    };

    VkDeviceCreateInfo create_info{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &device_features,
        .flags                   = 0,
        .queueCreateInfoCount    = queue_create_infos.size(),
        .pQueueCreateInfos       = queue_create_infos.data(),
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = loon::gpu::kRequiredDeviceExtensionsCount,
        .ppEnabledExtensionNames = loon::gpu::kRequiredDeviceExtensions,
        .pEnabledFeatures        = nullptr,
    };

    if (!chk(vkCreateDevice(m_physical_device, &create_info, nullptr, &m_device))) {
        log(LogLevel_Error, "Failed to create vulkan device"_sv);
        return false;
    }

    volkLoadDeviceTable(&m_api, m_device);

    // Initialize VMA:

    VmaAllocatorCreateInfo vma_create_info{
        .flags                          = 0,
        .physicalDevice                 = m_physical_device,
        .device                         = m_device,
        .preferredLargeHeapBlockSize    = 0,
        .pAllocationCallbacks           = nullptr,
        .pDeviceMemoryCallbacks         = nullptr,
        .pHeapSizeLimit                 = nullptr,
        .pVulkanFunctions               = nullptr,
        .instance                       = m_instance,
        .vulkanApiVersion               = VK_API_VERSION_1_3,
        .pTypeExternalMemoryHandleTypes = nullptr,
    };
    VmaVulkanFunctions vulkan_functions;
    vmaImportVulkanFunctionsFromVolk(&vma_create_info, &vulkan_functions);
    vma_create_info.pVulkanFunctions = &vulkan_functions;

    vma_create_info.flags            = 0;
    vma_create_info.vulkanApiVersion = VK_API_VERSION_1_3;
    vma_create_info.physicalDevice   = m_physical_device;
    vmaCreateAllocator(&vma_create_info, &m_vma);

    m_default_descriptor_layout = create_default_descriptor_layout(m_device, m_api);
    m_default_graphics_layout
        = create_default_graphics_layout(m_device, m_api, m_default_descriptor_layout);

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
    chk(vmaCreateBuffer(m_vma, &create_info, &alloc_info, &vk_buffer, &vk_allocation, nullptr));

    const uint32_t buffer_idx = buffer_pool.get();
    buffer_pool[buffer_idx]   = {
          .vk_buffer     = vk_buffer,
          .vk_allocation = vk_allocation,
    };
    return {.h = buffer_idx};
}

void Device::Impl::free(Handle<Buffer> buffer) {
    auto& b = buffer_pool[buffer.h];
    vmaDestroyBuffer(m_vma, b.vk_buffer, b.vk_allocation);
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
            .stage               = VK_SHADER_STAGE_FRAGMENT_BIT,
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
        color_blend_attachment_states.push(loon::gpu::bridge(desc.blendstate));
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

    VkPipelineViewportStateCreateInfo viewport_state{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = 0,
        .viewportCount = 0,
        .scissorCount  = 0,
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
        .pViewportState      = &viewport_state,  // Viewport state is dynamic
        .pRasterizationState = &rasterization_state,
        .pMultisampleState   = &multisample_state,
        .pDepthStencilState  = nullptr,
        .pColorBlendState    = &color_blend_state,
        .pDynamicState       = &dynamic_state,
        .layout              = m_default_graphics_layout,
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

Handle<Queue> Device::Impl::getQueue(QUEUE_TYPE type) {
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

void Device::Impl::log(LogLevel lvl, Span<const char> msg) {
    m_log_callback(lvl, msg, m_log_userdata);
};

bool Device::Impl::chk(VkResult result) {
    if (result == VK_SUCCESS) { return true; }

    switch (result) {
        case VK_NOT_READY: log(LogLevel_Error, "VK_NOT_READY"_sv); break;
        case VK_TIMEOUT: log(LogLevel_Error, "VK_TIMEOUT"_sv); break;
        case VK_EVENT_SET: log(LogLevel_Error, "VK_EVENT_SET"_sv); break;
        case VK_EVENT_RESET: log(LogLevel_Error, "VK_EVENT_RESET"_sv); break;
        case VK_INCOMPLETE: log(LogLevel_Error, "VK_INCOMPLETE"_sv); break;
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            log(LogLevel_Error, "VK_ERROR_OUT_OF_HOST_MEMORY"_sv);
            break;
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            log(LogLevel_Error, "VK_ERROR_OUT_OF_DEVICE_MEMORY");
            break;
        case VK_ERROR_INITIALIZATION_FAILED:
            log(LogLevel_Error, "VK_ERROR_INITIALIZATION_FAILED");
            break;
        case VK_ERROR_DEVICE_LOST: log(LogLevel_Error, "VK_ERROR_DEVICE_LOST"); break;
        case VK_ERROR_MEMORY_MAP_FAILED: log(LogLevel_Error, "VK_ERROR_MEMORY_MAP_FAILED"); break;
        case VK_ERROR_LAYER_NOT_PRESENT: log(LogLevel_Error, "VK_ERROR_LAYER_NOT_PRESENT"); break;
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            log(LogLevel_Error, "VK_ERROR_EXTENSION_NOT_PRESENT");
            break;
        case VK_ERROR_FEATURE_NOT_PRESENT:
            log(LogLevel_Error, "VK_ERROR_FEATURE_NOT_PRESENT");
            break;
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            log(LogLevel_Error, "VK_ERROR_INCOMPATIBLE_DRIVER");
            break;
        case VK_ERROR_TOO_MANY_OBJECTS: log(LogLevel_Error, "VK_ERROR_TOO_MANY_OBJECTS"); break;
        case VK_ERROR_FORMAT_NOT_SUPPORTED:
            log(LogLevel_Error, "VK_ERROR_FORMAT_NOT_SUPPORTED");
            break;
        case VK_ERROR_FRAGMENTED_POOL: log(LogLevel_Error, "VK_ERROR_FRAGMENTED_POOL"); break;
        case VK_ERROR_UNKNOWN: log(LogLevel_Error, "VK_ERROR_UNKNOWN"); break;
        // case VK_ERROR_VALIDATION_FAILED: log(LogLevel_Error, "VK_ERROR_VALIDATION_FAILED");
        // break;
        case VK_ERROR_OUT_OF_POOL_MEMORY: log(LogLevel_Error, "VK_ERROR_OUT_OF_POOL_MEMORY"); break;
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:
            log(LogLevel_Error, "VK_ERROR_INVALID_EXTERNAL_HANDLE");
            break;
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
            log(LogLevel_Error, "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS");
            break;
        case VK_ERROR_FRAGMENTATION: log(LogLevel_Error, "VK_ERROR_FRAGMENTATION"); break;
        case VK_PIPELINE_COMPILE_REQUIRED:
            log(LogLevel_Error, "VK_PIPELINE_COMPILE_REQUIRED");
            break;
        // case VK_ERROR_NOT_PERMITTED: log(LogLevel_Error, "VK_ERROR_NOT_PERMITTED"); break;
        case VK_ERROR_SURFACE_LOST_KHR: log(LogLevel_Error, "VK_ERROR_SURFACE_LOST_KHR"); break;
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
            log(LogLevel_Error, "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR");
            break;
        case VK_SUBOPTIMAL_KHR: log(LogLevel_Error, "VK_SUBOPTIMAL_KHR"); break;
        case VK_ERROR_OUT_OF_DATE_KHR: log(LogLevel_Error, "VK_ERROR_OUT_OF_DATE_KHR"); break;
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
            log(LogLevel_Error, "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR");
            break;
        case VK_ERROR_INVALID_SHADER_NV: log(LogLevel_Error, "VK_ERROR_INVALID_SHADER_NV"); break;
        case VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR:
            log(LogLevel_Error, "VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR:
            log(LogLevel_Error, "VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR:
            log(LogLevel_Error, "VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR:
            log(LogLevel_Error, "VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR:
            log(LogLevel_Error, "VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR:
            log(LogLevel_Error, "VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:
            log(LogLevel_Error, "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT");
            break;
        // case VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT:
        //     log(LogLevel_Error, "VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT");
        //     break;
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
            log(LogLevel_Error, "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT");
            break;
        case VK_THREAD_IDLE_KHR: log(LogLevel_Error, "VK_THREAD_IDLE_KHR"); break;
        case VK_THREAD_DONE_KHR: log(LogLevel_Error, "VK_THREAD_DONE_KHR"); break;
        case VK_OPERATION_DEFERRED_KHR: log(LogLevel_Error, "VK_OPERATION_DEFERRED_KHR"); break;
        case VK_OPERATION_NOT_DEFERRED_KHR:
            log(LogLevel_Error, "VK_OPERATION_NOT_DEFERRED_KHR");
            break;
        case VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR:
            log(LogLevel_Error, "VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR");
            break;
        case VK_ERROR_COMPRESSION_EXHAUSTED_EXT:
            log(LogLevel_Error, "VK_ERROR_COMPRESSION_EXHAUSTED_EXT");
            break;
        case VK_INCOMPATIBLE_SHADER_BINARY_EXT:
            log(LogLevel_Error, "VK_INCOMPATIBLE_SHADER_BINARY_EXT");
            break;
        // case VK_PIPELINE_BINARY_MISSING_KHR:
        //     log(LogLevel_Error, "VK_PIPELINE_BINARY_MISSING_KHR");
        //     break;
        // case VK_ERROR_NOT_ENOUGH_SPACE_KHR:
        //     log(LogLevel_Error, "VK_ERROR_NOT_ENOUGH_SPACE_KHR");
        //     break;
        default: log(LogLevel_Error, "Unknown error"_sv); break;
    }

    return false;
}

Arena* Device::Impl::get_thread_local_arena() {
    auto state = reinterpret_cast<ThreadLocalState*>(loon::gpu::tls_get_data(m_tls_key));
    if (state == nullptr) {
        auto tls_block = m_allocator.alloc(sizeof(ThreadLocalState));
        if (tls_block.ptr == nullptr) {
            log(LogLevel_Error, "Allocator out of memory"_sv);
            return nullptr;
        }
        state = ::new (tls_block.ptr) ThreadLocalState(m_allocator);
        loon::gpu::tls_set_data(m_tls_key, state);
    }
    return &state->arena;
}

Device Device::create(const DeviceDesc& desc) {
    Impl* impl = new Impl;
    if (impl->initialize(desc)) { return Device(impl); }

    return Device(nullptr);
}

Device::~Device() {
    if (impl) { delete impl; }
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

Handle<Queue> Device::getQueue(QUEUE_TYPE type) {
    return impl->getQueue(type);
}

Handle<CommandBuffer> Device::startCommandRecording(Handle<Queue> queue) {
    return {};
}

void Device::submit(Handle<Queue> queue, Span<const Handle<CommandBuffer>> commandBuffers) {}

void Device::cancel(Handle<Queue> queue, Span<const Handle<CommandBuffer>> commandBuffers) {}


// MARK: Commmand Buffer

void Handle<CommandBuffer>::setPipeline(Handle<Pipeline> pipeline) {}
void Handle<CommandBuffer>::beginRenderPass(RenderPassDesc desc) {}

void Handle<CommandBuffer>::endRenderPass() {}

void Handle<CommandBuffer>::draw(GpuPtr   vertexDataGpu,
                                 GpuPtr   fragmentDataGpu,
                                 uint32_t vertexCount,
                                 uint32_t instanceCount,
                                 uint32_t firstVertex,
                                 uint32_t firstInstance) {}

}  // namespace loon::gpu
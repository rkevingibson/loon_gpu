#include "gpu/loon_gpu.h"

#include <cstddef>

#include "containers.h"
#include "gpu_to_vk.h"
#include "utilities.h"
#include "vma_usage.h"
#include "volk.h"
#include "vulkan/vulkan_core.h"

namespace loon::gpu {

template class Function<void>;

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

struct Texture {
    VkImage         vk_image;
    VmaAllocation   vk_allocation;
    VkImageViewType vk_type = VK_IMAGE_VIEW_TYPE_2D;
};

struct TextureView {
    VkImageView vk_image_view;
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

struct Surface {
    static constexpr uint32_t kMaxSwapchainImages = 8;

    VkSurfaceKHR   surface   = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;

    // No swapchain will support all the formats, but makes memory allocation simpler.
    static constexpr size_t kMaxNumFormats      = FORMAT_ValidCount;
    static constexpr size_t kMaxNumPresentModes = PRESENT_MODE_VALID_COUNT;
    FORMAT                  supported_formats[kMaxNumFormats];
    PRESENT_MODE            supported_present_modes[kMaxNumPresentModes];
    size_t                  num_supported_formats       = 0;
    size_t                  num_supported_present_modes = 0;

    VkFormat   swapchain_format = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchain_extent = {0, 0};

    uint32_t        current_image_idx = 0;
    uint32_t        image_count       = 0;
    Handle<Texture> swapchain_images[kMaxSwapchainImages];

    // Semaphores are ticked around the ring buffer independently from images, since we need to know
    // which semaphore to use before we know which image index we have.
    uint32_t    current_semaphore_idx = 0;
    VkSemaphore acquire_semaphores[kMaxSwapchainImages];
    VkSemaphore present_semaphores[kMaxSwapchainImages];

    // We store the command buffer that recording
    // the transition to PRESENT_KHR, so we can
    // signal an extra semaphore on submission.
    VkCommandBuffer transitioning_command[kMaxSwapchainImages];
};

struct Queue {
    struct Event {
        uint64_t       completed_time;
        Function<void> callback;
    };

    VkQueue       queue          = VK_NULL_HANDLE;
    VkCommandPool command_pool   = VK_NULL_HANDLE;
    VkSemaphore   timeline       = VK_NULL_HANDLE;
    uint32_t      queue_family   = 0;
    uint64_t      timeline_value = 0;
    Vector<Event> pending_events;
};

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

struct BufferAndOffset {
    VkBuffer buffer;
    uint32_t offset;
};

struct Device::Impl {
    bool initialize(const DeviceDesc& desc);

    void shutdown();

    // Surface:
    SurfaceCapabilities get_surface_capabilities();
    bool                configure_surface(const SurfaceConfiguration& config);
    void                unconfigure_surface();
    SurfaceTextureInfo  get_current_texture();
    void                present(Handle<Queue> queue);

    // Buffers:
    Handle<Buffer>  malloc(size_t bytes, MEMORY memory = MEMORY_DEFAULT);
    Handle<Buffer>  malloc(size_t bytes, size_t align, MEMORY memory = MEMORY_DEFAULT);
    void            free(Handle<Buffer> buffer);
    GpuPtr          getDevicePointer(Handle<Buffer> buffer);
    BufferAndOffset buffer_and_offset_from_ptr(GpuPtr ptr);


    // Textures:
    Handle<Texture>     createTexture(const GpuTextureDesc& desc);
    Handle<TextureHeap> createTextureHeap(size_t size);
    Handle<TextureView> createTextureView(Handle<Texture> texture, TextureViewDesc desc);

    uint32_t addTextureViewToHeap(Handle<TextureHeap>, Handle<TextureView>);
    void     removeTextureViewFromHeap(Handle<TextureHeap>, uint32_t);

    void free(Handle<Texture>);
    void free(Handle<TextureHeap>);
    void free(Handle<TextureView> view);

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
    Handle<Queue> getQueue(QUEUE_TYPE type);
    CommandBuffer startCommandRecording(Handle<Queue> queue);
    void          submit(Handle<Queue>             queue,
                         Span<const CommandBuffer> commandBuffers,
                         Span<const SemaphoreInfo> wait_semaphores,
                         Span<const SemaphoreInfo> signal_semaphores);
    void          cancel(Handle<Queue> queue, Span<Handle<CommandBuffer>> commandBuffers);

    void on_submitted_work_completed(Handle<Queue> queue, Function<void>&& fn);
    void process_events(Handle<Queue> queue);

    // Semaphores
    Handle<Semaphore> createSemaphore(uint64_t initValue);
    void              waitSemaphore(Handle<Semaphore> sema, uint64_t value);
    void              destroySemaphore(Handle<Semaphore> sema);


   private:
    friend class CommandBuffer;

    Allocator          m_allocator;
    ProcLogCallback    m_log_callback = nullptr;
    void*              m_log_userdata = nullptr;
    LogLevel           m_log_level    = LogLevel_Off;
    loon::gpu::tls_key m_tls_key;

    VkInstance       m_instance                   = VK_NULL_HANDLE;
    VkPhysicalDevice m_physical_device            = VK_NULL_HANDLE;
    uint32_t         m_graphics_queue_family      = -1;
    uint32_t         m_transfer_queue_family      = -1;
    uint32_t         m_async_compute_queue_family = -1;

    Surface m_surface;

    VkDevice        m_device;
    VolkDeviceTable m_api;
    VmaAllocator    m_vma = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_default_descriptor_layout;
    VkPipelineLayout      m_default_graphics_layout;

    ObjectPool<Buffer, kMaxNumBuffers>           m_buffer_pool;
    ObjectPool<Texture, kMaxNumTextures>         m_texture_pool;
    ObjectPool<TextureView, kMaxNumTextureViews> m_texture_view_pool;

    Queue m_queues[QUEUE_VALID_COUNT];

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
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (desc.native_instance_handle != 0 || desc.native_window_handle != 0) {
#if defined(VK_USE_PLATFORM_WIN32_KHR)
        const VkWin32SurfaceCreateInfoKHR surface_info = {
            .sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            .pNext     = nullptr,
            .flags     = 0,
            .hinstance = (HINSTANCE)desc.native_instance_handle,
            .hwnd      = (HWND)desc.native_window_handle,
        };
        if (!chk(vkCreateWin32SurfaceKHR(m_instance, &surface_info, nullptr, &surface))) {
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
        if (!chk(vkCreateXlibSurfaceKHR(m_instance, &surface_info, nullptr, &surface))) {
            return false;
        }
#elif defined(VK_USE_PLATFORM_METAL_EXT)
        const VkMetalSurfaceCreateInfoEXT surface_info{
            .sType  = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
            .pNext  = nullptr,
            .flags  = 0,
            .pLayer = reinterpret_cast<CAMetalLayer*>(desc.native_window_handle),
        };
        VkResult result = vkCreateMetalSurfaceEXT(m_instance, &surface_info, nullptr, &surface);
        if (result != VK_SUCCESS) return false;
#else
        default: log(LogLevel_Error, "Unsupported surface source"); return false;
#endif
    }

    Arena arena = *get_thread_local_arena();

    auto physical_device_info
        = select_physical_device(m_instance, surface, desc.gpu_preference, arena);
    if (physical_device_info.device == VK_NULL_HANDLE) {
        log(LogLevel_Error, physical_device_info.error_string);
        return false;
    }
    m_physical_device            = physical_device_info.device;
    m_graphics_queue_family      = physical_device_info.graphics_queue_family;
    m_transfer_queue_family      = physical_device_info.transfer_queue_family;
    m_async_compute_queue_family = physical_device_info.async_compute_queue_family;

    // Create logical device and request queues.:

    float                         queue_priority = 1.0f;
    Span<VkDeviceQueueCreateInfo> queue_create_infos;
    queue_create_infos = concat(&arena,
                                queue_create_infos,
                                VkDeviceQueueCreateInfo{
                                    .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                    .pNext            = nullptr,
                                    .flags            = 0,
                                    .queueFamilyIndex = m_graphics_queue_family,
                                    .queueCount       = 1,
                                    .pQueuePriorities = &queue_priority,
                                });

    if (m_async_compute_queue_family != -1
        && m_async_compute_queue_family != m_graphics_queue_family) {
        queue_create_infos = concat(&arena,
                                    queue_create_infos,
                                    {
                                        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                        .pNext = nullptr,
                                        .flags = 0,
                                        .queueFamilyIndex = m_async_compute_queue_family,
                                        .queueCount       = 1,
                                        .pQueuePriorities = &queue_priority,
                                    });
    }

    if (m_transfer_queue_family != -1 && m_transfer_queue_family != m_async_compute_queue_family
        && m_transfer_queue_family != m_graphics_queue_family) {
        queue_create_infos = concat(&arena,
                                    queue_create_infos,
                                    {
                                        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                        .pNext = nullptr,
                                        .flags = 0,
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
        .queueCreateInfoCount    = static_cast<uint32_t>(queue_create_infos.size()),
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

    // Initialize the surface objects:
    m_surface.surface = surface;
    const VkSemaphoreCreateInfo semaphore_create_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    for (uint32_t i = 0; i < Surface::kMaxSwapchainImages; ++i) {
        VkSemaphore s = VK_NULL_HANDLE;
        chk(m_api.vkCreateSemaphore(m_device, &semaphore_create_info, nullptr, &s));
        m_surface.acquire_semaphores[i] = s;

        chk(m_api.vkCreateSemaphore(m_device, &semaphore_create_info, nullptr, &s));
        m_surface.present_semaphores[i] = s;
    }

    m_default_descriptor_layout = create_default_descriptor_layout(m_device, m_api);
    m_default_graphics_layout
        = create_default_graphics_layout(m_device, m_api, m_default_descriptor_layout);

    return true;
}

void Device::Impl::shutdown() {
    vkDestroyInstance(m_instance, nullptr);
    volkFinalize();
}

// MARK: Surface

SurfaceCapabilities Device::Impl::get_surface_capabilities() {
    // Surface formats:
    uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device,
                                         m_surface.surface,
                                         &format_count,
                                         nullptr);

    Arena arena = *get_thread_local_arena();  // Just copy the arena so all memory will be "freed"
                                              // at function exit.

    auto vk_formats = reinterpret_cast<VkSurfaceFormatKHR*>(
        arena.alloc(sizeof(VkSurfaceFormatKHR) * format_count));
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device,
                                         m_surface.surface,
                                         &format_count,
                                         vk_formats);

    m_surface.num_supported_formats = 0;
    for (uint32_t i = 0; i < format_count; ++i) {
        FORMAT fmt = bridge(vk_formats[i].format);
        if (fmt < FORMAT_ValidCount) {
            m_surface.supported_formats[m_surface.num_supported_formats] = fmt;
            m_surface.num_supported_formats++;
        }
    }

    // Present modes:
    uint32_t present_mode_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device,
                                              m_surface.surface,
                                              &present_mode_count,
                                              nullptr);
    auto vk_modes = reinterpret_cast<VkPresentModeKHR*>(
        arena.alloc(sizeof(VkPresentModeKHR) * present_mode_count));

    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device,
                                              m_surface.surface,
                                              &present_mode_count,
                                              vk_modes);

    for (uint32_t i = 0; i < present_mode_count; ++i) {
        PRESENT_MODE mode = bridge(vk_modes[i]);
        if (mode < PRESENT_MODE_VALID_COUNT) {
            m_surface.supported_present_modes[m_surface.num_supported_present_modes] = mode;
            m_surface.num_supported_present_modes++;
        }
    }

    // Capabilities has usages
    VkSurfaceCapabilitiesKHR vk_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device,
                                              m_surface.surface,
                                              &vk_capabilities);

    return SurfaceCapabilities{
        .usages  = bridge_usage_flags(vk_capabilities.supportedUsageFlags),
        .formats = Span<const FORMAT>(m_surface.supported_formats, m_surface.num_supported_formats),
        .present_modes = Span<const PRESENT_MODE>(m_surface.supported_present_modes,
                                                  m_surface.num_supported_present_modes),
    };
}

bool Device::Impl::configure_surface(const SurfaceConfiguration& config) {
    // Create the VkSwapchain based on the configuration
    VkSurfaceCapabilitiesKHR vk_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device,
                                              m_surface.surface,
                                              &vk_capabilities);

    uint32_t image_count = vk_capabilities.minImageCount + 1;
    if (vk_capabilities.maxImageCount > 0 && image_count > vk_capabilities.maxImageCount) {
        image_count = vk_capabilities.maxImageCount;
    }

    const auto extent = VkExtent2D{
        .width  = config.width,
        .height = config.height,
    };

    VkSwapchainCreateInfoKHR swapchain_info{
        .sType                 = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext                 = nullptr,
        .flags                 = 0,
        .surface               = m_surface.surface,
        .minImageCount         = image_count,
        .imageFormat           = loon::gpu::bridge(config.format),
        .imageColorSpace       = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,  // TODO: Colorspace support?
        .imageExtent           = extent,
        .imageArrayLayers      = 1,
        .imageUsage            = loon::gpu::bridge_usage_flags(config.usages),
        .imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE,  // We only support one queue.
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .preTransform          = vk_capabilities.currentTransform,
        .compositeAlpha        = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode           = loon::gpu::bridge(config.present_mode),
        .clipped               = true,
        .oldSwapchain          = m_surface.swapchain,
    };

    if (!chk(
            m_api.vkCreateSwapchainKHR(m_device, &swapchain_info, nullptr, &m_surface.swapchain))) {
        log(LogLevel_Error, "Failed in call to vkCreateSwapchainKHR"_sv);
        return false;
    }

    image_count = 0;
    if (!chk(m_api.vkGetSwapchainImagesKHR(m_device, m_surface.swapchain, &image_count, nullptr))) {
        log(LogLevel_Error, "Failed in call to vkGetSwapchainImagesKHR"_sv);
        return false;
    }

    if (image_count > Surface::kMaxSwapchainImages) {
        log(LogLevel_Error, "Swapchain creating too many images"_sv);
        return false;
    }

    VkImage swapchain_images[Surface::kMaxSwapchainImages];

    if (!chk(m_api.vkGetSwapchainImagesKHR(m_device,
                                           m_surface.swapchain,
                                           &image_count,
                                           swapchain_images))) {
        log(LogLevel_Error, "Swapchain failed to retrieve images"_sv);
        return false;
    }

    // Convert the swapchain images here to handles, by inserting them into the object pool.
    for (int i = 0; i < image_count; ++i) {
        const uint32_t handle_idx  = m_texture_pool.get();
        m_texture_pool[handle_idx] = {
            .vk_image = swapchain_images[i],
        };
        m_surface.swapchain_images[i] = Handle<Texture>{.h = handle_idx};
    }

    return true;
}

void Device::Impl::unconfigure_surface() {
    m_api.vkDestroySwapchainKHR(m_device, m_surface.swapchain, nullptr);
    m_surface.swapchain         = VK_NULL_HANDLE;
    m_surface.image_count       = 0;
    m_surface.current_image_idx = 0;
}

SurfaceTextureInfo Device::Impl::get_current_texture() {
    auto semaphore = m_surface.acquire_semaphores[m_surface.current_semaphore_idx];
    m_surface.current_semaphore_idx
        = (m_surface.current_semaphore_idx + 1) % Surface::kMaxSwapchainImages;
    VkAcquireNextImageInfoKHR acquire_info{
        .sType      = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,
        .pNext      = nullptr,
        .swapchain  = m_surface.swapchain,
        .timeout    = 0,
        .semaphore  = semaphore,
        .fence      = VK_NULL_HANDLE,
        .deviceMask = 1,
    };
    uint32_t           image_idx = 0;
    VkResult           result = m_api.vkAcquireNextImage2KHR(m_device, &acquire_info, &image_idx);
    SurfaceTextureInfo info{
        .status            = SurfaceTextureInfo::STATUS_SUCCESS,
        .texture           = m_surface.swapchain_images[image_idx],
        .acquire_semaphore = {.h = reinterpret_cast<uintptr_t>(semaphore)},
    };
    m_surface.current_image_idx = image_idx;

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        info.status = SurfaceTextureInfo::STATUS_OUT_OF_DATE;
    } else if (result == VK_SUBOPTIMAL_KHR) {
        info.status = SurfaceTextureInfo::STATUS_SUBOPTIMAL;
    } else if (result < 0) {
        log(LogLevel_Error, "Error in swapchain acquireNextImage"_sv);
        info.status = SurfaceTextureInfo::STATUS_ERROR;
    }

    return info;
}

void Device::Impl::present(Handle<Queue> queue) {
    auto presenting_texture_handle = m_surface.swapchain_images[m_surface.current_image_idx];

    VkPresentInfoKHR present_info{
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext              = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = &m_surface.present_semaphores[m_surface.current_semaphore_idx],
        .swapchainCount     = 1,
        .pSwapchains        = &m_surface.swapchain,
        .pImageIndices      = &m_surface.current_image_idx,
        .pResults           = nullptr,
    };

    m_api.vkQueuePresentKHR(m_queues[queue.h].queue, &present_info);
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

    const uint32_t buffer_idx = m_buffer_pool.get();
    m_buffer_pool[buffer_idx] = {
        .vk_buffer     = vk_buffer,
        .vk_allocation = vk_allocation,
    };
    return {.h = buffer_idx};
}

void Device::Impl::free(Handle<Buffer> buffer) {
    auto& b = m_buffer_pool[buffer.h];
    vmaDestroyBuffer(m_vma, b.vk_buffer, b.vk_allocation);
}

GpuPtr Device::Impl::getDevicePointer(Handle<Buffer> buffer) {
    VkBufferDeviceAddressInfo addr_info{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = m_buffer_pool[buffer.h].vk_buffer,
    };
    return m_api.vkGetBufferDeviceAddress(m_device, &addr_info);
}

BufferAndOffset Device::Impl::buffer_and_offset_from_ptr(GpuPtr ptr) {
    // TODO: On buffer creation, store the ptr in a sorted list, so we can look it up later.
    return {.buffer = VK_NULL_HANDLE, .offset = 0};
}

// MARK: Textures

Handle<Texture> Device::Impl::createTexture(const GpuTextureDesc& desc) {
    return {};
}

Handle<TextureView> Device::Impl::createTextureView(Handle<Texture> tex, TextureViewDesc desc) {
    auto& texture = m_texture_pool[tex.h];
    const VkImageViewCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags      = 0,
        .image      = texture.vk_image,
        .viewType   = texture.vk_type,
        .format     = bridge(desc.format),
        .components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                        VK_COMPONENT_SWIZZLE_IDENTITY,
                        VK_COMPONENT_SWIZZLE_IDENTITY,
                        VK_COMPONENT_SWIZZLE_IDENTITY,},
        .subresourceRange = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = desc.baseMip,
        .levelCount = desc.mipCount,
        .baseArrayLayer = desc.baseLayer,
        .layerCount = desc.layerCount,
        },
    };
    VkImageView image_view;
    if (!chk(m_api.vkCreateImageView(m_device, &info, nullptr, &image_view))) { return {}; }
    const auto handle = m_texture_view_pool.get();

    m_texture_view_pool[handle].vk_image_view = image_view;
    return {.h = handle};
}

void Device::Impl::free(Handle<TextureView> view) {
    m_api.vkDestroyImageView(m_device, m_texture_view_pool[view.h].vk_image_view, nullptr);
    m_texture_view_pool.release(static_cast<uint32_t>(view.h));
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
    Arena                                     arena = *get_thread_local_arena();
    Span<VkPipelineColorBlendAttachmentState> color_blend_attachment_states{};
    Span<VkFormat>                            color_attachment_formats{};

    for (auto& t : desc.colorTargets) {
        // const auto attachment_state = loon::gpu::bridge(target);
        color_blend_attachment_states
            = concat(&arena, color_blend_attachment_states, loon::gpu::bridge(desc.blendstate));
        color_attachment_formats
            = concat(&arena, color_attachment_formats, loon::gpu::bridge(t.format));
    }

    VkPipelineColorBlendStateCreateInfo color_blend_state{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0,
        .logicOpEnable   = false,
        .logicOp         = VK_LOGIC_OP_NO_OP,
        .attachmentCount = static_cast<uint32_t>(color_blend_attachment_states.size()),
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
        .colorAttachmentCount    = static_cast<uint32_t>(color_attachment_formats.size()),
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
    // Initialize the queue on-demand.
    if (m_queues[type].queue == VK_NULL_HANDLE) {
        uint32_t queue_family = 0;
        switch (type) {
            case QUEUE_DEFAULT: queue_family = m_graphics_queue_family; break;
            case QUEUE_COMPUTE: queue_family = m_async_compute_queue_family; break;
            case QUEUE_TRANSFER: queue_family = m_transfer_queue_family; break;
            case QUEUE_VALID_COUNT: break;
        }

        VkQueue queue;
        m_api.vkGetDeviceQueue(m_device, queue_family, 0, &queue);

        VkCommandPoolCreateInfo pool_info{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext            = nullptr,
            .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = queue_family,
        };
        VkCommandPool command_pool;
        chk(m_api.vkCreateCommandPool(m_device, &pool_info, nullptr, &command_pool));

        VkSemaphore timeline = reinterpret_cast<VkSemaphore>(createSemaphore(0).h);

        m_queues[type] = {
            .queue          = queue,
            .command_pool   = command_pool,
            .timeline       = timeline,
            .queue_family   = queue_family,
            .timeline_value = 0,
            .pending_events = Vector<Queue::Event>(m_allocator),
        };
    }

    return {.h = (uint64_t)type};
}

CommandBuffer Device::Impl::startCommandRecording(Handle<Queue> queue) {
    const VkCommandBufferAllocateInfo alloc_info{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = m_queues[queue.h].command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    chk(m_api.vkAllocateCommandBuffers(m_device, &alloc_info, &cmd));

    const VkCommandBufferBeginInfo begin_info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    m_api.vkBeginCommandBuffer(cmd, &begin_info);

    return CommandBuffer(reinterpret_cast<uint64_t>(cmd), reinterpret_cast<uint64_t>(this));
}

void Device::Impl::submit(Handle<Queue>             queue,
                          Span<const CommandBuffer> command_buffers,
                          Span<const SemaphoreInfo> wait_semaphores,
                          Span<const SemaphoreInfo> signal_semaphores)

{
    auto& q = m_queues[queue.h];

    auto arena = *get_thread_local_arena();

    Span<VkSemaphoreSubmitInfo>     wait_info;
    Span<VkCommandBufferSubmitInfo> command_info;
    Span<VkSemaphoreSubmitInfo>     signal_info;

    for (uint32_t i = 0; i < wait_semaphores.size(); ++i) {
        wait_info
            = concat(&arena,
                     wait_info,
                     VkSemaphoreSubmitInfo{
                         .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                         .pNext     = nullptr,
                         .semaphore = reinterpret_cast<VkSemaphore>(wait_semaphores[i].semaphore.h),
                         .value     = wait_semaphores[i].value,
                         .stageMask = bridge_pipeline_stage(wait_semaphores[i].stage),
                         .deviceIndex = 0,
                     });
    }

    for (uint32_t i = 0; i < command_buffers.size(); i++) {
        auto buf = reinterpret_cast<VkCommandBuffer>(command_buffers[i].buffer);

        chk(m_api.vkEndCommandBuffer(buf));

        command_info = concat(&arena,
                              command_info,
                              VkCommandBufferSubmitInfo{
                                  .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                  .pNext         = nullptr,
                                  .commandBuffer = buf,
                                  .deviceMask    = 1,
                              });

        if (buf == m_surface.transitioning_command[m_surface.current_semaphore_idx]) {
            signal_info = concat(
                &arena,
                signal_info,
                VkSemaphoreSubmitInfo{
                    .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .pNext       = nullptr,
                    .semaphore   = m_surface.present_semaphores[m_surface.current_semaphore_idx],
                    .value       = 0,
                    .stageMask   = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    .deviceIndex = 0,
                });
        }
    }

    for (uint32_t i = 0; i < signal_semaphores.size(); ++i) {
        signal_info = concat(
            &arena,
            signal_info,
            VkSemaphoreSubmitInfo{
                .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext       = nullptr,
                .semaphore   = reinterpret_cast<VkSemaphore>(signal_semaphores[i].semaphore.h),
                .value       = signal_semaphores[i].value,
                .stageMask   = bridge_pipeline_stage(signal_semaphores[i].stage),
                .deviceIndex = 0,
            });
    }

    // We add one extra signal to advance the queue timeline
    signal_info = concat(&arena,
                         signal_info,
                         VkSemaphoreSubmitInfo{
                             .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                             .pNext       = nullptr,
                             .semaphore   = q.timeline,
                             .value       = ++q.timeline_value,
                             .stageMask   = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                             .deviceIndex = 0,
                         });

    VkSubmitInfo2 submit_info{
        .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext                    = nullptr,
        .flags                    = 0,
        .waitSemaphoreInfoCount   = static_cast<uint32_t>(wait_info.size()),
        .pWaitSemaphoreInfos      = wait_info.data(),
        .commandBufferInfoCount   = static_cast<uint32_t>(command_info.size()),
        .pCommandBufferInfos      = command_info.data(),
        .signalSemaphoreInfoCount = static_cast<uint32_t>(signal_info.size()),
        .pSignalSemaphoreInfos    = signal_info.data(),
    };

    m_api.vkQueueSubmit2(q.queue, 1, &submit_info, VK_NULL_HANDLE);

    // Need to free/reset the command buffers once this submission is done - might be worth copying
    // the whole span of command buffers to a temporary allocation instead of pushing a bunch of
    // completions here instead.
    for (CommandBuffer cmd : command_buffers) {
        q.pending_events.emplace_back(
            Queue::Event{.completed_time = q.timeline_value,
                         .callback       = [command_pool = q.command_pool, cmd]() {
                             auto impl = reinterpret_cast<Impl*>(cmd.device);
                             impl->m_api.vkFreeCommandBuffers(
                                 impl->m_device,
                                 command_pool,
                                 1,
                                 reinterpret_cast<const VkCommandBuffer*>(&cmd.buffer));
                         }});
    }
}

void Device::Impl::on_submitted_work_completed(Handle<Queue> queue, Function<void>&& fn) {
    auto& q = m_queues[queue.h];
    q.pending_events.emplace_back(
        Queue::Event{.completed_time = q.timeline_value, .callback = std::move(fn)});
}

void Device::Impl::process_events(Handle<Queue> queue) {
    auto&    q            = m_queues[queue.h];
    uint64_t current_time = 0;
    chk(m_api.vkGetSemaphoreCounterValue(m_device, q.timeline, &current_time));
    uint32_t i = 0;
    while (i < q.pending_events.size() && q.pending_events[i].completed_time <= current_time) {
        q.pending_events[i].callback();
        i++;
    }
    if (i != 0) { q.pending_events.erase(q.pending_events.begin(), q.pending_events.begin() + i); }
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

SurfaceCapabilities Device::get_surface_capabilities() {
    return impl->get_surface_capabilities();
}

bool Device::configure_surface(const SurfaceConfiguration& config) {
    return impl->configure_surface(config);
}

void Device::unconfigure_surface() {
    return impl->unconfigure_surface();
}

SurfaceTextureInfo Device::get_current_texture() {
    return impl->get_current_texture();
}

void Device::present(Handle<Queue> queue) {
    impl->present(queue);
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

Handle<Texture> Device::createTexture(const GpuTextureDesc& desc) {
    return {};
}
Handle<TextureHeap> Device::createTextureHeap(size_t size) {
    return {};
}

Handle<TextureView> Device::createTextureView(Handle<Texture> texture, TextureViewDesc desc) {
    return impl->createTextureView(texture, desc);
}

void Device::free(Handle<Texture>) {}
void Device::free(Handle<TextureHeap>) {}
void Device::free(Handle<TextureView> view) {
    return impl->free(view);
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

CommandBuffer Device::startCommandRecording(Handle<Queue> queue) {
    return impl->startCommandRecording(queue);
}

void Device::submit(Handle<Queue>             queue,
                    Span<const CommandBuffer> commandBuffers,
                    Span<const SemaphoreInfo> wait_semaphores,
                    Span<const SemaphoreInfo> signal_semaphores) {
    impl->submit(queue, commandBuffers, wait_semaphores, signal_semaphores);
}

void Device::cancel(Handle<Queue> queue, Span<const Handle<CommandBuffer>> commandBuffers) {}

void Device::on_submitted_work_completed(Handle<Queue> queue, Function<void>&& fn) {
    impl->on_submitted_work_completed(queue, std::move(fn));
}
void Device::process_events(Handle<Queue> queue) {
    impl->process_events(queue);
}

Handle<Semaphore> Device::createSemaphore(uint64_t initValue) {
    return impl->createSemaphore(initValue);
}

void Device::waitSemaphore(Handle<Semaphore> sema, uint64_t value) {
    impl->waitSemaphore(sema, value);
}

void Device::destroySemaphore(Handle<Semaphore> sema) {
    impl->destroySemaphore(sema);
}


// MARK: Commmand Buffer

void CommandBuffer::setPipeline(Handle<Pipeline> pipeline) {
    auto impl = reinterpret_cast<Device::Impl*>(device);
    impl->m_api.vkCmdBindPipeline(
        reinterpret_cast<VkCommandBuffer>(buffer),
        VK_PIPELINE_BIND_POINT_GRAPHICS,  // TODO: Fix this, can't be hardcoded.
        reinterpret_cast<VkPipeline>(pipeline.h));
}

void CommandBuffer::beginRenderPass(RenderPassDesc desc) {
    auto impl = reinterpret_cast<Device::Impl*>(device);

    Arena                           arena = *impl->get_thread_local_arena();
    Span<VkRenderingAttachmentInfo> color_attachments;

    for (const auto& attachment : desc.color_attachments) {
        color_attachments = concat(&arena, color_attachments, {
            .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext              = nullptr,
            .imageView          = impl->m_texture_view_pool[attachment.texture_view.h].vk_image_view,
            .imageLayout        = VK_IMAGE_LAYOUT_GENERAL,
            .resolveMode        = VK_RESOLVE_MODE_NONE,  // TODO: Multisampling support
            .resolveImageView   = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .loadOp             = bridge(attachment.load_op),
            .storeOp            = bridge(attachment.store_op),
            .clearValue         = {.color = {.uint32 = {attachment.clear_color.r,attachment.clear_color.g,attachment.clear_color.b,attachment.clear_color.a},},},
        });
    }

    // TODO: Depth/stencil values
    const VkRect2D render_rect = {
            .offset = {.x = (int32_t)desc.render_area.offset_x, .y = (int32_t)desc.render_area.offset_y,},
            .extent = {.width = desc.render_area.width, .height = desc.render_area.height,},
        };
    const VkRenderingInfo rendering_info{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext                = nullptr,
        .flags                = 0,
        .renderArea           = render_rect,
        .layerCount           = 1,
        .viewMask             = 0,
        .colorAttachmentCount = static_cast<uint32_t>(color_attachments.size()),
        .pColorAttachments    = color_attachments.data(),
        .pDepthAttachment     = nullptr,
        .pStencilAttachment   = nullptr,
    };
    impl->m_api.vkCmdBeginRendering(reinterpret_cast<VkCommandBuffer>(buffer), &rendering_info);

    // Set default values for dynamic state:
    impl->m_api.vkCmdSetDepthTestEnable(reinterpret_cast<VkCommandBuffer>(buffer), false);
    impl->m_api.vkCmdSetStencilTestEnable(reinterpret_cast<VkCommandBuffer>(buffer), false);
    VkViewport viewport{
        .x        = 0,
        .y        = 0,
        .width    = (float)desc.render_area.width,
        .height   = (float)desc.render_area.height,
        .minDepth = 0,
        .maxDepth = 1.0,
    };
    impl->m_api.vkCmdSetViewportWithCount(reinterpret_cast<VkCommandBuffer>(buffer), 1, &viewport);
    impl->m_api.vkCmdSetScissorWithCount(reinterpret_cast<VkCommandBuffer>(buffer),
                                         1,
                                         &render_rect);
}

void CommandBuffer::endRenderPass() {
    auto impl = reinterpret_cast<Device::Impl*>(device);
    impl->m_api.vkCmdEndRendering(reinterpret_cast<VkCommandBuffer>(buffer));
}

void CommandBuffer::set_graphics_ptrs(GpuPtr vertexDataGpu, GpuPtr fragmentDataGpu) {
    auto impl = reinterpret_cast<Device::Impl*>(device);
    if (vertexDataGpu != 0 || fragmentDataGpu != 0) {
        VkDeviceAddress           addresses[2] = {vertexDataGpu, fragmentDataGpu};
        const VkPushConstantsInfo info{
            .sType      = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
            .pNext      = nullptr,
            .layout     = impl->m_default_graphics_layout,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset     = 0,
            .size       = sizeof(addresses),
            .pValues    = addresses,
        };
        impl->m_api.vkCmdPushConstants2(reinterpret_cast<VkCommandBuffer>(buffer), &info);
    }
}

void CommandBuffer::draw(GpuPtr   vertexDataGpu,
                         GpuPtr   fragmentDataGpu,
                         uint32_t vertexCount,
                         uint32_t instanceCount,
                         uint32_t firstVertex,
                         uint32_t firstInstance) {
    auto impl = reinterpret_cast<Device::Impl*>(device);
    set_graphics_ptrs(vertexDataGpu, fragmentDataGpu);

    impl->m_api.vkCmdDraw(reinterpret_cast<VkCommandBuffer>(buffer),
                          vertexCount,
                          instanceCount,
                          firstVertex,
                          firstInstance);
}

void CommandBuffer::drawIndexedInstanced(GpuPtr   vertexDataGpu,
                                         GpuPtr   fragmentDataGpu,
                                         GpuPtr   indicesGpu,
                                         uint32_t indexCount,
                                         uint32_t instanceCount) {
    auto impl = reinterpret_cast<Device::Impl*>(device);
    set_graphics_ptrs(vertexDataGpu, fragmentDataGpu);
    const auto indices = impl->buffer_and_offset_from_ptr(indicesGpu);
    impl->m_api.vkCmdBindIndexBuffer(reinterpret_cast<VkCommandBuffer>(buffer),
                                     indices.buffer,
                                     indices.offset,
                                     VK_INDEX_TYPE_UINT32);
    impl->m_api.vkCmdDrawIndexed(reinterpret_cast<VkCommandBuffer>(buffer),
                                 indexCount,
                                 instanceCount,
                                 0,
                                 0,
                                 0);
}

void CommandBuffer::make_surface_writable() {
    auto impl = reinterpret_cast<Device::Impl*>(device);

    auto image           = impl->m_surface.swapchain_images[impl->m_surface.current_image_idx];
    auto swapchain_image = impl->m_texture_pool[image.h].vk_image;

    const VkImageMemoryBarrier2 barrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = 0,
        .dstQueueFamilyIndex = 0,
        .image = swapchain_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    const VkDependencyInfo info{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                    = nullptr,
        .dependencyFlags          = 0,
        .memoryBarrierCount       = 0,
        .pMemoryBarriers          = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers    = nullptr,
        .imageMemoryBarrierCount  = 1,
        .pImageMemoryBarriers     = &barrier,
    };
    impl->m_api.vkCmdPipelineBarrier2(reinterpret_cast<VkCommandBuffer>(buffer), &info);
}

void CommandBuffer::make_surface_presentable() {
    auto impl = reinterpret_cast<Device::Impl*>(device);

    auto image           = impl->m_surface.swapchain_images[impl->m_surface.current_image_idx];
    auto swapchain_image = impl->m_texture_pool[image.h].vk_image;

    const VkImageMemoryBarrier2 barrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .srcQueueFamilyIndex = 0,
        .dstQueueFamilyIndex = 0,
        .image = swapchain_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    const VkDependencyInfo info{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                    = nullptr,
        .dependencyFlags          = 0,
        .memoryBarrierCount       = 0,
        .pMemoryBarriers          = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers    = nullptr,
        .imageMemoryBarrierCount  = 1,
        .pImageMemoryBarriers     = &barrier,
    };
    impl->m_api.vkCmdPipelineBarrier2(reinterpret_cast<VkCommandBuffer>(buffer), &info);
    impl->m_surface.transitioning_command[impl->m_surface.current_semaphore_idx]
        = reinterpret_cast<VkCommandBuffer>(buffer);
}

}  // namespace loon::gpu
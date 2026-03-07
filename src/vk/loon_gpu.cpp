#include "gpu/loon_gpu.h"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "containers.h"
#include "gpu_to_vk.h"
#include "platform_utils.h"
#include "vma_usage.h"
#include "volk.h"
#include "vulkan/vulkan_core.h"

namespace loon::gpu {
template class Span<const char>;
template class Span<uint8_t>;
template class Span<const gpu::SamplerDesc>;
template class Span<const gpu::ColorTarget>;
template class Span<const gpu::RenderAttachment>;
template class Span<const gpu::Format>;
template class Span<const gpu::PresentMode>;
template class Span<const gpu::CommandBuffer>;
template class Span<const gpu::SemaphoreInfo>;
template class Span<const Handle<gpu::CommandBuffer>>;
template class Span<const gpu::TextureTransition>;

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
    void*         host_ptr;
    GpuPtr        device_ptr;
};

struct Texture {
    VkImage     vk_image;
    VkImageView default_image_view
        = VK_NULL_HANDLE;  // We store a default image view of the base mip and layer if
                           // the image can be used as a render target.
    VmaAllocation   vk_allocation;
    VkImageViewType vk_type = VK_IMAGE_VIEW_TYPE_2D;
    Format          format;
    bool            is_swapchain_image = false;
};

struct TextureHeap {
    VkDescriptorSet     vk_descriptor_set;
    TwoLevelBitset      bitset;
    Vector<VkImageView> image_views;
};

struct Semaphore {
    VkSemaphore vk_semaphore;
};

struct Pipeline {
    VkPipeline          vk_pipeline;
    VkPipelineBindPoint bind_point;
};

struct VulkanInstanceInfo {
    VkResult   result;
    VkInstance instance;
};

struct PhysicalDeviceInfo {
    VkResult         result                     = VK_SUCCESS;
    VkPhysicalDevice device                     = VK_NULL_HANDLE;
    uint32_t         graphics_queue_family      = 0;
    uint32_t         transfer_queue_family      = 0;
    uint32_t         async_compute_queue_family = 0;
};

struct SurfaceCreationResult {
    VkResult     result;
    VkSurfaceKHR surface;
};

struct MemoryRequirements {
    VkMemoryRequirements gpu_mem_requirements;
    VkMemoryRequirements buffer_mem_requirements;
    VkResult             result;
};

struct VmaCreateResult {
    VkResult     result;
    VmaAllocator allocator;
};

struct LogicalDeviceCreateResult {
    VkResult result;
    VkDevice logical_device;
};


struct DepthStencilState : DepthStencilDesc {};

static void              log(Device d, LogLevel lvl, Span<const char> msg);
static bool              chk(Device d, VkResult result);
static loon::gpu::Arena* get_thread_local_arena(Device d);

struct Surface {
    static constexpr uint32_t kMaxSwapchainImages = 8;
    static constexpr uint32_t kMaxFramesInFlight  = 3;

    VkSurfaceKHR   surface   = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;

    // No swapchain will support all the formats, but makes memory allocation simpler.
    static constexpr size_t kMaxNumFormats      = static_cast<size_t>(Format::ValidCount);
    static constexpr size_t kMaxNumPresentModes = static_cast<size_t>(PresentMode::ValidCount);
    Format                  supported_formats[kMaxNumFormats];
    PresentMode             supported_present_modes[kMaxNumPresentModes];
    size_t                  num_supported_formats       = 0;
    size_t                  num_supported_present_modes = 0;

    VkFormat   swapchain_format = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchain_extent = {0, 0};

    uint32_t        current_image_idx = 0;
    uint32_t        image_count       = 0;
    Handle<Texture> swapchain_images[kMaxSwapchainImages];

    // Semaphores are ticked around the ring buffer independently from images, since we need to know
    // which semaphore to use before we know which image index we have.
    uint64_t          frame_idx = 0;
    Handle<Semaphore> frame_semaphore;
    Handle<Semaphore> acquire_semaphores[kMaxFramesInFlight];
    Handle<Semaphore> present_semaphores[kMaxSwapchainImages];

    // We store the command buffer that first uses the surface image (tracked by barriers doing an
    // image transition which is necessary before use), so we can wait on an extra semaphore on
    // submission.
    VkCommandBuffer first_use_command[kMaxFramesInFlight] = {VK_NULL_HANDLE};
    // We store the command buffer that recording
    // the transition to PRESENT_KHR, so we can
    // signal an extra semaphore on submission.
    VkCommandBuffer transitioning_command[kMaxSwapchainImages] = {VK_NULL_HANDLE};
};

struct CommandPool;
struct CommandBufferImpl {
    Device          device;
    Queue           queue;
    CommandPool*    pool = nullptr;
    VkCommandBuffer buffer;
};

struct CommandPool {
    VkCommandPool             command_pool = VK_NULL_HANDLE;
    Vector<CommandBufferImpl> command_buffers;
    uint64_t                  buffer_free_idx = 0;  // Index of the next command_buffer to use.
    uint64_t                  frame_idx = 0;  // Frame index of the last time this pool was used.
};

struct CommandSuperpool {
    static constexpr uint32_t kPoolsPerGroup           = Surface::kMaxFramesInFlight;
    static constexpr uint32_t kMaxSimultaneousCommands = 64;
    int64_t                   available_pools          = ~0;
    CommandPool               pools[kMaxSimultaneousCommands * kPoolsPerGroup] = {};
};

struct QueueImpl {
    struct Event {
        uint64_t       completed_time;
        Function<void> callback;
    };

    Device            device            = nullptr;
    VkQueue           queue             = VK_NULL_HANDLE;
    CommandSuperpool  command_superpool = {};
    Handle<Semaphore> timeline          = {};
    uint32_t          queue_family      = 0;
    uint64_t          timeline_value    = 0;
    Vector<Event>     pending_events;
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
    VkBuffer      buffer;
    uint32_t      offset;
    VmaAllocation alloc;
};

struct GpuPtrMap {
    GpuPtr         ptr;
    Handle<Buffer> buffer;
};
static constexpr auto kPtrMapCompare
    = [](const GpuPtrMap& a, const GpuPtrMap& b) -> bool { return a.ptr > b.ptr; };

static constexpr auto lower_bound = [](GpuPtrMap* first, GpuPtrMap* last, const GpuPtrMap& value) {
    GpuPtrMap* it;
    size_t     count = last - first;
    while (count > 0) {
        const size_t step = count / 2;
        it                = first + step;
        if (kPtrMapCompare(*it, value)) {
            first = ++it;
            count -= step + 1;
        } else {
            count = step;
        }
    }

    return first;
};

struct DeviceImpl {
    Allocator          m_allocator;
    ProcLogCallback    m_log_callback = nullptr;
    void*              m_log_userdata = nullptr;
    LogLevel           m_log_level    = LogLevel::Off;
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

    VkDescriptorPool      m_descriptor_pool;
    VkDescriptorSetLayout m_default_descriptor_layout;
    VkPipelineLayout      m_default_graphics_layout;
    VkPipelineLayout      m_default_compute_layout;
    VkMemoryRequirements  m_gpu_mem_requirements;
    VkMemoryRequirements  m_buffer_mem_requirements;

    SlotMap<Buffer>            m_buffer_pool;
    SlotMap<Texture>           m_texture_pool;
    SlotMap<TextureHeap>       m_texture_heap_pool;
    SlotMap<DepthStencilState> m_depth_stencil_pool;
    SlotMap<Semaphore>         m_semaphore_pool;
    SlotMap<Pipeline>          m_pipeline_pool;

    Vector<VkSampler> m_immutable_samplers;
    Vector<GpuPtrMap> m_ptr_map;
    QueueImpl         m_queues[static_cast<size_t>(QueueType::ValidCount)];
};

// MARK: Initialization

static VulkanInstanceInfo create_instance() {
    VkResult result = volkInitialize();
    if (result != VK_SUCCESS) { return {.result = result}; };

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

    VkInstance instance = VK_NULL_HANDLE;
    result              = vkCreateInstance(&instance_info, nullptr, &instance);
    if (result != VK_SUCCESS) { return {.result = result}; };

    volkLoadInstanceOnly(instance);

    return {
        .result   = result,
        .instance = instance,
    };
}

static PhysicalDeviceInfo select_physical_device(VkInstance    instance,
                                                 VkSurfaceKHR  surface,
                                                 GpuPreference preference,
                                                 Arena         arena) {
    uint32_t device_count = 0;
    VkResult vkresult     = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (vkresult != VK_SUCCESS) {
        return {
            .result = vkresult,
        };
    }

    auto physical_devices = (VkPhysicalDevice*)arena.alloc(sizeof(VkPhysicalDevice) * device_count);
    if (physical_devices == nullptr) {
        return {
            .result = VK_ERROR_OUT_OF_HOST_MEMORY,
        };
    }

    vkresult = vkEnumeratePhysicalDevices(instance, &device_count, physical_devices);
    if (vkresult != VK_SUCCESS) { device_count = 0; }

    const bool prefer_integrated = preference == GpuPreference::Integrated;
    const bool prefer_dedicated  = preference == GpuPreference::Discrete;

    PhysicalDeviceInfo best_device_info = {
        .result = vkresult,
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

static SurfaceCreationResult create_surface(VkInstance instance, const DeviceDesc& desc) {
    // Create surface if requested:
    VkResult     result  = VK_SUCCESS;
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
        result = vkCreateWin32SurfaceKHR(m_instance, &surface_info, nullptr, &surface);
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
        const VkXlibSurfaceCreateInfoKHR surface_info{
            .sType  = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
            .pNext  = nullptr,
            .flags  = 0,
            .dpy    = reinterpret_cast<Display*>(desc.native_instance_handle),
            .window = desc.native_window_handle,
        };
        result = vkCreateXlibSurfaceKHR(m_instance, &surface_info, nullptr, &surface);
#elif defined(VK_USE_PLATFORM_METAL_EXT)
        const VkMetalSurfaceCreateInfoEXT surface_info{
            .sType  = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
            .pNext  = nullptr,
            .flags  = 0,
            .pLayer = reinterpret_cast<CAMetalLayer*>(desc.native_window_handle),
        };
        result = vkCreateMetalSurfaceEXT(instance, &surface_info, nullptr, &surface);
#else
#    error "Unsupported platform"
#endif
    }

    return {
        .result  = result,
        .surface = surface,
    };
}

static VkDescriptorSetLayout create_descriptor_layout(VkDevice               device,
                                                      const VolkDeviceTable& api,
                                                      uint32_t               size,
                                                      Span<const VkSampler>  samplers) {
    VkDescriptorBindingFlags descVariableFlag[] = {
        0,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
            | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
            | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT,
    };

    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount  = 2,
        .pBindingFlags = descVariableFlag};

    VkDescriptorSetLayoutBinding bindings[] = {
        {
            .binding            = 0,
            .descriptorType     = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount    = static_cast<uint32_t>(samplers.size()),
            .stageFlags         = VK_SHADER_STAGE_ALL,
            .pImmutableSamplers = samplers.data(),
        },
        {
            .binding         = 2,
            .descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = size,
            .stageFlags      = VK_SHADER_STAGE_ALL,
        },
    };

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext        = &binding_flags,
        .flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = 2,
        .pBindings    = bindings,
    };

    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkResult              result                = api.vkCreateDescriptorSetLayout(device,
                                                      &descriptor_set_layout_create_info,
                                                      nullptr,
                                                      &descriptor_set_layout);
    return result == VK_SUCCESS ? descriptor_set_layout : VK_NULL_HANDLE;
}

static VkPipelineLayout create_default_graphics_layout(VkDevice               device,
                                                       const VolkDeviceTable& api,
                                                       VkDescriptorSetLayout  set_layout) {
    // We create 2 push constants - for vertex and fragment data.
    VkPushConstantRange push_constant_ranges = VkPushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset     = 0,
        .size       = 2 * sizeof(VkDeviceAddress),
    };

    VkPipelineLayoutCreateInfo create_info{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0,
        .setLayoutCount         = 1,
        .pSetLayouts            = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_constant_ranges,
    };

    VkPipelineLayout pipeline_layout;
    VkResult result = api.vkCreatePipelineLayout(device, &create_info, nullptr, &pipeline_layout);
    return result == VK_SUCCESS ? pipeline_layout : VK_NULL_HANDLE;
}

static VkPipelineLayout create_default_compute_layout(VkDevice               device,
                                                      const VolkDeviceTable& api,
                                                      VkDescriptorSetLayout  set_layout) {
    // We create 1 push constant for compute data
    VkPushConstantRange push_constant_range = {
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset     = 0,
            .size       = sizeof(VkDeviceAddress),
        },
    };

    VkPipelineLayoutCreateInfo create_info{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0,
        .setLayoutCount         = 1,
        .pSetLayouts            = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_constant_range,
    };

    VkPipelineLayout pipeline_layout;
    VkResult result = api.vkCreatePipelineLayout(device, &create_info, nullptr, &pipeline_layout);
    return result == VK_SUCCESS ? pipeline_layout : VK_NULL_HANDLE;
}

static Vector<VkSampler> create_immutable_samplers(Allocator               alloc,
                                                   const VolkDeviceTable&  api,
                                                   VkDevice                device,
                                                   Span<const SamplerDesc> sampler_descs) {
    auto samplers = Vector<VkSampler>(alloc);

    constexpr auto bridge_filter = [](SamplerFilter f) {
        switch (f) {
            case SamplerFilter::Nearest: return VK_FILTER_NEAREST;
            case SamplerFilter::Linear: return VK_FILTER_LINEAR;
        }
        return VK_FILTER_MAX_ENUM;
    };

    constexpr auto bridge_mip_mode = [](SamplerFilter f) {
        switch (f) {
            case SamplerFilter::Nearest: return VK_SAMPLER_MIPMAP_MODE_NEAREST;
            case SamplerFilter::Linear: return VK_SAMPLER_MIPMAP_MODE_LINEAR;
        }
        return VK_SAMPLER_MIPMAP_MODE_MAX_ENUM;
    };

    constexpr auto bridge_address = [](SamplerAddressing a) {
        switch (a) {
            case SamplerAddressing::ClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            case SamplerAddressing::Repeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case SamplerAddressing::Mirrored: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        }
        return VK_SAMPLER_ADDRESS_MODE_MAX_ENUM;
    };

    for (const auto& sampler_desc : sampler_descs) {
        const VkSamplerCreateInfo sampler_info{
            .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext                   = nullptr,
            .flags                   = 0,
            .magFilter               = bridge_filter(sampler_desc.filter),
            .minFilter               = bridge_filter(sampler_desc.filter),
            .mipmapMode              = bridge_mip_mode(sampler_desc.filter),
            .addressModeU            = bridge_address(sampler_desc.address),
            .addressModeV            = bridge_address(sampler_desc.address),
            .addressModeW            = bridge_address(sampler_desc.address),
            .anisotropyEnable        = (sampler_desc.max_anisotropy != 0.0f),
            .maxAnisotropy           = sampler_desc.max_anisotropy,
            .compareEnable           = VK_FALSE,
            .minLod                  = -1000.0f,
            .maxLod                  = VK_LOD_CLAMP_NONE,
            .borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
            .unnormalizedCoordinates = sampler_desc.coord == SamplerCoords::Pixel,
        };

        VkSampler sampler;
        if (api.vkCreateSampler(device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
            return {};
        }

        samplers.push_back(sampler);
    }
    return samplers;
}

static MemoryRequirements get_memory_requirements(const VolkDeviceTable& api, VkDevice device) {
    VkBuffer                     test_device_buffer;
    constexpr VkBufferUsageFlags kDefaultUsages
        = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
          | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
          | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
          | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

    VkBufferCreateInfo test_buffer_info{
        .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .size                  = 1024ull * 1024,
        .usage                 = kDefaultUsages,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,  // TODO: Support multiple queues.
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
    };
    VkResult result = api.vkCreateBuffer(device, &test_buffer_info, nullptr, &test_device_buffer);
    if (result != VK_SUCCESS) { return {.result = result}; }

    VkMemoryRequirements buffer_requirements;
    api.vkGetBufferMemoryRequirements(device, test_device_buffer, &buffer_requirements);

    api.vkDestroyBuffer(device, test_device_buffer, nullptr);

    VkImage                 test_color_image;
    const VkImageCreateInfo test_color_info{
        .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .imageType             = VK_IMAGE_TYPE_2D,
        .format                = VK_FORMAT_R8G8B8A8_SRGB,
        .extent                = {.width = 4096, .height = 4096, .depth = 1},
        .mipLevels             = 1,
        .arrayLayers           = 1,
        .samples               = VK_SAMPLE_COUNT_1_BIT,  // TODO: Support multisampling
        .tiling                = VK_IMAGE_TILING_OPTIMAL,
        .usage                 = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    result = api.vkCreateImage(device, &test_color_info, nullptr, &test_color_image);
    if (result != VK_SUCCESS) { return {.result = result}; }

    VkMemoryRequirements color_image_requirements;
    api.vkGetImageMemoryRequirements(device, test_color_image, &color_image_requirements);
    api.vkDestroyImage(device, test_color_image, nullptr);

    VkImage                 test_depth_image;
    const VkImageCreateInfo test_depth_info{
        .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .imageType             = VK_IMAGE_TYPE_2D,
        .format                = VK_FORMAT_D16_UNORM,
        .extent                = {.width = 4096, .height = 4096, .depth = 1},
        .mipLevels             = 1,
        .arrayLayers           = 1,
        .samples               = VK_SAMPLE_COUNT_1_BIT,  // TODO: Support multisampling
        .tiling                = VK_IMAGE_TILING_OPTIMAL,
        .usage                 = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    result = api.vkCreateImage(device, &test_depth_info, nullptr, &test_depth_image);
    if (result != VK_SUCCESS) { return {.result = result}; }

    VkMemoryRequirements depth_image_requirements;
    api.vkGetImageMemoryRequirements(device, test_depth_image, &depth_image_requirements);
    api.vkDestroyImage(device, test_depth_image, nullptr);

    size_t alignment = (color_image_requirements.alignment > depth_image_requirements.alignment)
                           ? (color_image_requirements.alignment > buffer_requirements.alignment
                                  ? color_image_requirements.alignment
                                  : buffer_requirements.alignment)
                           : (depth_image_requirements.alignment > buffer_requirements.alignment
                                  ? depth_image_requirements.alignment
                                  : buffer_requirements.alignment);

    // For GPU-local memory, our requirements need to meet all of the memory type bits for color
    // and depth images. For Default and Readback, we only need to meet the requirements of
    // buffer allocation, since we don't support making textures on those memory types.

    return MemoryRequirements{.buffer_mem_requirements = buffer_requirements,
                              .gpu_mem_requirements    = VkMemoryRequirements{
                                     .size           = 0,
                                     .alignment      = alignment,
                                     .memoryTypeBits = buffer_requirements.memoryTypeBits
                                                    & color_image_requirements.memoryTypeBits
                                                    & depth_image_requirements.memoryTypeBits,
                              },
                            .result = VK_SUCCESS,};
}

static VmaCreateResult create_vma_allocator(VkInstance       instance,
                                            VkPhysicalDevice physical_device,
                                            VkDevice         device) {
    VmaAllocatorCreateInfo vma_create_info{
        .flags                          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice                 = physical_device,
        .device                         = device,
        .preferredLargeHeapBlockSize    = 0,
        .pAllocationCallbacks           = nullptr,
        .pDeviceMemoryCallbacks         = nullptr,
        .pHeapSizeLimit                 = nullptr,
        .pVulkanFunctions               = nullptr,
        .instance                       = instance,
        .vulkanApiVersion               = VK_API_VERSION_1_3,
        .pTypeExternalMemoryHandleTypes = nullptr,
    };
    VmaVulkanFunctions vulkan_functions;
    vmaImportVulkanFunctionsFromVolk(&vma_create_info, &vulkan_functions);
    vma_create_info.pVulkanFunctions = &vulkan_functions;
    VmaAllocator allocator;
    VkResult     result = vmaCreateAllocator(&vma_create_info, &allocator);

    return {
        .result    = result,
        .allocator = allocator,
    };
}

static LogicalDeviceCreateResult create_logical_device(
    const DeviceDesc&         desc,
    Arena                     arena,
    const PhysicalDeviceInfo& physical_device_info) {
    float                         queue_priority = 1.0f;
    Span<VkDeviceQueueCreateInfo> queue_create_infos;
    queue_create_infos = concat(&arena,
                                queue_create_infos,
                                VkDeviceQueueCreateInfo{
                                    .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                    .pNext            = nullptr,
                                    .flags            = 0,
                                    .queueFamilyIndex = physical_device_info.graphics_queue_family,
                                    .queueCount       = 1,
                                    .pQueuePriorities = &queue_priority,
                                });

    if (physical_device_info.async_compute_queue_family != -1
        && physical_device_info.async_compute_queue_family
               != physical_device_info.graphics_queue_family) {
        queue_create_infos
            = concat(&arena,
                     queue_create_infos,
                     {
                         .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                         .pNext            = nullptr,
                         .flags            = 0,
                         .queueFamilyIndex = physical_device_info.async_compute_queue_family,
                         .queueCount       = 1,
                         .pQueuePriorities = &queue_priority,
                     });
    }

    if (physical_device_info.transfer_queue_family != -1
        && physical_device_info.transfer_queue_family
               != physical_device_info.async_compute_queue_family
        && physical_device_info.transfer_queue_family
               != physical_device_info.graphics_queue_family) {
        queue_create_infos
            = concat(&arena,
                     queue_create_infos,
                     {
                         .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                         .pNext            = nullptr,
                         .flags            = 0,
                         .queueFamilyIndex = physical_device_info.transfer_queue_family,
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
    vulkan_12_features.sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan_12_features.pNext                = &vulkan_13_features;
    vulkan_12_features.timelineSemaphore    = true;
    vulkan_12_features.bufferDeviceAddress  = true;
    vulkan_12_features.descriptorIndexing   = true;
    vulkan_12_features.shaderInt8           = true;
    vulkan_12_features.storagePushConstant8 = true;
    vulkan_12_features.scalarBlockLayout    = true;
    vulkan_12_features.runtimeDescriptorArray                       = true;
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
            .features = {
                .samplerAnisotropy = true,
            },
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

    VkDevice device = VK_NULL_HANDLE;
    VkResult result = vkCreateDevice(physical_device_info.device, &create_info, nullptr, &device);

    return {
        .result         = result,
        .logical_device = device,
    };
}

Device create_device(const DeviceDesc& desc) {
    Allocator alloc
        = desc.alloc_callback ? Allocator(desc.alloc_callback, desc.alloc_userdata) : Allocator();

    auto blk = alloc.alloc(sizeof(DeviceImpl));
    if (blk.ptr == 0) { return nullptr; }


    auto  arena_blk = alloc.alloc(256 * 1024ull);
    Arena init_arena(arena_blk.ptr, arena_blk.len);

    auto [result, instance] = create_instance();

    VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
    if (result == VK_SUCCESS) {
        auto s     = create_surface(instance, desc);
        result     = s.result;
        vk_surface = s.surface;
    }

    PhysicalDeviceInfo physical_device_info{};
    if (result == VK_SUCCESS) {
        auto p = select_physical_device(instance, vk_surface, desc.gpu_preference, init_arena);
        result = p.result;
        physical_device_info = p;
    }

    VkDevice logical_device = VK_NULL_HANDLE;
    if (result == VK_SUCCESS) {
        auto l         = create_logical_device(desc, init_arena, physical_device_info);
        result         = l.result;
        logical_device = l.logical_device;
    }

    alloc.free(arena_blk);

    VolkDeviceTable api;
    VmaAllocator    vma = VK_NULL_HANDLE;
    if (result == VK_SUCCESS) {
        volkLoadDeviceTable(&api, logical_device);
        auto vma_result
            = create_vma_allocator(instance, physical_device_info.device, logical_device);
        result = vma_result.result;
        vma    = vma_result.allocator;
    }

    Vector<VkSampler> immutable_samplers;
    if (result == VK_SUCCESS) {
        immutable_samplers = create_immutable_samplers(alloc, api, logical_device, desc.samplers);
        result             = desc.samplers.size() == immutable_samplers.size() ? VK_SUCCESS
                                                                               : VK_ERROR_INITIALIZATION_FAILED;
    }

    VkDescriptorSetLayout default_descriptor_layout;
    VkPipelineLayout      default_graphics_layout;
    VkPipelineLayout      default_compute_layout;
    if (result == VK_SUCCESS) {
        default_descriptor_layout = create_descriptor_layout(logical_device,
                                                             api,
                                                             kMaxTextureHeapSize,
                                                             immutable_samplers);
        default_graphics_layout
            = create_default_graphics_layout(logical_device, api, default_descriptor_layout);
        default_compute_layout
            = create_default_compute_layout(logical_device, api, default_descriptor_layout);

        result = (default_descriptor_layout != nullptr && default_graphics_layout != nullptr
                  && default_compute_layout != nullptr)
                     ? VK_SUCCESS
                     : VK_ERROR_INITIALIZATION_FAILED;
    }

    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    if (result == VK_SUCCESS) {
        VkDescriptorPoolSize pool_sizes[] = {
            {
                .type            = VK_DESCRIPTOR_TYPE_SAMPLER,
                .descriptorCount = immutable_samplers.size(),
            },
            {
                .type            = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .descriptorCount = kMaxTextureHeapSize,
            },
        };

        VkDescriptorPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT
                     | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets       = kMaxNumTextureHeaps,
            .poolSizeCount = 2,
            .pPoolSizes    = pool_sizes,
        };

        result = api.vkCreateDescriptorPool(logical_device, &pool_info, nullptr, &descriptor_pool);
    }

    MemoryRequirements memory_requirements{};
    if (result == VK_SUCCESS) {
        memory_requirements = get_memory_requirements(api, logical_device);
        result              = memory_requirements.result;
    }

    if (result != VK_SUCCESS) {
        // TODO: Cleanup other resources
        if (instance) { vkDestroyInstance(instance, nullptr); }
        return nullptr;
    }

    // NOTE: Not 100% sure about if this is UB - I need a pointer to the device in order to capture
    // it in the lambdas, that are created during construction of the object. It probably is
    // technically undefined behviour. I'm also pretty certain it's totally fine in practice, if a
    // bit wonky.
    auto d = reinterpret_cast<DeviceImpl*>(blk.ptr);
    return new (blk.ptr) DeviceImpl{
        .m_allocator                  = alloc,
        .m_log_callback               = desc.log_callback,
        .m_log_userdata               = desc.log_userdata,
        .m_log_level                  = desc.log_level,
        .m_tls_key                    = loon::gpu::tls_alloc([](void* data) {
            auto state = reinterpret_cast<ThreadLocalState*>(data);
            state->~ThreadLocalState();
        }),
        .m_instance                   = instance,
        .m_physical_device            = physical_device_info.device,
        .m_graphics_queue_family      = physical_device_info.graphics_queue_family,
        .m_transfer_queue_family      = physical_device_info.transfer_queue_family,
        .m_async_compute_queue_family = physical_device_info.async_compute_queue_family,
        .m_surface                    = {
            .surface = vk_surface,
        },
        .m_device                     = logical_device,
        .m_api                        = std::move(api),
        .m_vma                        = vma,

        .m_descriptor_pool = descriptor_pool,
        .m_default_descriptor_layout = default_descriptor_layout,
        .m_default_graphics_layout = default_graphics_layout,
        .m_default_compute_layout = default_compute_layout,
        .m_gpu_mem_requirements = memory_requirements.gpu_mem_requirements,
        .m_buffer_mem_requirements = memory_requirements.buffer_mem_requirements,

        .m_buffer_pool
        = SlotMap<Buffer>(alloc,
                          [d](Buffer* b) {
                              vmaDestroyBuffer(d->m_vma, b->vk_buffer, b->vk_allocation);
                              auto it = lower_bound(d->m_ptr_map.begin(),
                                                    d->m_ptr_map.end(),
                                                    GpuPtrMap{.ptr = b->device_ptr});
                              d->m_ptr_map.erase(it, it + 1);
                              b->~Buffer();
                          }),
        .m_texture_pool = SlotMap<Texture>(
            alloc,
            [d](Texture* t) {
                if (t->default_image_view != VK_NULL_HANDLE) {
                    d->m_api.vkDestroyImageView(d->m_device, t->default_image_view, nullptr);
                }
                if (t->vk_allocation != VK_NULL_HANDLE) {
                    vmaDestroyImage(d->m_vma, t->vk_image, t->vk_allocation);
                } else if (t->vk_image && !t->is_swapchain_image) {
                    d->m_api.vkDestroyImage(d->m_device, t->vk_image, nullptr);
                }
            }),
        .m_texture_heap_pool
        = SlotMap<TextureHeap>(alloc,
                               [d](TextureHeap* h) {
                                   d->m_api.vkFreeDescriptorSets(d->m_device,
                                                                 d->m_descriptor_pool,
                                                                 1,
                                                                 &h->vk_descriptor_set);
                                   for (auto v : h->image_views) {
                                       // TODO: Can we be faster than iterating over the entire
                                       // vector? Maybe iterate over the bitset instead
                                       if (v != VK_NULL_HANDLE) {
                                           d->m_api.vkDestroyImageView(d->m_device, v, nullptr);
                                       }
                                   }
                                   h->~TextureHeap();
                               }),
        .m_depth_stencil_pool
        = SlotMap<DepthStencilState>(alloc, [](DepthStencilState* d) { d->~DepthStencilState(); }),
        .m_semaphore_pool
        = SlotMap<Semaphore>(alloc,
                             [d](Semaphore* s) {
                                 d->m_api.vkDestroySemaphore(d->m_device, s->vk_semaphore, nullptr);
                                 s->~Semaphore();
                             }),
        .m_pipeline_pool
        = SlotMap<Pipeline>(alloc,
                            [d](Pipeline* p) {
                                d->m_api.vkDestroyPipeline(d->m_device, p->vk_pipeline, nullptr);
                                p->~Pipeline();
                            }),
        .m_immutable_samplers = std::move(immutable_samplers),
        .m_ptr_map            = Vector<GpuPtrMap>(alloc),
    };
}

void destroy_device(Device d) {
    device_wait_for_idle(d);
    unconfigure_surface(d);

    for (auto& q : d->m_queues) {
        if (q.queue != VK_NULL_HANDLE) {
            for (auto& p : q.command_superpool.pools) {
                if (p.command_pool) {
                    d->m_api.vkDestroyCommandPool(d->m_device, p.command_pool, nullptr);
                }
            }
        }
    }

    d->m_buffer_pool.clear();
    d->m_texture_pool.clear();
    d->m_texture_heap_pool.clear();
    d->m_semaphore_pool.clear();
    d->m_pipeline_pool.clear();

    for (auto& s : d->m_immutable_samplers) { d->m_api.vkDestroySampler(d->m_device, s, nullptr); }

    d->m_api.vkDestroyDescriptorPool(d->m_device, d->m_descriptor_pool, nullptr);
    d->m_api.vkDestroyDescriptorSetLayout(d->m_device, d->m_default_descriptor_layout, nullptr);
    d->m_api.vkDestroyPipelineLayout(d->m_device, d->m_default_graphics_layout, nullptr);
    d->m_api.vkDestroyPipelineLayout(d->m_device, d->m_default_compute_layout, nullptr);

    vmaDestroyAllocator(d->m_vma);

    d->m_api.vkDestroyDevice(d->m_device, nullptr);

    if (d->m_surface.surface) { vkDestroySurfaceKHR(d->m_instance, d->m_surface.surface, nullptr); }

    vkDestroyInstance(d->m_instance, nullptr);
    volkFinalize();

    tls_free(d->m_tls_key);

    auto allocator = d->m_allocator;
    d->~DeviceImpl();

    allocator.free({.ptr = d, .len = sizeof(DeviceImpl)});
}

void device_wait_for_idle(Device d) {
    chk(d, d->m_api.vkDeviceWaitIdle(d->m_device));
}

// MARK: Surface

SurfaceCapabilities get_surface_capabilities(Device d) {
    // Surface formats:
    uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(d->m_physical_device,
                                         d->m_surface.surface,
                                         &format_count,
                                         nullptr);

    Arena arena = *get_thread_local_arena(d);  // Just copy the arena so all memory will be "freed"
                                               // at function exit.

    auto vk_formats = reinterpret_cast<VkSurfaceFormatKHR*>(
        arena.alloc(sizeof(VkSurfaceFormatKHR) * format_count));
    vkGetPhysicalDeviceSurfaceFormatsKHR(d->m_physical_device,
                                         d->m_surface.surface,
                                         &format_count,
                                         vk_formats);

    d->m_surface.num_supported_formats = 0;
    for (uint32_t i = 0; i < format_count; ++i) {
        Format fmt = bridge(vk_formats[i].format);
        if (fmt < Format::ValidCount) {
            d->m_surface.supported_formats[d->m_surface.num_supported_formats] = fmt;
            d->m_surface.num_supported_formats++;
        }
    }

    // Present modes:
    uint32_t present_mode_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(d->m_physical_device,
                                              d->m_surface.surface,
                                              &present_mode_count,
                                              nullptr);
    auto vk_modes = reinterpret_cast<VkPresentModeKHR*>(
        arena.alloc(sizeof(VkPresentModeKHR) * present_mode_count));

    vkGetPhysicalDeviceSurfacePresentModesKHR(d->m_physical_device,
                                              d->m_surface.surface,
                                              &present_mode_count,
                                              vk_modes);

    for (uint32_t i = 0; i < present_mode_count; ++i) {
        PresentMode mode = bridge(vk_modes[i]);
        if (mode < PresentMode::ValidCount) {
            d->m_surface.supported_present_modes[d->m_surface.num_supported_present_modes] = mode;
            d->m_surface.num_supported_present_modes++;
        }
    }

    // Capabilities has usages
    VkSurfaceCapabilitiesKHR vk_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d->m_physical_device,
                                              d->m_surface.surface,
                                              &vk_capabilities);

    return SurfaceCapabilities{
        .usages = bridge_usage_flags(vk_capabilities.supportedUsageFlags),
        .formats
        = Span<const Format>(d->m_surface.supported_formats, d->m_surface.num_supported_formats),
        .present_modes = Span<const PresentMode>(d->m_surface.supported_present_modes,
                                                 d->m_surface.num_supported_present_modes),
    };
}

bool configure_surface(Device d, const SurfaceConfiguration& config) {
    // Create the VkSwapchain based on the configuration
    VkSurfaceCapabilitiesKHR vk_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d->m_physical_device,
                                              d->m_surface.surface,
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
        .surface               = d->m_surface.surface,
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
        .oldSwapchain          = d->m_surface.swapchain,
    };

    if (!chk(d,
             d->m_api.vkCreateSwapchainKHR(d->m_device,
                                           &swapchain_info,
                                           nullptr,
                                           &d->m_surface.swapchain))) {
        log(d, LogLevel::Error, "Failed in call to vkCreateSwapchainKHR"_sv);
        return false;
    }

    image_count = 0;
    if (!chk(d,
             d->m_api.vkGetSwapchainImagesKHR(d->m_device,
                                              d->m_surface.swapchain,
                                              &image_count,
                                              nullptr))) {
        log(d, LogLevel::Error, "Failed in call to vkGetSwapchainImagesKHR"_sv);
        return false;
    }

    if (image_count > Surface::kMaxSwapchainImages) {
        log(d, LogLevel::Error, "Swapchain creating too many images"_sv);
        return false;
    }

    VkImage swapchain_images[Surface::kMaxSwapchainImages];

    if (!chk(d,
             d->m_api.vkGetSwapchainImagesKHR(d->m_device,
                                              d->m_surface.swapchain,
                                              &image_count,
                                              swapchain_images))) {
        log(d, LogLevel::Error, "Swapchain failed to retrieve images"_sv);
        return false;
    }

    const VkSemaphoreCreateInfo semaphore_create_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    // Create the acquire semaphores
    for (uint32_t i = 0; i < Surface::kMaxFramesInFlight; ++i) {
        VkSemaphore s = VK_NULL_HANDLE;
        chk(d, d->m_api.vkCreateSemaphore(d->m_device, &semaphore_create_info, nullptr, &s));
        d->m_surface.acquire_semaphores[i] = d->m_semaphore_pool.emplace({s});
    }

    // Convert the swapchain images here to handles, by inserting them into the object pool. Also
    // create the present semaphores.
    for (int i = 0; i < image_count; ++i) {
        VkImageView default_image_view = VK_NULL_HANDLE;
        if ((config.usages & UsageFlags::ColorAttachment) != UsageFlags::None
            || (config.usages & UsageFlags::DepthStencilAttachment) != UsageFlags::None) {
            const VkImageViewCreateInfo view_info {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags      = 0,
                .image      = swapchain_images[i],
                .viewType   = VK_IMAGE_VIEW_TYPE_2D,
                .format     = bridge(config.format),
                .components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY,},
                .subresourceRange = {
                    .aspectMask = aspects_for_format(config.format),
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };

            chk(d,
                d->m_api.vkCreateImageView(d->m_device, &view_info, nullptr, &default_image_view));
        }

        d->m_surface.swapchain_images[i] = d->m_texture_pool.emplace({
            .vk_image           = swapchain_images[i],
            .default_image_view = default_image_view,
            .vk_allocation      = VK_NULL_HANDLE,
            .is_swapchain_image = true,
        });

        VkSemaphore s = VK_NULL_HANDLE;
        chk(d, d->m_api.vkCreateSemaphore(d->m_device, &semaphore_create_info, nullptr, &s));
        d->m_surface.present_semaphores[i] = d->m_semaphore_pool.emplace({s});
    }

    d->m_surface.frame_idx       = 0;
    d->m_surface.frame_semaphore = create_semaphore(d, 0);

    return true;
}

void unconfigure_surface(Device d) {
    if (d->m_surface.swapchain) {
        d->m_api.vkDestroySwapchainKHR(d->m_device, d->m_surface.swapchain, nullptr);
        d->m_surface.swapchain         = VK_NULL_HANDLE;
        d->m_surface.image_count       = 0;
        d->m_surface.current_image_idx = 0;
        for (int i = 0; i < d->m_surface.image_count; ++i) {
            d->m_semaphore_pool.erase(d->m_surface.present_semaphores[i]);
            d->m_surface.transitioning_command[i] = VK_NULL_HANDLE;
        }
        for (uint32_t i = 0; i < Surface::kMaxFramesInFlight; ++i) {
            d->m_semaphore_pool.erase(d->m_surface.acquire_semaphores[i]);
        }
        free(d, d->m_surface.frame_semaphore);
    }
}

SurfaceTextureInfo get_current_texture(Device d) {
    d->m_surface.frame_idx++;
    auto semaphore
        = d->m_surface.acquire_semaphores[d->m_surface.frame_idx % Surface::kMaxFramesInFlight];
    d->m_surface.first_use_command[d->m_surface.frame_idx % Surface::kMaxFramesInFlight]
        = VK_NULL_HANDLE;
    const uint64_t wait_value = d->m_surface.frame_idx > Surface::kMaxFramesInFlight
                                    ? d->m_surface.frame_idx - Surface::kMaxFramesInFlight
                                    : 0;
    wait_semaphore(d, d->m_surface.frame_semaphore, wait_value);

    VkAcquireNextImageInfoKHR acquire_info{
        .sType      = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,
        .pNext      = nullptr,
        .swapchain  = d->m_surface.swapchain,
        .timeout    = 0,
        .semaphore  = d->m_semaphore_pool[semaphore].vk_semaphore,
        .fence      = VK_NULL_HANDLE,
        .deviceMask = 1,
    };
    uint32_t image_idx = 0;
    VkResult result    = d->m_api.vkAcquireNextImage2KHR(d->m_device, &acquire_info, &image_idx);
    SurfaceTextureInfo info{
        .status  = SurfaceStatus::Success,
        .texture = d->m_surface.swapchain_images[image_idx],
    };
    d->m_surface.current_image_idx = image_idx;

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        info.status = SurfaceStatus::OutOfDate;
    } else if (result == VK_SUBOPTIMAL_KHR) {
        info.status = SurfaceStatus::Suboptimal;
    } else if (result < 0) {
        log(d, LogLevel::Error, "Error in swapchain acquireNextImage"_sv);
        info.status = SurfaceStatus::Error;
    } else if (result != VK_SUCCESS) {
        log(d, LogLevel::Error, "Unknown swapchain status"_sv);
        info.status = SurfaceStatus::OutOfDate;
    }

    return info;
}

SurfaceStatus present(Device d, Queue q) {
    auto presenting_texture_handle = d->m_surface.swapchain_images[d->m_surface.current_image_idx];

    auto s = d->m_semaphore_pool[d->m_surface.present_semaphores[d->m_surface.current_image_idx]];
    VkPresentInfoKHR present_info{
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext              = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = &s.vk_semaphore,
        .swapchainCount     = 1,
        .pSwapchains        = &d->m_surface.swapchain,
        .pImageIndices      = &d->m_surface.current_image_idx,
        .pResults           = nullptr,
    };

    VkResult res = d->m_api.vkQueuePresentKHR(q->queue, &present_info);

    switch (res) {
        case VK_SUCCESS: return SurfaceStatus::Success;
        case VK_SUBOPTIMAL_KHR: return SurfaceStatus::Suboptimal;
        case VK_ERROR_OUT_OF_DATE_KHR: return SurfaceStatus::OutOfDate;
        default: chk(d, res); return SurfaceStatus::Error;
    }
}

// MARK: Buffers

Handle<Buffer> malloc(Device d, size_t bytes, Memory memory) {
    return malloc(d, bytes, 64, memory);
}

Handle<Buffer> malloc(Device d, size_t bytes, size_t align, Memory memory) {
    // TODO: Alignment is not respected currently.

    constexpr VkBufferUsageFlags kDefaultUsages
        = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
          | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
          | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
          | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

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

    VmaAllocationCreateFlags flags    = 0;
    VkMemoryPropertyFlags    vk_flags = 0;
    switch (memory) {
        case Memory::Default:
            flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            vk_flags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            break;
        case Memory::Gpu:
            flags    = 0;
            vk_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        case Memory::Readback:
            flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            vk_flags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            break;
    }

    VmaAllocationCreateInfo alloc_info{
        .flags          = flags,
        .usage          = VMA_MEMORY_USAGE_UNKNOWN,
        .requiredFlags  = vk_flags,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool           = VK_NULL_HANDLE,
        .pUserData      = 0,
        .priority       = 0.,
    };

    VkBuffer          vk_buffer     = nullptr;
    VmaAllocation     vk_allocation = nullptr;
    VmaAllocationInfo vma_alloc_info;

    VkMemoryRequirements memory_requirements
        = memory == Memory::Gpu ? d->m_gpu_mem_requirements : d->m_buffer_mem_requirements;
    memory_requirements.alignment
        = memory_requirements.alignment > align ? memory_requirements.alignment : align;
    memory_requirements.size = bytes;

    chk(d, d->m_api.vkCreateBuffer(d->m_device, &create_info, nullptr, &vk_buffer));

    chk(d,
        vmaAllocateMemory(d->m_vma,
                          &memory_requirements,
                          &alloc_info,
                          &vk_allocation,
                          &vma_alloc_info));

    chk(d, vmaBindBufferMemory(d->m_vma, vk_allocation, vk_buffer));

    VkBufferDeviceAddressInfo addr_info{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = vk_buffer,
    };
    GpuPtr device_ptr = d->m_api.vkGetBufferDeviceAddress(d->m_device, &addr_info);

    const auto handle = d->m_buffer_pool.emplace(Buffer{
        .vk_buffer     = vk_buffer,
        .vk_allocation = vk_allocation,
        .host_ptr      = vma_alloc_info.pMappedData,
        .device_ptr    = device_ptr,
    });

    // TODO: Don't re-sort the whole thing, insert in the right spot.
    const auto insertion_pos
        = lower_bound(d->m_ptr_map.begin(), d->m_ptr_map.end(), {.ptr = device_ptr});
    d->m_ptr_map.insert(insertion_pos, {.ptr = device_ptr, .buffer = handle});

    return handle;
}

void free(Device d, Handle<Buffer> buffer) {
    d->m_buffer_pool.erase(buffer);
}

GpuPtr get_device_pointer(Device d, Handle<Buffer> buffer) {
    return d->m_buffer_pool[buffer].device_ptr;
}

void* get_host_pointer(Device d, Handle<Buffer> buffer) {
    return d->m_buffer_pool[buffer].host_ptr;
}

BufferAndOffset buffer_and_offset_from_ptr(Device d, GpuPtr ptr) {
    const auto  it = lower_bound(d->m_ptr_map.begin(), d->m_ptr_map.end(), GpuPtrMap{.ptr = ptr});
    const auto& b  = d->m_buffer_pool[it->buffer];
    return {
        .buffer = b.vk_buffer,
        .offset = static_cast<uint32_t>(ptr - b.device_ptr),
        .alloc  = b.vk_allocation,
    };
}

// MARK: Textures

TextureSizeAlign get_texture_size_align(Device d, const TextureDesc& desc) {
    VkImageCreateInfo info{
        .sType     = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext     = nullptr,
        .flags     = 0,
        .imageType = bridge(desc.type),
        .format    = bridge(desc.format),
        .extent
        = {.width = desc.dimensions.x, .height = desc.dimensions.y, .depth = desc.dimensions.z},
        .mipLevels             = desc.mip_count,
        .arrayLayers           = desc.layer_count,
        .samples               = VK_SAMPLE_COUNT_1_BIT,  // TODO: Support multisampling
        .tiling                = VK_IMAGE_TILING_OPTIMAL,
        .usage                 = bridge_usage_flags(desc.usage),
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VkImage vk_image;
    chk(d, d->m_api.vkCreateImage(d->m_device, &info, nullptr, &vk_image));

    VkMemoryRequirements requirements{};
    d->m_api.vkGetImageMemoryRequirements(d->m_device, vk_image, &requirements);
    d->m_api.vkDestroyImage(d->m_device, vk_image, nullptr);

    return {.size = requirements.size, .align = requirements.alignment};
}

Handle<Texture> create_texture(Device d, const TextureDesc& desc, GpuPtr location) {
    VkImageCreateInfo info{
        .sType     = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext     = nullptr,
        .flags     = 0,
        .imageType = bridge(desc.type),
        .format    = bridge(desc.format),
        .extent
        = {.width = desc.dimensions.x, .height = desc.dimensions.y, .depth = desc.dimensions.z},
        .mipLevels             = desc.mip_count,
        .arrayLayers           = desc.layer_count,
        .samples               = VK_SAMPLE_COUNT_1_BIT,  // TODO: Support multisampling
        .tiling                = VK_IMAGE_TILING_OPTIMAL,
        .usage                 = bridge_usage_flags(desc.usage),
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VkImage       image      = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    if (location != 0) {
        chk(d, d->m_api.vkCreateImage(d->m_device, &info, nullptr, &image));

        auto memory = buffer_and_offset_from_ptr(d, location);
        vmaBindImageMemory2(d->m_vma, memory.alloc, memory.offset, image, nullptr);
    } else {
        VmaAllocationCreateInfo alloc_info{
            .flags          = 0,
            .usage          = VMA_MEMORY_USAGE_AUTO,
            .requiredFlags  = 0,
            .preferredFlags = 0,
            .memoryTypeBits = 0,
            .pool           = VK_NULL_HANDLE,
            .pUserData      = 0,
            .priority       = 0.,
        };

        if (!chk(d, vmaCreateImage(d->m_vma, &info, &alloc_info, &image, &allocation, nullptr))) {
            return {};
        }
    }
    // Create a default image view for use as a render target attachment.
    VkImageView default_image_view = VK_NULL_HANDLE;
    if ((desc.usage & UsageFlags::ColorAttachment) != UsageFlags::None
        || (desc.usage & UsageFlags::DepthStencilAttachment) != UsageFlags::None) {
        const VkImageViewCreateInfo view_info {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags      = 0,
                .image      = image,
                .viewType   = VK_IMAGE_VIEW_TYPE_2D,
                .format     = bridge(desc.format),
                .components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY,},
                .subresourceRange = {
                    .aspectMask = aspects_for_format(desc.format),
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };

        chk(d, d->m_api.vkCreateImageView(d->m_device, &view_info, nullptr, &default_image_view));
    }

    const auto handle = d->m_texture_pool.emplace(Texture{
        .vk_image           = image,
        .default_image_view = default_image_view,
        .vk_allocation      = allocation,
        .vk_type            = bridge_view_type(desc.type),
        .format             = desc.format,
    });

    return handle;
}

Handle<TextureHeap> create_texture_heap(Device d, size_t size) {
    const uint32_t                                           descriptor_count = size;
    const VkDescriptorSetVariableDescriptorCountAllocateInfo allocate_info    = {
           .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
           .pNext = nullptr,
           .descriptorSetCount = 1,
           .pDescriptorCounts  = &descriptor_count,
    };

    VkDescriptorSetAllocateInfo info{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext              = &allocate_info,
        .descriptorPool     = d->m_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &d->m_default_descriptor_layout,
    };

    VkDescriptorSet set;
    chk(d, d->m_api.vkAllocateDescriptorSets(d->m_device, &info, &set));

    const auto handle = d->m_texture_heap_pool.emplace(TextureHeap{
        .vk_descriptor_set = set,
        .bitset            = TwoLevelBitset(d->m_allocator, size),
        .image_views       = Vector<VkImageView>(d->m_allocator, VkImageView{0}, size),
    });

    return {handle};
}

TextureView add_texture_view_to_heap(Device                 d,
                                     Handle<TextureHeap>    heap,
                                     const TextureViewDesc& desc) {
    auto& texture_heap = d->m_texture_heap_pool[heap];
    auto  texture      = d->m_texture_pool[desc.texture];

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
        .aspectMask = aspects_for_format(desc.format),
        .baseMipLevel = desc.base_mip,
        .levelCount = desc.mip_count,
        .baseArrayLayer = desc.base_layer,
        .layerCount = desc.layer_count,
        },
    };
    VkImageView image_view;
    if (!chk(d, d->m_api.vkCreateImageView(d->m_device, &info, nullptr, &image_view))) {
        return -1;
    }

    uint32_t free_slot                  = texture_heap.bitset.set_leading_zero();
    texture_heap.image_views[free_slot] = image_view;

    const VkDescriptorImageInfo image_info{
        .sampler     = VK_NULL_HANDLE,
        .imageView   = image_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    const VkWriteDescriptorSet write{
        .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext            = nullptr,
        .dstSet           = texture_heap.vk_descriptor_set,
        .dstBinding       = 2,
        .dstArrayElement  = free_slot,
        .descriptorCount  = 1,
        .descriptorType   = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo       = &image_info,
        .pBufferInfo      = nullptr,
        .pTexelBufferView = nullptr,
    };
    d->m_api.vkUpdateDescriptorSets(d->m_device, 1, &write, 0, nullptr);

    return free_slot;
}

void free(Device d, Handle<Texture> t) {
    d->m_texture_pool.erase(t);
}

void remove_texture_view_from_heap(Device d, Handle<TextureHeap> heap, TextureView idx) {
    auto& texture_heap = d->m_texture_heap_pool[heap];
    texture_heap.bitset.clear_bit(idx);

    d->m_api.vkDestroyImageView(d->m_device, texture_heap.image_views[idx], nullptr);
    texture_heap.image_views[idx] = 0;

    // This isn't strictly valid without an extension feature from VK_KHR_robustness2.
    // What's a better option?
    // const VkDescriptorImageInfo image_info{
    //     .sampler     = VK_NULL_HANDLE,
    //     .imageView   = VK_NULL_HANDLE,
    //     .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    // };

    // const VkWriteDescriptorSet write{
    //     .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    //     .pNext            = nullptr,
    //     .dstSet           = texture_heap.vk_descriptor_set,
    //     .dstBinding       = 2,
    //     .dstArrayElement  = idx,
    //     .descriptorCount  = 1,
    //     .descriptorType   = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    //     .pImageInfo       = &image_info,
    //     .pBufferInfo      = nullptr,
    //     .pTexelBufferView = nullptr,
    // };
    // d->m_api.vkUpdateDescriptorSets(d->m_device, 1, &write, 0, nullptr);
}

void free(Device d, Handle<TextureHeap> heap) {
    d->m_texture_heap_pool.erase(heap);
}

// MARK: Pipelines

Handle<Pipeline> create_compute_pipeline(Device d, ShaderSource source) {
    VkShaderModuleCreateInfo module_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = source.spirv.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(source.spirv.data()),
    };

    VkShaderModule module;
    chk(d, d->m_api.vkCreateShaderModule(d->m_device, &module_info, nullptr, &module));

    VkComputePipelineCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = module,
            .pName = source.entry_point.data(), // TODO: Need to ensure null-terminated, copy to local arena.
            .pSpecializationInfo = nullptr,
        }, 
        .layout = d->m_default_compute_layout, 
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    };

    VkPipeline pipeline;
    if (!chk(d,
             d->m_api.vkCreateComputePipelines(d->m_device,
                                               VK_NULL_HANDLE,
                                               1,
                                               &info,
                                               nullptr,
                                               &pipeline))) {
        return {};
    }

    d->m_api.vkDestroyShaderModule(d->m_device, module, nullptr);

    auto h = d->m_pipeline_pool.emplace({
        .vk_pipeline = pipeline,
        .bind_point  = VK_PIPELINE_BIND_POINT_COMPUTE,
    });
    return h;
}

Handle<Pipeline> create_graphics_pipeline(Device            d,
                                          ShaderSource      vertex,
                                          ShaderSource      fragment,
                                          const RasterDesc& desc) {
    VkShaderModuleCreateInfo vert_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = vertex.spirv.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(vertex.spirv.data()),
    };
    VkShaderModule vert_module;
    chk(d, d->m_api.vkCreateShaderModule(d->m_device, &vert_info, nullptr, &vert_module));

    VkShaderModuleCreateInfo frag_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = fragment.spirv.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(fragment.spirv.data()),
    };
    VkShaderModule frag_module;
    chk(d, d->m_api.vkCreateShaderModule(d->m_device, &frag_info, nullptr, &frag_module));

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
    VkFormat depth_attachment_format   = bridge(desc.depth_format);
    VkFormat stencil_attachment_format = bridge(desc.stencil_format);

    // Color blend state
    Arena                                     arena = *get_thread_local_arena(d);
    Span<VkPipelineColorBlendAttachmentState> color_blend_attachment_states{};
    Span<VkFormat>                            color_attachment_formats{};

    for (auto& t : desc.color_targets) {
        // const auto attachment_state = loon::gpu::bridge(target);
        color_blend_attachment_states
            = concat(&arena, color_blend_attachment_states, loon::gpu::bridge(t.blendstate));
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
        case Cull::CCW: front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE; break;
        case Cull::CW: front_face = VK_FRONT_FACE_CLOCKWISE; break;
        case Cull::All: cull_mode = VK_CULL_MODE_FRONT_AND_BACK; break;
        case Cull::None: cull_mode = VK_CULL_MODE_NONE; break;
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
        .layout              = d->m_default_graphics_layout,
        .renderPass          = VK_NULL_HANDLE,
        .subpass             = 0,
        .basePipelineHandle  = VK_NULL_HANDLE,
        .basePipelineIndex   = 0,
    };
    VkPipeline vk_pipeline;
    chk(d,
        d->m_api.vkCreateGraphicsPipelines(d->m_device,
                                           VK_NULL_HANDLE,
                                           1,
                                           &create_info,
                                           nullptr,
                                           &vk_pipeline));

    d->m_api.vkDestroyShaderModule(d->m_device, vert_module, nullptr);
    d->m_api.vkDestroyShaderModule(d->m_device, frag_module, nullptr);

    auto h = d->m_pipeline_pool.emplace({
        .vk_pipeline = vk_pipeline,
        .bind_point  = VK_PIPELINE_BIND_POINT_GRAPHICS,
    });

    return h;
}

void free(Device d, Handle<Pipeline> pipeline) {
    d->m_pipeline_pool.erase(pipeline);
}

Handle<DepthStencilState> create_depth_stencil_state(Device d, const DepthStencilDesc& desc) {
    auto h = d->m_depth_stencil_pool.emplace(DepthStencilState{desc});
    return h;
}

// MARK: Queue

Queue get_queue(Device d, QueueType type) {
    // Initialize the queue on-demand.
    if (d->m_queues[static_cast<uint32_t>(type)].queue == VK_NULL_HANDLE) {
        uint32_t queue_family = 0;
        switch (type) {
            case QueueType::Default: queue_family = d->m_graphics_queue_family; break;
            case QueueType::Compute: queue_family = d->m_async_compute_queue_family; break;
            case QueueType::Transfer: queue_family = d->m_transfer_queue_family; break;
            case QueueType::ValidCount: break;
        }

        VkQueue queue;
        d->m_api.vkGetDeviceQueue(d->m_device, queue_family, 0, &queue);
        auto timeline = create_semaphore(d, 0);

        d->m_queues[static_cast<uint32_t>(type)] = {
            .device            = d,
            .queue             = queue,
            .command_superpool = {},
            .timeline          = timeline,
            .queue_family      = queue_family,
            .timeline_value    = 0,
            .pending_events    = Vector<QueueImpl::Event>(d->m_allocator),
        };
    }

    return &d->m_queues[static_cast<uint32_t>(type)];
}

// MARK: Sempahores

Handle<Semaphore> create_semaphore(Device d, uint64_t initValue) {
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
    chk(d, d->m_api.vkCreateSemaphore(d->m_device, &timeline_create_info, nullptr, &s));

    return d->m_semaphore_pool.emplace({.vk_semaphore = s});
}

void wait_semaphore(Device d, Handle<Semaphore> sema, uint64_t value) {
    VkSemaphore         s = d->m_semaphore_pool[sema].vk_semaphore;
    VkSemaphoreWaitInfo wait_info{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext          = nullptr,
        .flags          = 0,
        .semaphoreCount = 1,
        .pSemaphores    = &s,
        .pValues        = &value,
    };
    chk(d, d->m_api.vkWaitSemaphores(d->m_device, &wait_info, UINT64_MAX));
}

void free(Device d, Handle<Semaphore> sema) {
    d->m_semaphore_pool.erase(sema);
}

void log(Device d, LogLevel lvl, Span<const char> msg) {
    d->m_log_callback(lvl, msg, d->m_log_userdata);
};

bool chk(Device d, VkResult result) {
    if (result == VK_SUCCESS) { return true; }

    switch (result) {
        case VK_NOT_READY: log(d, LogLevel::Error, "VK_NOT_READY"_sv); break;
        case VK_TIMEOUT: log(d, LogLevel::Error, "VK_TIMEOUT"_sv); break;
        case VK_EVENT_SET: log(d, LogLevel::Error, "VK_EVENT_SET"_sv); break;
        case VK_EVENT_RESET: log(d, LogLevel::Error, "VK_EVENT_RESET"_sv); break;
        case VK_INCOMPLETE: log(d, LogLevel::Error, "VK_INCOMPLETE"_sv); break;
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            log(d, LogLevel::Error, "VK_ERROR_OUT_OF_HOST_MEMORY"_sv);
            break;
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            log(d, LogLevel::Error, "VK_ERROR_OUT_OF_DEVICE_MEMORY");
            break;
        case VK_ERROR_INITIALIZATION_FAILED:
            log(d, LogLevel::Error, "VK_ERROR_INITIALIZATION_FAILED");
            break;
        case VK_ERROR_DEVICE_LOST: log(d, LogLevel::Error, "VK_ERROR_DEVICE_LOST"); break;
        case VK_ERROR_MEMORY_MAP_FAILED:
            log(d, LogLevel::Error, "VK_ERROR_MEMORY_MAP_FAILED");
            break;
        case VK_ERROR_LAYER_NOT_PRESENT:
            log(d, LogLevel::Error, "VK_ERROR_LAYER_NOT_PRESENT");
            break;
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            log(d, LogLevel::Error, "VK_ERROR_EXTENSION_NOT_PRESENT");
            break;
        case VK_ERROR_FEATURE_NOT_PRESENT:
            log(d, LogLevel::Error, "VK_ERROR_FEATURE_NOT_PRESENT");
            break;
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            log(d, LogLevel::Error, "VK_ERROR_INCOMPATIBLE_DRIVER");
            break;
        case VK_ERROR_TOO_MANY_OBJECTS: log(d, LogLevel::Error, "VK_ERROR_TOO_MANY_OBJECTS"); break;
        case VK_ERROR_FORMAT_NOT_SUPPORTED:
            log(d, LogLevel::Error, "VK_ERROR_FORMAT_NOT_SUPPORTED");
            break;
        case VK_ERROR_FRAGMENTED_POOL: log(d, LogLevel::Error, "VK_ERROR_FRAGMENTED_POOL"); break;
        case VK_ERROR_UNKNOWN: log(d, LogLevel::Error, "VK_ERROR_UNKNOWN"); break;
        case VK_ERROR_OUT_OF_POOL_MEMORY:
            log(d, LogLevel::Error, "VK_ERROR_OUT_OF_POOL_MEMORY");
            break;
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:
            log(d, LogLevel::Error, "VK_ERROR_INVALID_EXTERNAL_HANDLE");
            break;
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
            log(d, LogLevel::Error, "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS");
            break;
        case VK_ERROR_FRAGMENTATION: log(d, LogLevel::Error, "VK_ERROR_FRAGMENTATION"); break;
        case VK_PIPELINE_COMPILE_REQUIRED:
            log(d, LogLevel::Error, "VK_PIPELINE_COMPILE_REQUIRED");
            break;
        case VK_ERROR_SURFACE_LOST_KHR: log(d, LogLevel::Error, "VK_ERROR_SURFACE_LOST_KHR"); break;
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
            log(d, LogLevel::Error, "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR");
            break;
        case VK_SUBOPTIMAL_KHR: log(d, LogLevel::Error, "VK_SUBOPTIMAL_KHR"); break;
        case VK_ERROR_OUT_OF_DATE_KHR: log(d, LogLevel::Error, "VK_ERROR_OUT_OF_DATE_KHR"); break;
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
            log(d, LogLevel::Error, "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR");
            break;
        case VK_ERROR_INVALID_SHADER_NV:
            log(d, LogLevel::Error, "VK_ERROR_INVALID_SHADER_NV");
            break;
        case VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR:
            log(d, LogLevel::Error, "VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR:
            log(d, LogLevel::Error, "VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR:
            log(d, LogLevel::Error, "VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR:
            log(d, LogLevel::Error, "VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR:
            log(d, LogLevel::Error, "VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR:
            log(d, LogLevel::Error, "VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR");
            break;
        case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:
            log(d, LogLevel::Error, "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT");
            break;
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
            log(d, LogLevel::Error, "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT");
            break;
        case VK_THREAD_IDLE_KHR: log(d, LogLevel::Error, "VK_THREAD_IDLE_KHR"); break;
        case VK_THREAD_DONE_KHR: log(d, LogLevel::Error, "VK_THREAD_DONE_KHR"); break;
        case VK_OPERATION_DEFERRED_KHR: log(d, LogLevel::Error, "VK_OPERATION_DEFERRED_KHR"); break;
        case VK_OPERATION_NOT_DEFERRED_KHR:
            log(d, LogLevel::Error, "VK_OPERATION_NOT_DEFERRED_KHR");
            break;
        case VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR:
            log(d, LogLevel::Error, "VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR");
            break;
        case VK_ERROR_COMPRESSION_EXHAUSTED_EXT:
            log(d, LogLevel::Error, "VK_ERROR_COMPRESSION_EXHAUSTED_EXT");
            break;
        case VK_INCOMPATIBLE_SHADER_BINARY_EXT:
            log(d, LogLevel::Error, "VK_INCOMPATIBLE_SHADER_BINARY_EXT");
            break;
        default: log(d, LogLevel::Error, "Unknown error"_sv); break;
    }

    return false;
}

Arena* get_thread_local_arena(Device d) {
    auto state = reinterpret_cast<ThreadLocalState*>(loon::gpu::tls_get_data(d->m_tls_key));
    if (state == nullptr) {
        auto tls_block = d->m_allocator.alloc(sizeof(ThreadLocalState));
        if (tls_block.ptr == nullptr) {
            log(d, LogLevel::Error, "Allocator out of memory"_sv);
            return nullptr;
        }
        state = ::new (tls_block.ptr) ThreadLocalState(d->m_allocator);
        loon::gpu::tls_set_data(d->m_tls_key, state);
    }
    return &state->arena;
}

// MARK: Queue

static void reset_command_pool(const VolkDeviceTable& api, VkDevice device, CommandPool* pool) {
    api.vkResetCommandPool(device, pool->command_pool, 0);
    pool->buffer_free_idx = 0;
}

CommandPool* get_command_pool(Queue queue, uint64_t frame_idx) {
    CommandSuperpool& superpool       = queue->command_superpool;
    CommandPool*      pool            = nullptr;
    int64_t           available_pools = atomic_load(&superpool.available_pools);
    bool              index_good      = false;
    uint64_t          idx;
    while (!index_good && available_pools != 0) {
        // Try to clear the lowest set bit using a compare_exchange loop.
        idx                    = count_trailing_zeros(available_pools);
        const uint64_t mask    = ~(1ull << idx);
        const int64_t  desired = static_cast<int64_t>(available_pools & mask);
        index_good = atomic_compare_exchange(&superpool.available_pools, &available_pools, desired);
    };

    if (index_good) {
        pool = &superpool.pools[CommandSuperpool::kPoolsPerGroup * idx
                                + (frame_idx % CommandSuperpool::kPoolsPerGroup)];

        if (pool->command_pool == VK_NULL_HANDLE) {
            // Initialize the command pool here.
            VkCommandPoolCreateInfo pool_info{
                .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext            = nullptr,
                .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                .queueFamilyIndex = queue->queue_family,
            };
            VkCommandPool command_pool = VK_NULL_HANDLE;
            chk(queue->device,
                queue->device->m_api.vkCreateCommandPool(queue->device->m_device,
                                                         &pool_info,
                                                         nullptr,
                                                         &command_pool));
            *pool = CommandPool{
                .command_pool    = command_pool,
                .command_buffers = Vector<CommandBufferImpl>(queue->device->m_allocator),
                .buffer_free_idx = 0,
            };
        } else if (pool->frame_idx != frame_idx) {
            // Last time this was used was on a different frame, so reset the pool.
            reset_command_pool(queue->device->m_api, queue->device->m_device, pool);
        }
    } else {
        log(queue->device,
            LogLevel::Error,
            "Unable to get command pool - too many command buffers in flight at once"_sv);
    }

    return pool;
}

static void release_command_pool(Queue q, CommandPool* pool) {
    auto&         superpool = q->command_superpool;
    const int64_t idx       = (pool - superpool.pools) / CommandSuperpool::kPoolsPerGroup;

    // Need to set the bit in available pools using a compare-exchange loop
    int64_t previous = atomic_load(&superpool.available_pools);

    int64_t desired = previous | (1ll << idx);
    while (!atomic_compare_exchange(&superpool.available_pools, &previous, desired)) {
        desired = previous | (1ll << idx);
    }
}

static CommandBufferImpl* get_command_buffer(Queue q, CommandPool* pool) {
    auto device = q->device;

    if (pool->command_buffers.size() <= pool->buffer_free_idx) {
        const VkCommandBufferAllocateInfo info{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext              = nullptr,
            .commandPool        = pool->command_pool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer buf;
        if (!chk(device, device->m_api.vkAllocateCommandBuffers(device->m_device, &info, &buf))) {
            return nullptr;
        }
        pool->command_buffers.emplace_back(CommandBufferImpl{
            .device = device,
            .queue  = q,
            .pool   = pool,
            .buffer = buf,
        });
    }

    CommandBufferImpl* result = pool->command_buffers.data() + pool->buffer_free_idx;
    pool->buffer_free_idx++;
    return result;
}

CommandBuffer queue_start_command_recording(Queue q) {
    auto d = q->device;

    CommandPool* pool = get_command_pool(q, d->m_surface.frame_idx);
    if (pool == nullptr) { return nullptr; }

    CommandBuffer buffer = get_command_buffer(q, pool);
    if (buffer) {
        const VkCommandBufferBeginInfo begin_info{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext            = nullptr,
            .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        chk(d, d->m_api.vkBeginCommandBuffer(buffer->buffer, &begin_info));
    }
    return buffer;
}

void cmd_finalize(CommandBuffer cmd) {
    auto d = cmd->device;
    auto q = cmd->queue;
    chk(d, d->m_api.vkEndCommandBuffer(cmd->buffer));

    release_command_pool(q, cmd->pool);
}

void queue_submit(Queue                     q,
                  Span<const CommandBuffer> command_buffers,
                  Span<const SemaphoreInfo> wait_semaphores,
                  Span<const SemaphoreInfo> signal_semaphores) {
    auto d = q->device;

    auto arena = *get_thread_local_arena(d);

    Span<VkSemaphoreSubmitInfo>     wait_info;
    Span<VkCommandBufferSubmitInfo> command_info;
    Span<VkSemaphoreSubmitInfo>     signal_info;

    for (uint32_t i = 0; i < wait_semaphores.size(); ++i) {
        wait_info = concat(
            &arena,
            wait_info,
            VkSemaphoreSubmitInfo{
                .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext       = nullptr,
                .semaphore   = d->m_semaphore_pool[wait_semaphores[i].semaphore].vk_semaphore,
                .value       = wait_semaphores[i].value,
                .stageMask   = bridge_pipeline_stage(wait_semaphores[i].stage),
                .deviceIndex = 0,
            });
    }

    for (uint32_t i = 0; i < command_buffers.size(); i++) {
        auto buf = command_buffers[i]->buffer;

        command_info = concat(&arena,
                              command_info,
                              VkCommandBufferSubmitInfo{
                                  .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                  .pNext         = nullptr,
                                  .commandBuffer = buf,
                                  .deviceMask    = 1,
                              });

        if (buf
            == d->m_surface
                   .first_use_command[d->m_surface.frame_idx % Surface::kMaxFramesInFlight]) {
            const auto acquire_semaphore
                = d->m_semaphore_pool[d->m_surface.acquire_semaphores
                                          [d->m_surface.frame_idx % Surface::kMaxFramesInFlight]]
                      .vk_semaphore;

            // TODO: PERF - we could check the configured surface usages to get a more optimal
            // stage mask - if it's only used as an attachment that can let more work run
            // concurrently.
            wait_info = concat(&arena,
                               wait_info,
                               VkSemaphoreSubmitInfo{
                                   .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                   .pNext       = nullptr,
                                   .semaphore   = acquire_semaphore,
                                   .value       = 0,
                                   .stageMask   = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                   .deviceIndex = 0,
                               });
        }
        if (buf == d->m_surface.transitioning_command[d->m_surface.current_image_idx]) {
            // Immediately clear this so we don't accidentally signal it twice (can happen due
            // to cmd buffer reuse,e.g.)
            d->m_surface.transitioning_command[d->m_surface.current_image_idx] = 0;
            VkSemaphore present_semaphore
                = d->m_semaphore_pool[d->m_surface
                                          .present_semaphores[d->m_surface.current_image_idx]]
                      .vk_semaphore;

            signal_info = concat(&arena,
                                 signal_info,
                                 VkSemaphoreSubmitInfo{
                                     .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                     .pNext       = nullptr,
                                     .semaphore   = present_semaphore,
                                     .value       = 0,
                                     .stageMask   = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     .deviceIndex = 0,
                                 });
            signal_info = concat(
                &arena,
                signal_info,
                VkSemaphoreSubmitInfo{
                    .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .pNext       = nullptr,
                    .semaphore   = d->m_semaphore_pool[d->m_surface.frame_semaphore].vk_semaphore,
                    .value       = d->m_surface.frame_idx,
                    .stageMask   = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
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
                .semaphore   = d->m_semaphore_pool[signal_semaphores[i].semaphore].vk_semaphore,
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
                             .semaphore   = d->m_semaphore_pool[q->timeline].vk_semaphore,
                             .value       = ++(q->timeline_value),
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

    d->m_api.vkQueueSubmit2(q->queue, 1, &submit_info, VK_NULL_HANDLE);
}

void queue_cancel(Queue q, Span<const Handle<CommandBuffer>> commandBuffers) {}

void queue_on_submitted_work_completed(Queue q, Function<void>&& fn) {
    q->pending_events.emplace_back(
        QueueImpl::Event{.completed_time = q->timeline_value, .callback = std::move(fn)});
}

void queue_process_events(Queue q) {
    auto     d            = q->device;
    uint64_t current_time = 0;
    chk(d,
        d->m_api.vkGetSemaphoreCounterValue(d->m_device,
                                            d->m_semaphore_pool[q->timeline].vk_semaphore,
                                            &current_time));
    uint32_t i = 0;
    while (i < q->pending_events.size() && q->pending_events[i].completed_time <= current_time) {
        q->pending_events[i].callback();
        i++;
    }
    if (i != 0) {
        q->pending_events.erase(q->pending_events.begin(), q->pending_events.begin() + i);
    }
}

// MARK: Commmand Buffer

void cmd_memcpy(CommandBuffer cmd, GpuPtr destGpu, GpuPtr srcGpu, size_t size) {
    auto impl = cmd->device;

    auto src = buffer_and_offset_from_ptr(impl, srcGpu);
    auto dst = buffer_and_offset_from_ptr(impl, destGpu);

    VkBufferCopy region{
        .srcOffset = src.offset,
        .dstOffset = dst.offset,
        .size      = size,
    };
    impl->m_api.vkCmdCopyBuffer(cmd->buffer, src.buffer, dst.buffer, 1, &region);
}

void cmd_copy_to_texture(CommandBuffer                  cmd,
                         GpuPtr                         srcPtr,
                         Handle<Texture>                texture,
                         const BufferToTextureCopyInfo& info) {
    auto impl = cmd->device;

    auto        src = buffer_and_offset_from_ptr(impl, srcPtr);
    const auto& tex = impl->m_texture_pool[texture];
    const VkBufferImageCopy region{
        .bufferOffset      = src.offset,
        .bufferRowLength   = info.buffer_image_size.x,
        .bufferImageHeight = info.buffer_image_size.y,
        .imageSubresource = {
            .aspectMask = aspects_for_format(tex.format),
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset
        = {.x = static_cast<int32_t>(info.image_offset.x), .y = static_cast<int32_t>(info.image_offset.y), .z = static_cast<int32_t>(info.image_offset.z),},
        .imageExtent = {.width = info.image_extent.x, .height = info.image_extent.y, .depth = info.image_extent.z,},
    };

    impl->m_api.vkCmdCopyBufferToImage(cmd->buffer,
                                       src.buffer,
                                       tex.vk_image,
                                       VK_IMAGE_LAYOUT_GENERAL,
                                       1,
                                       &region);
}

void cmd_copy_from_texture(CommandBuffer   cmd,
                           GpuPtr          destGpu,
                           GpuPtr          srcGpu,
                           Handle<Texture> texture) {
    assert(false);
}

void cmd_set_texture_heap(CommandBuffer cmd, Handle<TextureHeap> heap) {
    auto impl = cmd->device;

    auto vk_descriptor_set = impl->m_texture_heap_pool[heap].vk_descriptor_set;
    impl->m_api.vkCmdBindDescriptorSets(cmd->buffer,
                                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        impl->m_default_graphics_layout,
                                        0,
                                        1,
                                        &vk_descriptor_set,
                                        0,
                                        nullptr);
    impl->m_api.vkCmdBindDescriptorSets(cmd->buffer,
                                        VK_PIPELINE_BIND_POINT_COMPUTE,
                                        impl->m_default_compute_layout,
                                        0,
                                        1,
                                        &vk_descriptor_set,
                                        0,
                                        nullptr);
}

void cmd_barrier(CommandBuffer                 cmd,
                 StageFlags                    before,
                 StageFlags                    after,
                 Span<const TextureTransition> image_transitions,
                 HazardFlags                   hazards) {
    auto impl = cmd->device;
    // TODO: Use HazardFlags to reduce the stage/access_masks unless necessary.
    const auto src_stage = bridge_pipeline_stage(before);
    auto       dst_stage = bridge_pipeline_stage(after);
    if ((hazards & HazardFlags::DrawArguments) != HazardFlags::None) {
        dst_stage |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    }

    constexpr auto access = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

    const VkMemoryBarrier2 barrier_info{
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext         = nullptr,
        .srcStageMask  = src_stage,
        .srcAccessMask = access,
        .dstStageMask  = dst_stage,
        .dstAccessMask = access,
    };

    Arena arena = *get_thread_local_arena(impl);

    Span<VkImageMemoryBarrier2> image_barriers;
    for (const auto& t : image_transitions) {
        const auto& tex = impl->m_texture_pool[t.texture];
        image_barriers = concat(&arena,
                                image_barriers,
                                VkImageMemoryBarrier2{
                                    .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    .pNext            = nullptr,
                                    .srcStageMask     = src_stage,
                                    .srcAccessMask    = access,
                                    .dstStageMask     = dst_stage,
                                    .dstAccessMask    = t.new_layout == Layout::Present ? 0 : access,
                                    .oldLayout        = bridge(t.old_layout),
                                    .newLayout        = bridge(t.new_layout),
                                    .image            = tex.vk_image,
                                    .subresourceRange = VkImageSubresourceRange{
                                        .aspectMask = aspects_for_format(tex.format),
                                        .baseMipLevel = 0,
                                        .levelCount = VK_REMAINING_MIP_LEVELS,
                                        .baseArrayLayer =0 ,
                                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                    },
                                });


        if (t.texture == impl->m_surface.swapchain_images[impl->m_surface.current_image_idx]) {
            if (t.old_layout == Layout::DontCare || t.old_layout == Layout::Present) {
                auto* first_use_command
                    = &impl->m_surface.first_use_command[impl->m_surface.frame_idx
                                                         % Surface::kMaxFramesInFlight];
                int64_t expected = 0;
                // We try and set the first use command value if it hasn't already been set by
                // an earlier barrier.
                atomic_compare_exchange(reinterpret_cast<int64_t*>(first_use_command),
                                        &expected,
                                        reinterpret_cast<int64_t>(cmd->buffer));
            }
            if (t.new_layout == Layout::Present) {
                impl->m_surface.transitioning_command[impl->m_surface.current_image_idx]
                    = reinterpret_cast<VkCommandBuffer>(cmd->buffer);
            }
        }
    }

    const VkDependencyInfo info{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                    = nullptr,
        .dependencyFlags          = 0,
        .memoryBarrierCount       = 1,
        .pMemoryBarriers          = &barrier_info,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers    = nullptr,
        .imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size()),
        .pImageMemoryBarriers     = image_barriers.data(),
    };
    impl->m_api.vkCmdPipelineBarrier2(cmd->buffer, &info);
}

void cmd_set_pipeline(CommandBuffer cmd, Handle<Pipeline> pipeline) {
    auto  impl = cmd->device;
    auto& p    = impl->m_pipeline_pool[pipeline];
    impl->m_api.vkCmdBindPipeline(cmd->buffer, p.bind_point, p.vk_pipeline);
}

void cmd_set_depth_stencil_state(CommandBuffer cmd, Handle<DepthStencilState> state) {
    auto impl = cmd->device;

    auto& desc = impl->m_depth_stencil_pool[state];

    impl->m_api.vkCmdSetDepthWriteEnable(cmd->buffer, bool(desc.depth_mode & DepthFlags::Write));
    impl->m_api.vkCmdSetDepthTestEnable(cmd->buffer, bool(desc.depth_mode & DepthFlags::Read));
    impl->m_api.vkCmdSetDepthCompareOp(cmd->buffer, bridge(desc.depth_test));
    // TODO: More stuff here.
    impl->m_api.vkCmdSetStencilTestEnable(cmd->buffer, false);
    impl->m_api.vkCmdSetStencilOp(cmd->buffer,
                                  VK_STENCIL_FACE_FRONT_AND_BACK,
                                  VK_STENCIL_OP_KEEP,
                                  VK_STENCIL_OP_KEEP,
                                  VK_STENCIL_OP_KEEP,
                                  VK_COMPARE_OP_ALWAYS);
}

void cmd_set_scissor_rect(CommandBuffer cmd, const Rect2D& rect) {
    auto impl = cmd->device;
    const VkRect2D vk_rect{
        .offset = {.x = (int32_t)rect.offset_x, .y = (int32_t)rect.offset_y,},
        .extent = {.width = rect.width, .height = rect.height},
    };
    impl->m_api.vkCmdSetScissorWithCount(cmd->buffer, 1, &vk_rect);
}

static void cmd_set_compute_ptr(CommandBuffer cmd, GpuPtr dataGpu) {
    auto impl = cmd->device;
    if (dataGpu != 0) {
        VkDeviceAddress addresses = dataGpu;
        impl->m_api.vkCmdPushConstants(cmd->buffer,
                                       impl->m_default_compute_layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT,
                                       0,
                                       sizeof(VkDeviceAddress),
                                       &addresses);
    }
}

void cmd_dispatch(CommandBuffer cmd, GpuPtr dataGpu, const Dimension3D& gridDimensions) {
    auto impl = cmd->device;
    cmd_set_compute_ptr(cmd, dataGpu);
    impl->m_api.vkCmdDispatch(cmd->buffer, gridDimensions.x, gridDimensions.y, gridDimensions.z);
}

void cmd_dispatch_indirect(CommandBuffer cmd, GpuPtr dataGpu, GpuPtr gridDimensionsGpu) {
    auto impl = cmd->device;
    auto dim  = buffer_and_offset_from_ptr(impl, gridDimensionsGpu);
    cmd_set_compute_ptr(cmd, dataGpu);
    impl->m_api.vkCmdDispatchIndirect(cmd->buffer, dim.buffer, dim.offset);
}

void cmd_begin_render_pass(CommandBuffer cmd, RenderPassDesc desc) {
    auto impl = cmd->device;

    Arena                           arena = *get_thread_local_arena(impl);
    Span<VkRenderingAttachmentInfo> color_attachments;

    for (const auto& attachment : desc.color_attachments) {
        color_attachments = concat(&arena, color_attachments, {
            .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext              = nullptr,
            .imageView          = impl->m_texture_pool[attachment.texture].default_image_view,
            .imageLayout        = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .resolveMode        = VK_RESOLVE_MODE_NONE,  // TODO: Multisampling support
            .resolveImageView   = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .loadOp             = bridge(attachment.load_op),
            .storeOp            = bridge(attachment.store_op),
            .clearValue         = {.color = {.uint32 = {attachment.clear_color.r,attachment.clear_color.g,attachment.clear_color.b,attachment.clear_color.a},},},
        });
    }

    // TODO: stencil attachment

    const bool                has_depth_attachment = desc.depth_attachment.texture.h != 0;
    VkRenderingAttachmentInfo depth_attachment{};
    if (has_depth_attachment) {
        depth_attachment = VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = impl->m_texture_pool[desc.depth_attachment.texture].default_image_view,
            .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .loadOp             = bridge(desc.depth_attachment.load_op),
            .storeOp            = bridge(desc.depth_attachment.store_op),
            .clearValue         = {.color = {.uint32 = {desc.depth_attachment.clear_color.r,desc.depth_attachment.clear_color.g,desc.depth_attachment.clear_color.b,desc.depth_attachment.clear_color.a},},},
        };
    }

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
        .pDepthAttachment     = has_depth_attachment ? &depth_attachment : nullptr,
        .pStencilAttachment   = nullptr,
    };

    auto buf = cmd->buffer;
    impl->m_api.vkCmdBeginRendering(buf, &rendering_info);

    // Set default values for dynamic state:
    impl->m_api.vkCmdSetDepthWriteEnable(buf, false);
    impl->m_api.vkCmdSetDepthTestEnable(buf, false);
    impl->m_api.vkCmdSetDepthCompareOp(buf, VK_COMPARE_OP_ALWAYS);
    impl->m_api.vkCmdSetDepthBoundsTestEnable(buf, false);
    impl->m_api.vkCmdSetStencilTestEnable(buf, false);
    impl->m_api.vkCmdSetStencilOp(buf,
                                  VK_STENCIL_FACE_FRONT_AND_BACK,
                                  VK_STENCIL_OP_KEEP,
                                  VK_STENCIL_OP_KEEP,
                                  VK_STENCIL_OP_KEEP,
                                  VK_COMPARE_OP_ALWAYS);

    VkViewport viewport{
        .x        = 0,
        .y        = 0,
        .width    = (float)desc.render_area.width,
        .height   = (float)desc.render_area.height,
        .minDepth = 0,
        .maxDepth = 1.0,
    };
    impl->m_api.vkCmdSetViewportWithCount(buf, 1, &viewport);
    impl->m_api.vkCmdSetScissorWithCount(buf, 1, &render_rect);
}

void cmd_end_render_pass(CommandBuffer cmd) {
    auto impl = cmd->device;
    impl->m_api.vkCmdEndRendering(cmd->buffer);
}

static void cmd_set_graphics_ptrs(CommandBuffer cmd, GpuPtr vertexDataGpu, GpuPtr fragmentDataGpu) {
    auto impl = cmd->device;
    if (vertexDataGpu != 0 || fragmentDataGpu != 0) {
        VkDeviceAddress addresses[] = {vertexDataGpu, fragmentDataGpu};
        impl->m_api.vkCmdPushConstants(cmd->buffer,
                                       impl->m_default_graphics_layout,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0,
                                       2 * sizeof(VkDeviceAddress),
                                       &addresses);
    }
}

void cmd_draw(CommandBuffer cmd,
              GpuPtr        vertexDataGpu,
              GpuPtr        fragmentDataGpu,
              uint32_t      vertexCount,
              uint32_t      instanceCount) {
    auto impl = cmd->device;
    cmd_set_graphics_ptrs(cmd, vertexDataGpu, fragmentDataGpu);

    impl->m_api.vkCmdDraw(cmd->buffer, vertexCount, instanceCount, 0, 0);
}

void cmd_draw_indexed_instanced(CommandBuffer cmd,
                                GpuPtr        vertexDataGpu,
                                GpuPtr        fragmentDataGpu,
                                GpuPtr        indicesGpu,
                                uint32_t      indexCount,
                                uint32_t      instanceCount) {
    auto impl = cmd->device;
    cmd_set_graphics_ptrs(cmd, vertexDataGpu, fragmentDataGpu);
    const auto indices = buffer_and_offset_from_ptr(impl, indicesGpu);
    impl->m_api.vkCmdBindIndexBuffer(cmd->buffer,
                                     indices.buffer,
                                     indices.offset,
                                     VK_INDEX_TYPE_UINT16);
    impl->m_api.vkCmdDrawIndexed(cmd->buffer, indexCount, instanceCount, 0, 0, 0);
}

void cmd_draw_indexed_instanced_indirect(CommandBuffer cmd,
                                         GpuPtr        vertexDataGpu,
                                         GpuPtr        pixelDataGpu,
                                         GpuPtr        indicesGpu,
                                         GpuPtr        argsGpu) {
    auto impl = cmd->device;
    ;
    cmd_set_graphics_ptrs(cmd, vertexDataGpu, pixelDataGpu);
    const auto indices = buffer_and_offset_from_ptr(impl, indicesGpu);
    impl->m_api.vkCmdBindIndexBuffer(cmd->buffer,
                                     indices.buffer,
                                     indices.offset,
                                     VK_INDEX_TYPE_UINT16);

    const auto args = buffer_and_offset_from_ptr(impl, argsGpu);
    impl->m_api.vkCmdDrawIndexedIndirect(cmd->buffer,
                                         args.buffer,
                                         args.offset,
                                         1,
                                         sizeof(VkDrawIndexedIndirectCommand));
}

void cmd_draw_indexed_instanced_indirect_multi(CommandBuffer cmd,
                                               GpuPtr        vertexDataGpu,
                                               GpuPtr        pixelDataGpu,
                                               GpuPtr        indicesGpu,
                                               GpuPtr        argsGpu,
                                               GpuPtr        drawCountGpu,
                                               uint32_t      maxDraws) {
    auto impl = cmd->device;
    cmd_set_graphics_ptrs(cmd, vertexDataGpu, pixelDataGpu);
    const auto indices = buffer_and_offset_from_ptr(impl, indicesGpu);
    impl->m_api.vkCmdBindIndexBuffer(cmd->buffer,
                                     indices.buffer,
                                     indices.offset,
                                     VK_INDEX_TYPE_UINT16);

    const auto args  = buffer_and_offset_from_ptr(impl, argsGpu);
    const auto count = buffer_and_offset_from_ptr(impl, drawCountGpu);
    impl->m_api.vkCmdDrawIndexedIndirectCount(cmd->buffer,
                                              args.buffer,
                                              args.offset,
                                              count.buffer,
                                              count.offset,
                                              maxDraws,
                                              sizeof(VkDrawIndexedIndirectCommand));
}

}  // namespace loon::gpu
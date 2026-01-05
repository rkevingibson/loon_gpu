#include "instance.h"

#include <cstdarg>
#include <cstdio>

#include "objects.h"
#include "utilities.h"
#include "validation.h"
#include "volk.h"
#include "vulkan/vulkan_core.h"

struct FutureParts {
    // Future.id looks like:
    //   60   56   52   48   44   40   36   32   28   24   20   16   12    8    4    0
    // crgg_gggg_gggg_gggg_gggg_gggg_gggg_gggg_gggg_gggg_gggg_gggg_gggg_iiii_iiii_iiii
    // - i is index
    // - g is generation, an atomically increasing number
    // - r is reserved, always set to 1
    // - c is "completed immediately" bit, as a tiny optimization for waitAll.

    uint32_t index;
    uint64_t generation;
    bool     was_completed_immediately;

    static constexpr uint64_t kIndexBits      = 12;
    static constexpr uint64_t kGenerationBits = 64 - kIndexBits - 2;

    static_assert(WGPUInstanceImpl::kMaxFutures == 1 << kIndexBits);
    static constexpr uint64_t kIndexMask               = (1ull << kIndexBits) - 1;
    static constexpr uint64_t kGenerationMask          = (1ull << kGenerationBits) - 1;
    static constexpr uint64_t kCompletedImmediatelyBit = (1ull << 63);
    static constexpr uint64_t kReservedBit             = (1ull << 62);

    static constexpr FutureParts from(WGPUFuture f) {
        return FutureParts{
            .index                     = static_cast<uint32_t>(f.id & kIndexMask),
            .generation                = (f.id >> kIndexBits) & kGenerationMask,
            .was_completed_immediately = (f.id & kCompletedImmediatelyBit) != 0,
        };
    }

    constexpr WGPUFuture to_future() const {
        return WGPUFuture{
            .id = (index & kIndexMask) | generation << kIndexBits
                  | (was_completed_immediately ? kCompletedImmediatelyBit : 0x4000'0000'0000'0000),
        };
    }
};

struct WGPUInstanceImpl::ThreadLocalState {
    webgpu::Allocator       allocator;
    WGPULoonMemoryBlock     arena_memory;
    webgpu::Arena           arena;
    static constexpr size_t kArenaSize = 64ll * 1024;

    ThreadLocalState(const webgpu::Allocator& alloc) :
        allocator{alloc},
        arena_memory{allocator.alloc(kArenaSize)},
        arena{webgpu::Arena(arena_memory.ptr, arena_memory.len)} {}
    ~ThreadLocalState() { allocator.free(arena_memory); }
};

namespace webgpu {
static void fire_callback(const CallbackData& cb) {
    switch (cb.type) {
        case CallbackType::BufferMap: {
            auto fn = reinterpret_cast<WGPUBufferMapCallback>(cb.callback);
            fn(cb.buffer_map.status, cb.message, cb.userdata1, cb.userdata2);
        } break;
        case CallbackType::CompilationInfo: {
            auto fn = reinterpret_cast<WGPUCompilationInfoCallback>(cb.callback);
            fn(cb.compilation_info.status, &cb.compilation_info.info, cb.userdata1, cb.userdata2);
        } break;
        case CallbackType::CreateComputePipeline: {
            auto fn = reinterpret_cast<WGPUCreateComputePipelineAsyncCallback>(cb.callback);
            fn(cb.create_compute_pipeline.status,
               cb.create_compute_pipeline.pipeline,
               cb.message,
               cb.userdata1,
               cb.userdata2);
        } break;
        case CallbackType::CreateRenderPipeline: {
            auto fn = reinterpret_cast<WGPUCreateRenderPipelineAsyncCallback>(cb.callback);
            fn(cb.create_render_pipeline.status,
               cb.create_render_pipeline.pipeline,
               cb.message,
               cb.userdata1,
               cb.userdata2);
        } break;
        case CallbackType::DeviceLost: {
            auto fn = reinterpret_cast<WGPUPopErrorScopeCallback>(cb.callback);
            fn(cb.pop_error_scope.status,
               cb.pop_error_scope.type,
               cb.message,
               cb.userdata1,
               cb.userdata2);
        } break;
        case CallbackType::QueueWorkDone: {
            auto fn = reinterpret_cast<WGPUQueueWorkDoneCallback>(cb.callback);
            fn(cb.queue_work_done.status, cb.message, cb.userdata1, cb.userdata2);
        } break;
        case CallbackType::RequestAdapter: {
            auto fn = reinterpret_cast<WGPURequestAdapterCallback>(cb.callback);
            fn(cb.request_adapter.status,
               cb.request_adapter.adapter,
               cb.message,
               cb.userdata1,
               cb.userdata2);
        } break;
        case CallbackType::RequestDevice: {
            auto fn = reinterpret_cast<WGPURequestDeviceCallback>(cb.callback);
            fn(cb.request_device.status,
               cb.request_device.device,
               cb.message,
               cb.userdata1,
               cb.userdata2);
        } break;
        case CallbackType::PopErrorScope: break;
    }

    // TODO: Free the message for the callback now, if there is one.
}
}  // namespace webgpu

WGPUInstance WGPUInstanceImpl::create(WGPULoonInstanceConfiguration* config) {
    webgpu::Allocator       allocator;
    WGPULoonLogLevel        log_level    = WGPULoonLogLevel_Off;
    WGPULoonProcLogCallback log          = nullptr;
    void*                   log_userdata = nullptr;

    if (config) {
        if (config->alloc) { allocator = webgpu::Allocator(*config); }
        log_level    = config->log_level;
        log          = config->log;
        log_userdata = config->log_userdata;
    }

    VkResult result = volkInitialize();
    if (result != VK_SUCCESS) return nullptr;

    VkInstance instance;

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

    result = vkCreateInstance(&instance_info, nullptr, &instance);
    if (result != VK_SUCCESS) return nullptr;

    volkLoadInstanceOnly(instance);

    auto instance_allocation = allocator.alloc(sizeof(WGPUInstanceImpl));

    auto instance_impl             = ::new (instance_allocation.ptr) WGPUInstanceImpl;
    instance_impl->vk_instance     = instance;
    instance_impl->allocator       = allocator;
    instance_impl->tls_key         = webgpu::tls_alloc([](void* data) {
        auto state = reinterpret_cast<WGPUInstanceImpl::ThreadLocalState*>(data);
        state->~ThreadLocalState();
    });
    instance_impl->log_level       = log_level;
    instance_impl->log_fn          = log;
    instance_impl->log_userdata    = log_userdata;
    instance_impl->allocation_size = instance_allocation.len;
    instance_impl->add_ref();

    return instance_impl;
}

WGPUInstanceImpl::~WGPUInstanceImpl() {
    vkDestroyInstance(vk_instance, nullptr);
    volkFinalize();
    webgpu::tls_free(tls_key);

    const auto size = allocation_size;
    allocator.free({this, allocation_size});
}

void WGPUInstanceImpl::add_ref() {
    refcount.add();
}

void WGPUInstanceImpl::release() {
    if (refcount.release() && can_destroy()) { this->~WGPUInstanceImpl(); }
}

bool WGPUInstanceImpl::can_destroy() {
    return refcount.count() == 0 && webgpu::atomic_load(&adapter_count) == 0
           && webgpu::atomic_load(&surface_count) == 0;
}

static void findAdapterFeatures(WGPUAdapter adapter) {
    // Each wgpu feature corresponds to a vulkan feature in some way.
    // See https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedFeatures for explanations
    // Some will be queried by vkGetPhysicalDeviceFormatProperties
    // TimestampQuery comes from the limits
    // Some just come from physical device features.
    // For now, do a bare minimum of the easy ones, and more advanced ones can come later if at all.
    VkPhysicalDeviceFeatures vk_features{};
    vkGetPhysicalDeviceFeatures(adapter->vk_physical_device, &vk_features);
    const auto add_feature = [&](WGPUFeatureName name) {
        adapter->supported_features[adapter->feature_count++] = name;
    };

    if (vk_features.textureCompressionBC) { add_feature(WGPUFeatureName_TextureCompressionBC); }
    if (vk_features.textureCompressionETC2) { add_feature(WGPUFeatureName_TextureCompressionETC2); }
    if (vk_features.textureCompressionASTC_LDR) {
        add_feature(WGPUFeatureName_TextureCompressionASTC);
    }
    if (vk_features.drawIndirectFirstInstance) {
        add_feature(WGPUFeatureName_IndirectFirstInstance);
    }
    if (vk_features.shaderClipDistance) { add_feature(WGPUFeatureName_ClipDistances); }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(adapter->vk_physical_device, &properties);
    auto& limits = properties.limits;
    if (limits.timestampPeriod != 0 && limits.timestampComputeAndGraphics) {
        add_feature(WGPUFeatureName_TimestampQuery);
    }

    VkFormatProperties format_properties{};
    vkGetPhysicalDeviceFormatProperties(adapter->vk_physical_device,
                                        VK_FORMAT_D32_SFLOAT,
                                        &format_properties);
    if (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        add_feature(WGPUFeatureName_Depth32FloatStencil8);
    }

    vkGetPhysicalDeviceFormatProperties(adapter->vk_physical_device,
                                        VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                                        &format_properties);
    constexpr auto rg11b10_format_usage
        = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
    if ((format_properties.optimalTilingFeatures & rg11b10_format_usage) == rg11b10_format_usage) {
        add_feature(WGPUFeatureName_RG11B10UfloatRenderable);
    }
    // A couple more here I don't care about right now.
}

static void getAdapterInfo(WGPUAdapter adapter) {
    VkPhysicalDeviceVulkan13Properties properties13{};
    properties13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
    properties13.pNext = nullptr;

    VkPhysicalDeviceProperties2 properties2{
        .sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext      = &properties13,
        .properties = {},
    };
    vkGetPhysicalDeviceProperties2(adapter->vk_physical_device, &properties2);

    auto& vk_limits = properties2.properties.limits;
    adapter->limits = {
        .nextInChain           = nullptr,
        .maxTextureDimension1D = vk_limits.maxImageDimension1D,
        .maxTextureDimension2D = vk_limits.maxImageDimension2D,
        .maxTextureDimension3D = vk_limits.maxImageDimension3D,
        .maxTextureArrayLayers = vk_limits.maxImageArrayLayers,
        .maxBindGroups         = vk_limits.maxBoundDescriptorSets,
        .maxBindGroupsPlusVertexBuffers
        = vk_limits.maxVertexInputBindings + webgpu::kMaxVertexBuffers,
        .maxBindingsPerBindGroup = 1000,
        .maxDynamicUniformBuffersPerPipelineLayout
        = vk_limits.maxDescriptorSetUniformBuffersDynamic,
        .maxDynamicStorageBuffersPerPipelineLayout
        = vk_limits.maxDescriptorSetStorageBuffersDynamic,
        .maxSampledTexturesPerShaderStage = vk_limits.maxPerStageDescriptorSampledImages,
        .maxSamplersPerShaderStage        = vk_limits.maxPerStageDescriptorSamplers,
        .maxStorageBuffersPerShaderStage  = vk_limits.maxPerStageDescriptorStorageBuffers,
        .maxStorageTexturesPerShaderStage = vk_limits.maxPerStageDescriptorStorageImages,
        .maxUniformBuffersPerShaderStage  = vk_limits.maxPerStageDescriptorUniformBuffers,
        .maxUniformBufferBindingSize      = vk_limits.maxUniformBufferRange,
        .maxStorageBufferBindingSize      = vk_limits.maxStorageBufferRange,
        .minUniformBufferOffsetAlignment
        = static_cast<uint32_t>(vk_limits.minUniformBufferOffsetAlignment),
        .minStorageBufferOffsetAlignment
        = static_cast<uint32_t>(vk_limits.minUniformBufferOffsetAlignment),
        .maxVertexBuffers                  = webgpu::kMaxVertexBuffers,
        .maxBufferSize                     = properties13.maxBufferSize,
        .maxVertexAttributes               = webgpu::kMaxVertexInputAttributes,
        .maxVertexBufferArrayStride        = vk_limits.maxVertexInputBindingStride,
        .maxInterStageShaderVariables      = vk_limits.maxVertexOutputComponents,
        .maxColorAttachments               = webgpu::kMaxColorAttachments,
        .maxColorAttachmentBytesPerSample  = webgpu::kMaxColorAttachmentBytesPerSample,
        .maxComputeWorkgroupStorageSize    = vk_limits.maxComputeSharedMemorySize,
        .maxComputeInvocationsPerWorkgroup = vk_limits.maxComputeWorkGroupInvocations,
        .maxComputeWorkgroupSizeX          = vk_limits.maxComputeWorkGroupSize[0],
        .maxComputeWorkgroupSizeY          = vk_limits.maxComputeWorkGroupSize[1],
        .maxComputeWorkgroupSizeZ          = vk_limits.maxComputeWorkGroupSize[2],
        .maxComputeWorkgroupsPerDimension  = vk_limits.maxComputeWorkGroupCount[0],
    };

    auto& props = properties2.properties;

    memcpy(adapter->device_name, props.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE);
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            adapter->adapter_type = WGPUAdapterType_IntegratedGPU;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            adapter->adapter_type = WGPUAdapterType_DiscreteGPU;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: adapter->adapter_type = WGPUAdapterType_CPU; break;
        default: adapter->adapter_type = WGPUAdapterType_Unknown; break;
    }
    adapter->vendor_id         = props.vendorID;
    adapter->device_id         = props.deviceID;
    adapter->subgroup_min_size = properties13.minSubgroupSize;
    adapter->subgroup_max_size = properties13.maxSubgroupSize;
}

WGPUFuture WGPUInstanceImpl::request_adapter(WGPU_NULLABLE WGPURequestAdapterOptions const* options,
                                             WGPURequestAdapterCallbackInfo callbackInfo) {
    // Need to pick a vulkan physical device, based on the options provided.
    // If options are not provided, default to picking a higher-end option.
    // Otherwise:
    // - If forceFallbackAdapter is set, fail - we don't support it.
    // - If powerPreference is lowPower, try to use an integrated GPU
    // - If backendType is set, it must be vulkan or we return null.
    // - Check if GPU can output to the given surface, otherwise fail.

    if (options->forceFallbackAdapter) { return WGPU_FUTURE_INIT; }
    if (options->backendType != WGPUBackendType_Undefined
        && options->backendType != WGPUBackendType_Vulkan) {
        return WGPU_FUTURE_INIT;
    }

    const auto device_info = selectPhysicalDevice(options);
    if (device_info.device == VK_NULL_HANDLE) { return WGPU_FUTURE_INIT; }

    auto alloc = allocator.alloc(sizeof(WGPUAdapterImpl));
    // TODO: Check for allocation failure
    auto adapter                = ::new (alloc.ptr) WGPUAdapterImpl;
    adapter->instance           = this;
    adapter->vk_physical_device = device_info.device;
    adapter->queue_family       = device_info.queue_family;
    adapter->add_ref();

    webgpu::atomic_fetch_add(&adapter_count, 1);

    findAdapterFeatures(adapter);
    getAdapterInfo(adapter);

    webgpu::CallbackData* cb;
    WGPUFuture            future = create_future(&cb);
    *cb                  = {
                         .callback  = reinterpret_cast<WGPUProc>(callbackInfo.callback),
                         .mode      = callbackInfo.mode,
                         .userdata1 = callbackInfo.userdata1,
                         .userdata2 = callbackInfo.userdata2,
                         .message   = WGPU_STRING_VIEW_INIT,
                         .type      = webgpu::CallbackType::RequestAdapter,
                         .request_adapter = {
                             .status  = WGPURequestAdapterStatus_Success,
                             .adapter = adapter,
        },
    };
    set_future_ready(future);

    return future;
}

void WGPUInstanceImpl::free_adapter(WGPUAdapter adapter) {
    adapter->~WGPUAdapterImpl();
    allocator.free({.ptr = adapter, .len = sizeof(WGPUAdapterImpl)});
    webgpu::atomic_fetch_add(&adapter_count, -1);
    if (can_destroy()) { this->~WGPUInstanceImpl(); }
}

WGPUSurface WGPUInstanceImpl::create_surface(WGPUSurfaceDescriptor const* descriptor) {
    VkSurfaceKHR       surface = VK_NULL_HANDLE;
    WGPUChainedStruct* chain   = descriptor->nextInChain;
    while (chain != nullptr) {
        switch (chain->sType) {
#if defined(VK_USE_PLATFORM_WIN32_KHR)
            case WGPUSType_SurfaceSourceWindowsHWND: {
                auto surface_source_windows_hwnd
                    = reinterpret_cast<WGPUSurfaceSourceWindowsHWND const*>(chain);

                const VkWin32SurfaceCreateInfoKHR surface_info = {
                    .sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                    .pNext     = nullptr,
                    .flags     = 0,
                    .hinstance = (HINSTANCE)surface_source_windows_hwnd->hinstance,
                    .hwnd      = (HWND)surface_source_windows_hwnd->hwnd,
                };
                VkResult result
                    = vkCreateWin32SurfaceKHR(get_vk_instance(), &surface_info, nullptr, &surface);

                if (result != VK_SUCCESS) return nullptr;
            } break;
#endif

#if defined(VK_USE_PLATFORM_XLIB_KHR)
            case WGPUSType_SurfaceSourceXlibWindow: {
                auto surface_source_xlib
                    = reinterpret_cast<WGPUSurfaceSourceXlibWindow const*>(chain);

                const VkXlibSurfaceCreateInfoKHR surface_info{
                    .sType  = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                    .pNext  = nullptr,
                    .flags  = 0,
                    .dpy    = reinterpret_cast<Display*>(surface_source_xlib->display),
                    .window = surface_source_xlib->window,
                };
                VkResult result
                    = vkCreateXlibSurfaceKHR(get_vk_instance(), &surface_info, nullptr, &surface);

                if (result != VK_SUCCESS) return nullptr;
            } break;
#endif

#if defined(VK_USE_PLATFORM_METAL_EXT)
            case WGPUSType_SurfaceSourceMetalLayer: {
                auto surface_source_mtl
                    = reinterpret_cast<WGPUSurfaceSourceMetalLayer const*>(chain);

                const VkMetalSurfaceCreateInfoEXT surface_info{
                    .sType  = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
                    .pNext  = nullptr,
                    .flags  = 0,
                    .pLayer = reinterpret_cast<CAMetalLayer*>(surface_source_mtl->layer),
                };
                VkResult result
                    = vkCreateMetalSurfaceEXT(get_vk_instance(), &surface_info, nullptr, &surface);
                if (result != VK_SUCCESS) return nullptr;
            } break;
#endif
            default: log(WGPULoonLogLevel_Error, "Unsupported surface source"); return nullptr;
        }
        chain = chain->next;
    }

    auto surface_allocation = allocator.alloc(sizeof(WGPUSurfaceImpl));
    if (surface_allocation.ptr == nullptr) { return nullptr; }

    auto impl        = ::new (surface_allocation.ptr) WGPUSurfaceImpl;
    impl->instance   = this;
    impl->vk_surface = surface;
    impl->label.set(allocator, descriptor->label);
    impl->add_ref();
    webgpu::atomic_fetch_add(&surface_count, 1);
    return impl;
}

void WGPUInstanceImpl::free_surface(WGPUSurface surface) {
    surface->~WGPUSurfaceImpl();
    allocator.free({.ptr = surface, .len = sizeof(WGPUSurfaceImpl)});
    webgpu::atomic_fetch_add(&surface_count, -1);
    if (can_destroy()) { this->~WGPUInstanceImpl(); }
}

void WGPUInstanceImpl::log(WGPULoonLogLevel lvl, const char* fmt, ...) {
    if (lvl > log_level) { return; }

    auto    arena = get_thread_local_arena();
    va_list args1, args2;
    va_start(args1, fmt);
    va_copy(args2, args1);
    int  msg_len   = vsnprintf(nullptr, 0, fmt, args1);
    auto msg_block = arena->alloc(msg_len + 1);
    vsnprintf(reinterpret_cast<char*>(msg_block), msg_len + 1, fmt, args2);
    // We null-terminate the string despite passing it to the log function as a string view, for
    // convenience & safety in the logging function - they won't need to null terminate it if using
    // an API that expects null terminated strings.
    ((char*)msg_block)[msg_len] = '\0';
    WGPUStringView msg{.data = reinterpret_cast<const char*>(msg_block), .length = (size_t)msg_len};
    log_fn(lvl, msg, log_userdata);
    arena->free(msg_block, msg_len + 1);
    va_end(args1);
    va_end(args2);
}

WGPUFuture WGPUInstanceImpl::create_future(webgpu::CallbackData** cbOut) {
    const auto future_idx = futures_pool.get();
    if (future_idx == futures_pool.INVALID_INDEX) { return WGPU_FUTURE_INIT; }
    if (cbOut) { *cbOut = &futures_pool[future_idx]; }

    // Future id is constructed from index and generation count. When a future is fired,
    auto generation = futures_generation.fetch_add(1);
    return FutureParts{
        .index                     = future_idx,
        .generation                = generation,
        .was_completed_immediately = false,
    }
        .to_future();
}

void WGPUInstanceImpl::complete_future(uint32_t index) {
    auto& callback = futures_pool[index];
    webgpu::fire_callback(callback);
    futures_pool.release(index);
}

void WGPUInstanceImpl::set_future_ready(WGPUFuture f) {
    const auto [index, generation, was_completed_immediately] = FutureParts::from(f);
    auto& callback                                            = futures_pool[index];
    if (callback.mode == WGPUCallbackMode_AllowSpontaneous) {
        complete_future(index);
    } else {
        // Add to a list of completed futures for calling in process events/wait.
        std::lock_guard<std::mutex> lock(ready_futures_mutex);
        callback.ready                       = true;
        ready_futures[ready_futures_count++] = f;
    }
}

void WGPUInstanceImpl::process_ready_futures() {
    std::lock_guard<std::mutex> lock(ready_futures_mutex);
    uint32_t                    futures_count = ready_futures_count;
    uint32_t                    i             = 0;
    while (i != futures_count) {
        const auto future   = FutureParts::from(ready_futures[i]);
        auto&      callback = futures_pool[future.index];
        assert(callback.ready);
        if (callback.mode == WGPUCallbackMode_AllowProcessEvents) {
            complete_future(future.index);
            ready_futures[future.index] = ready_futures[--futures_count];
        } else {
            ++i;
        }
    }
}

WGPUWaitStatus WGPUInstanceImpl::wait_on_futures(size_t              future_count,
                                                 WGPUFutureWaitInfo* futures,
                                                 uint64_t            timeoutNS) {
    // We're not truly asynchronous, so I don't think we're going to handle any of these properly in
    // terms of timeouts.
    (void)timeoutNS;
    WGPUWaitStatus result = WGPUWaitStatus_TimedOut;

    std::lock_guard<std::mutex> lock(ready_futures_mutex);
    for (size_t i = 0; i < future_count; ++i) {
        auto&      f      = futures[i];
        const auto future = FutureParts::from(f.future);

        if (future.was_completed_immediately) {
            f.completed = true;
            continue;
        }

        // Need to check if this future is in the completed futures list before getting it from the
        // pool - if it's not. Ideally, we have a fast way to get it's spot in the list rather than
        // just a linear search
        uint32_t completed_future_list_idx = ready_futures_count;
        for (uint32_t completed_future_idx = 0; completed_future_idx < ready_futures_count;
             ++completed_future_idx) {
            if (ready_futures[completed_future_idx].id == f.future.id) {
                completed_future_list_idx = completed_future_idx;
            }
        }

        if (completed_future_list_idx == ready_futures_count) {
            // Future was not in completed futures list. Since in our implementation we always ready
            // callbacks right away, we can assume this one has already fired/been waited on.
            result      = WGPUWaitStatus_Success;
            f.completed = true;
            continue;
        }

        auto& callback = futures_pool[future.index];
        f.completed    = callback.ready;
        if (callback.ready) {
            webgpu::fire_callback(callback);
            futures_pool.release(future.index);
            result = WGPUWaitStatus_Success;
            // Swap with the end to remove
            ready_futures[completed_future_list_idx] = ready_futures[--ready_futures_count];
        }
    }

    return result;
}

WGPUInstanceImpl::ThreadLocalState* WGPUInstanceImpl::get_thread_local_state() {
    auto state = reinterpret_cast<ThreadLocalState*>(webgpu::tls_get_data(tls_key));
    if (state == nullptr) {
        state = new ThreadLocalState(allocator);
        webgpu::tls_set_data(tls_key, state);
    }
    return state;
}

webgpu::Arena* WGPUInstanceImpl::get_thread_local_arena() {
    return &get_thread_local_state()->arena;
}

WGPUInstanceImpl::PhysicalDeviceInfo WGPUInstanceImpl::selectPhysicalDevice(
    WGPURequestAdapterOptions const* options) {
    constexpr uint32_t max_physical_devices = 8;
    VkPhysicalDevice   physical_devices[max_physical_devices];
    uint32_t           device_count = max_physical_devices;
    VkResult           vkresult
        = vkEnumeratePhysicalDevices(get_vk_instance(), &device_count, physical_devices);

    if (vkresult == VK_INCOMPLETE) {
        log(WGPULoonLogLevel_Warning,
            "Too many vulkan physical devices returned some will be ignored");
    }
    if (vkresult < 0) { return PhysicalDeviceInfo{}; }

    const bool prefer_integrated = options->powerPreference == WGPUPowerPreference_LowPower;
    const bool prefer_dedicated  = options->powerPreference == WGPUPowerPreference_HighPerformance;

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
            log(WGPULoonLogLevel_Warning,
                "Too many queue families on physical device, some may be missed");
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
            if (options->compatibleSurface) {
                const VkSurfaceKHR surface           = options->compatibleSurface->vk_surface;
                VkBool32           surface_supported = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(physical_devices[device_idx],
                                                     queue_family,
                                                     surface,
                                                     &surface_supported);

                is_valid = is_valid && surface_supported;
            }

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
            = allocator.alloc(sizeof(VkExtensionProperties) * extension_count);
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
        for (size_t required_ext_idx = 0; required_ext_idx < webgpu::kRequiredDeviceExtensionsCount;
             ++required_ext_idx) {
            const char* required_extension = webgpu::kRequiredDeviceExtensions[required_ext_idx];
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
        allocator.free(extension_properties_block);

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
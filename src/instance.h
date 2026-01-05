#pragma once

#include <cstddef>

#include "platform_utils.h"
#include "utilities.h"
#include "volk.h"
#include "webgpu/webgpu_loon.h"

struct WGPUInstanceImpl {
   public:
    static WGPUInstance create(WGPULoonInstanceConfiguration* config);
    void                add_ref();
    void                release();

    ~WGPUInstanceImpl();

    // Child objects:
    WGPUFuture request_adapter(WGPU_NULLABLE WGPURequestAdapterOptions const* options,
                               WGPURequestAdapterCallbackInfo                 callbackInfo);
    void       free_adapter(WGPUAdapter);

    WGPUSurface create_surface(WGPUSurfaceDescriptor const* descriptor);
    void        free_surface(WGPUSurface);

    void log(WGPULoonLogLevel lvl, const char* fmt, ...);
    // Managing futures:

    static constexpr size_t kMaxFutures = 1 << 12;
    WGPUFuture              create_future(webgpu::CallbackData** cbOut);
    void                    set_future_ready(WGPUFuture f);
    void                    process_ready_futures();
    WGPUWaitStatus          wait_on_futures(size_t              future_count,
                                            WGPUFutureWaitInfo* futures,
                                            uint64_t            timeoutNS);

    VkInstance               get_vk_instance() const { return vk_instance; };
    const webgpu::Allocator& get_allocator() const { return allocator; }
    webgpu::Arena*           get_thread_local_arena();

   private:
    bool can_destroy();

    struct PhysicalDeviceInfo {
        VkPhysicalDevice device       = VK_NULL_HANDLE;
        uint32_t         queue_family = 0;
    };

    PhysicalDeviceInfo selectPhysicalDevice(WGPURequestAdapterOptions const* options);

    webgpu::ReferenceCount refcount;
    // For created surfaces/adapters - we could store a vector of pointers, but really we just need
    // the count. They should be atomically manipulated
    int64_t adapter_count = 0;
    int64_t surface_count = 0;

    void complete_future(uint32_t index);
    struct ThreadLocalState;
    ThreadLocalState* get_thread_local_state();

    VkInstance vk_instance = VK_NULL_HANDLE;

    webgpu::ObjectPool<webgpu::CallbackData, kMaxFutures>
                          futures_pool;  // TODO: Can probably replace this with objectList
    std::atomic<uint64_t> futures_generation = {0};
    std::mutex            ready_futures_mutex;
    uint32_t              ready_futures_count = 0;
    WGPUFuture            ready_futures[kMaxFutures];
    webgpu::Allocator     allocator;

    webgpu::tls_key tls_key = 0;

    WGPULoonLogLevel        log_level    = WGPULoonLogLevel_Off;
    WGPULoonProcLogCallback log_fn       = nullptr;
    void*                   log_userdata = nullptr;

    uint32_t allocation_size
        = 0;  // The size of the allocation returned during the creation of the instance.
};
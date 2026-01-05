#pragma once

#include <cstdint>

#include "objects.h"
#include "volk.h"
#include "webgpu/webgpu.h"


struct WGPUQueueImpl : WGPUObjectBase {
   public:
    bool initialize(WGPUDevice device, uint32_t queue_family);
    void reset();

    void       submit(size_t command_count, WGPUCommandBuffer const* commands);
    WGPUFuture on_submitted_work_done(WGPUQueueWorkDoneCallbackInfo callback_info);
    void write_buffer(WGPUBuffer buffer, uint64_t buffer_offset, void const* data, size_t size);
    void write_texture(WGPUTexelCopyTextureInfo const*  destination,
                       void const*                      data,
                       size_t                           data_size,
                       WGPUTexelCopyBufferLayout const* data_layout,
                       WGPUExtent3D const*              write_size);

    // Helpers for WGPUSurfaceImpl

    // Waits until the current swapchain image is available to be acquired, and returns a semaphore
    // that should be signalled when the image is acquired.
    VkSemaphore wait_for_current_swapchain_image();
    WGPUStatus  present(WGPUTexture texture, VkSwapchainKHR swapchain, uint32_t image_index);

    uint64_t get_current_timeline_value() const;

   private:
    static constexpr uint32_t kMaxFramesInFlight     = 3;
    static constexpr uint32_t kMaxPresentSubmissions = 8;

    class SemaphorePool {
       public:
        SemaphorePool() = default;
        SemaphorePool(WGPUDevice device);
        ~SemaphorePool();
        SemaphorePool(SemaphorePool&& other) : SemaphorePool() { swap(*this, other); };
        SemaphorePool& operator=(SemaphorePool&& other) {
            swap(*this, other);
            return *this;
        };


        VkSemaphore        get();
        const VkSemaphore* data() const noexcept { return semaphores; }
        uint32_t           size() const noexcept { return count; }
        void               reset();
        void               clear() noexcept { count = 0; }
        friend void        swap(SemaphorePool& a, SemaphorePool& b);

       private:
        WGPUDevice   device;
        VkSemaphore* semaphores = nullptr;
        uint32_t     count      = 0;
        uint32_t     capacity   = 0;
    };
    friend void swap(SemaphorePool& a, SemaphorePool& b);

    struct SwapchainSynchronization {
        // The value of the timeline_semaphore that we need to wait for before we can safely acquire
        // this image again.
        uint64_t last_swapchain_submission_timeline_value = 0;

        // A semaphore that is signaled when this image is safe to use.
        //
        VkSemaphore acquire = VK_NULL_HANDLE;

        // Signals whether a call to wgpuQueueSubmit that touches the swapchain should wait for the
        // above semaphore. Because we strongly order all queue submissions, only the first one
        // needs to wait.
        bool should_wait_for_acquire = false;

        // A pool of semaphores that are signaled by any queue submission that touches this
        // swapchain image. All entries of the pool will be waited on during presentation.
        SemaphorePool semaphore_pool;

        VkCommandBuffer command_buffer;
    };

    VkQueue                  m_vk_queue     = VK_NULL_HANDLE;
    uint32_t                 m_queue_family = 0;
    VkCommandPool            m_swapchain_command_pool;
    int64_t                  m_frame_idx = 0;
    SwapchainSynchronization m_swapchain_synchronization[kMaxFramesInFlight];

    // A single global timeline semaphore used to enforce ordering between all submitted
    // CommandBuffers. Basically, each submitted CommandBuffer will wait for the previous value of
    // the timeline_semaphore, and increment it when it's done. This will be inefficient for
    // overlapping work, but is dead simple as a first pass implementation. Eventually, we could do
    // resource tracking to determine the lowest safe value to wait on before executing the
    // CommandBuffer, and get better work overlap if multiple CommandBuffers don't overlap
    // resources. Since timeline semaphores also support `vkWaitSemaphore`, this can also be used
    // for implementing onSubmittedWorkDone without needing VkFence's.
    VkSemaphore m_timeline_semaphore = VK_NULL_HANDLE;
    int64_t     m_timeline_value     = 0;  // Atomic
};

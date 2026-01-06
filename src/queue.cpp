#include "queue.h"

#include "commands.h"
#include "device.h"
#include "wgpu_to_vk.h"


// MARK: SemaphorePool

void swap(WGPUQueueImpl::SemaphorePool& a, WGPUQueueImpl::SemaphorePool& b) {
    std::swap(a.device, b.device);
    std::swap(a.semaphores, b.semaphores);
    std::swap(a.count, b.count);
    std::swap(a.capacity, b.capacity);
}

WGPUQueueImpl::SemaphorePool::SemaphorePool(WGPUDevice device) : device(device) {}

WGPUQueueImpl::SemaphorePool::~SemaphorePool() {}

void WGPUQueueImpl::SemaphorePool::reset() {
    for (uint32_t i = 0; i < capacity; ++i) {
        device->vk_api.vkDestroySemaphore(device->vk_device, semaphores[i], nullptr);
    }
    if (capacity) {
        device->get_allocator().free(
            {(void*)semaphores, static_cast<uint32_t>(capacity * sizeof(VkSemaphore))});
    }
    capacity = 0;
    count    = 0;
}

VkSemaphore WGPUQueueImpl::SemaphorePool::get() {
    auto& alloc  = device->get_allocator();
    auto& vk_api = device->vk_api;
    if (count == capacity) {
        // Need to reallocate
        const uint32_t new_capacity = (3 * capacity >> 1) > 4 ? (3 * capacity >> 1) : 4;

        auto new_block = alloc.alloc(sizeof(VkSemaphore) * new_capacity);

        // TODO: Potential error here - if new_block.size() % sizeof(VkSemaphore) != 0, we're losing
        // track of bytes and the allocator may complain.
        const uint32_t actual_capacity = new_block.len / sizeof(VkSemaphore);
        VkSemaphore*   new_array       = reinterpret_cast<VkSemaphore*>(new_block.ptr);

        for (uint32_t i = 0; i < capacity; ++i) { new_array[i] = semaphores[i]; }

        VkSemaphoreCreateInfo semaphore_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        };
        for (uint32_t i = capacity; i < actual_capacity; ++i) {
            // TODO: Error handling here. Not much that I can do if it fails tho - pretty much just
            // means out of memory.
            WGPU_VK_CHECK(
                vkCreateSemaphore(device->vk_device, &semaphore_info, nullptr, &new_array[i]));
        }

        if (semaphores) {
            alloc.free({(void*)semaphores, static_cast<uint32_t>(capacity * sizeof(VkSemaphore))});
        }
        semaphores = new_array;
        capacity   = actual_capacity;
    }

    return semaphores[count++];
}


// MARK: Queue

bool WGPUQueueImpl::initialize(WGPUDevice device, uint32_t queue_family) {
    VkQueue vk_queue = VK_NULL_HANDLE;
    auto&   vk_api   = device->vk_api;

    vk_api.vkGetDeviceQueue(device->vk_device, queue_family, 0, &vk_queue);

    // Create the timeline sempahore for queue ordering synchronization
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

    VkSemaphore timeline_semaphore = VK_NULL_HANDLE;
    WGPU_VK_CHECK(
        vkCreateSemaphore(device->vk_device, &timeline_create_info, nullptr, &timeline_semaphore));

    VkSemaphore acquire_semaphores[WGPUQueueImpl::kMaxFramesInFlight];
    for (uint32_t i = 0; i < WGPUQueueImpl::kMaxFramesInFlight; ++i) {
        VkSemaphore           semaphore;
        VkSemaphoreCreateInfo create_info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                          .pNext = 0,
                                          .flags = 0};
        WGPU_VK_CHECK(vkCreateSemaphore(device->vk_device, &create_info, nullptr, &semaphore));
        acquire_semaphores[i] = semaphore;
    }

    VkCommandPoolCreateInfo command_pool_info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family,
    };

    VkCommandPool command_pool;
    WGPU_VK_CHECK(
        vkCreateCommandPool(device->vk_device, &command_pool_info, nullptr, &command_pool));

    VkCommandBuffer             command_buffers[WGPUQueueImpl::kMaxFramesInFlight];
    VkCommandBufferAllocateInfo command_alloc_info{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = WGPUQueueImpl::kMaxFramesInFlight,
    };
    WGPU_VK_CHECK(
        vkAllocateCommandBuffers(device->vk_device, &command_alloc_info, command_buffers));


    this->device             = device;
    this->label              = {};
    m_vk_queue               = vk_queue;
    m_swapchain_command_pool = command_pool;
    m_frame_idx              = 0;
    m_timeline_semaphore     = timeline_semaphore;
    m_timeline_value         = 0;

    for (uint32_t i = 0; i < WGPUQueueImpl::kMaxFramesInFlight; ++i) {
        m_swapchain_synchronization[i]                = {};
        m_swapchain_synchronization[i].semaphore_pool = WGPUQueueImpl::SemaphorePool(device);
        m_swapchain_synchronization[i].acquire        = acquire_semaphores[i];
        m_swapchain_synchronization[i].command_buffer = command_buffers[i];
    }

    return true;
}

void WGPUQueueImpl::reset() {
    // Free the queue resources
    if (m_vk_queue) {
        device->vk_api.vkDestroyCommandPool(device->vk_device, m_swapchain_command_pool, nullptr);
        for (uint32_t i = 0; i < WGPUQueueImpl::kMaxFramesInFlight; ++i) {
            device->vk_api.vkDestroySemaphore(device->vk_device,
                                              m_swapchain_synchronization[i].acquire,
                                              nullptr);
            m_swapchain_synchronization[i].semaphore_pool.reset();
        }
        device->vk_api.vkDestroySemaphore(device->vk_device, m_timeline_semaphore, nullptr);
        m_vk_queue = VK_NULL_HANDLE;
    }
}

void WGPUQueueImpl::submit(size_t command_count, WGPUCommandBuffer const* commands) {
    auto arena       = device->get_thread_local_arena();
    auto cmd_buffers = reinterpret_cast<VkCommandBuffer*>(
        arena->alloc(sizeof(VkCommandBuffer) * 2 * command_count));

    bool     touches_surface_image        = false;
    auto&    vk_api                       = device->vk_api;
    int64_t  on_submission_timeline_value = ++m_timeline_value;
    uint32_t submitted_cmds_count         = 0;
    for (size_t cmd_idx = 0; cmd_idx < command_count; ++cmd_idx) {
        auto synchronization_cmd_buffer = commands[cmd_idx]->vk_cmd_buffers[0];

        bool synchronization_needed
            = loon::gpu::record_pre_submit_synchronization_cmd(synchronization_cmd_buffer,
                                                               device,
                                                               commands[cmd_idx]->first_usages);

        commands[cmd_idx]->last_usages.update_resource_last_usages();
        commands[cmd_idx]->submitted_timeline_value = on_submission_timeline_value;
        if (synchronization_needed) {
            cmd_buffers[submitted_cmds_count++] = synchronization_cmd_buffer;
        }
        cmd_buffers[submitted_cmds_count++] = commands[cmd_idx]->vk_cmd_buffers[1];
        if (commands[cmd_idx]->touches_surface_image) { touches_surface_image = true; }
    }

    // We have at most 2 semaphores to signal. Semaphore 0 is the timeline semaphore, which is
    // used for wgpuQueueOnSubmittedWorkDone. Semaphore 1 is used to mark when we're done with
    // the swapchain image.
    VkSemaphore signal_semaphores[2];
    uint64_t    signal_semaphore_values[2];
    signal_semaphores[0]       = m_timeline_semaphore;
    signal_semaphore_values[0] = on_submission_timeline_value;

    if (touches_surface_image) {
        signal_semaphores[1]       = m_swapchain_synchronization[m_frame_idx].semaphore_pool.get();
        signal_semaphore_values[1] = 0;  // Not a timeline semaphore, doesn't matter.
        m_swapchain_synchronization[m_frame_idx].last_swapchain_submission_timeline_value
            = on_submission_timeline_value;
    }

    VkTimelineSemaphoreSubmitInfo semaphore_info{
        .sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext                     = nullptr,
        .waitSemaphoreValueCount   = 0,
        .pWaitSemaphoreValues      = nullptr,
        .signalSemaphoreValueCount = touches_surface_image ? 2u : 1u,
        .pSignalSemaphoreValues    = signal_semaphore_values,
    };

    VkSubmitInfo submit_info = VkSubmitInfo{
        .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext                = &semaphore_info,
        .waitSemaphoreCount   = 0,
        .pWaitSemaphores      = nullptr,
        .pWaitDstStageMask    = nullptr,
        .commandBufferCount   = submitted_cmds_count,
        .pCommandBuffers      = cmd_buffers,
        .signalSemaphoreCount = touches_surface_image ? 2u : 1u,
        .pSignalSemaphores    = signal_semaphores,
    };

    VkPipelineStageFlags surface_wait_stage_flags = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    if (m_swapchain_synchronization[m_frame_idx].should_wait_for_acquire && touches_surface_image) {
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores    = &m_swapchain_synchronization[m_frame_idx].acquire;
        submit_info.pWaitDstStageMask  = &surface_wait_stage_flags;
        m_swapchain_synchronization[m_frame_idx].should_wait_for_acquire = false;
    }
    WGPU_VK_CHECK(vkQueueSubmit(m_vk_queue, 1, &submit_info, VK_NULL_HANDLE));

    arena->free((void*)cmd_buffers, sizeof(VkCommandBuffer) * 2 * command_count);
}

WGPUFuture WGPUQueueImpl::on_submitted_work_done(WGPUQueueWorkDoneCallbackInfo callback_info) {
    // TODO: IMPL
    return WGPU_FUTURE_INIT;
}

void WGPUQueueImpl::write_buffer(WGPUBuffer  buffer,
                                 uint64_t    buffer_offset,
                                 void const* data,
                                 size_t      size) {
    // TODO: IMPL
}
void WGPUQueueImpl::write_texture(WGPUTexelCopyTextureInfo const*  destination,
                                  void const*                      data,
                                  size_t                           data_size,
                                  WGPUTexelCopyBufferLayout const* data_layout,
                                  WGPUExtent3D const*              write_size) {
    // TODO: IMPL
}

VkSemaphore WGPUQueueImpl::wait_for_current_swapchain_image() {
    auto&          synchronization      = m_swapchain_synchronization[m_frame_idx];
    const uint64_t semaphore_wait_value = synchronization.last_swapchain_submission_timeline_value;
    VkSemaphoreWaitInfo wait_info{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext          = nullptr,
        .flags          = 0,
        .semaphoreCount = 1,
        .pSemaphores    = &m_timeline_semaphore,
        .pValues        = &semaphore_wait_value,
    };
    auto& vk_api = device->vk_api;
    WGPU_VK_CHECK(vkWaitSemaphores(device->vk_device, &wait_info, UINT64_MAX));
    synchronization.should_wait_for_acquire = true;
    return synchronization.acquire;
}

WGPUStatus WGPUQueueImpl::present(WGPUTexture    texture,
                                  VkSwapchainKHR swapchain,
                                  uint32_t       image_index) {
    auto& vk_api          = device->vk_api;
    auto& synchronization = m_swapchain_synchronization[m_frame_idx];

    // Record a small command buffer with just an image barrier to transition the layout to
    // PRESENT.
    {
        VkCommandBuffer cmd_buffer = synchronization.command_buffer;
        WGPU_VK_CHECK(vkResetCommandBuffer(cmd_buffer, 0));

        VkCommandBufferBeginInfo begin_info{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext            = nullptr,
            .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        WGPU_VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));

        VkImageMemoryBarrier2 image_barrier{
            .sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext         = nullptr,
            .srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
            .oldLayout = loon::gpu::image_layout_from_usage(texture->last_submitted_usage),
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = m_queue_family,
            .dstQueueFamilyIndex = m_queue_family,
            .image = texture->vk_image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        texture->last_submitted_usage = loon::gpu::kUsagePresent;
        VkDependencyInfo dependency_info{
            .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext                    = nullptr,
            .dependencyFlags          = 0,
            .memoryBarrierCount       = 0,
            .pMemoryBarriers          = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers    = nullptr,
            .imageMemoryBarrierCount  = 1,
            .pImageMemoryBarriers     = &image_barrier,
        };
        vk_api.vkCmdPipelineBarrier2(cmd_buffer, &dependency_info);
        WGPU_VK_CHECK(vkEndCommandBuffer(cmd_buffer));

        VkSemaphore semaphore = synchronization.semaphore_pool.get();

        VkSemaphore signal_semaphores[2]       = {semaphore, m_timeline_semaphore};
        uint64_t    signal_semaphore_values[2] = {0, static_cast<uint64_t>(++m_timeline_value)};

        VkTimelineSemaphoreSubmitInfo semaphore_info{
            .sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext                     = nullptr,
            .waitSemaphoreValueCount   = 0,
            .pWaitSemaphoreValues      = nullptr,
            .signalSemaphoreValueCount = 2u,
            .pSignalSemaphoreValues    = signal_semaphore_values,
        };

        VkSubmitInfo submit_info{
            .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext                = &semaphore_info,
            .waitSemaphoreCount   = 0,
            .pWaitSemaphores      = nullptr,
            .pWaitDstStageMask    = nullptr,
            .commandBufferCount   = 1,
            .pCommandBuffers      = &cmd_buffer,
            .signalSemaphoreCount = 2,
            .pSignalSemaphores    = signal_semaphores,
        };
        synchronization.last_swapchain_submission_timeline_value = signal_semaphore_values[1];
        WGPU_VK_CHECK(vkQueueSubmit(m_vk_queue, 1, &submit_info, VK_NULL_HANDLE));
    }

    VkPresentInfoKHR present_info{
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext              = nullptr,
        .waitSemaphoreCount = synchronization.semaphore_pool.size(),
        .pWaitSemaphores    = synchronization.semaphore_pool.data(),
        .swapchainCount     = 1,
        .pSwapchains        = &swapchain,
        .pImageIndices      = &image_index,
        .pResults           = nullptr,
    };
    VkResult present_result = vk_api.vkQueuePresentKHR(m_vk_queue, &present_info);

    synchronization.semaphore_pool.clear();
    m_frame_idx = (m_frame_idx + 1) % WGPUQueueImpl::kMaxFramesInFlight;

    if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR
        || present_result == VK_SUCCESS) {
        return WGPUStatus_Success;
    } else {
        return WGPUStatus_Error;
    }
}

uint64_t WGPUQueueImpl::get_current_timeline_value() const {
    uint64_t timeline_value = 0;
    auto&    vk_api         = device->vk_api;
    WGPU_VK_CHECK(
        vkGetSemaphoreCounterValue(device->vk_device, m_timeline_semaphore, &timeline_value));
    return timeline_value;
}
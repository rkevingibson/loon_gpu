#include "gpu/loon_gpu.h"

#include "containers.h"
#include "vma_usage.h"
#include "volk.h"
#include "vulkan/vulkan_core.h"

namespace loon::gpu {

struct Buffer {
    VkBuffer      vk_buffer;
    VmaAllocation vk_allocation;
};

struct Semaphore {
    VkSemaphore s;
};

struct Device::Impl {
    VkDevice        device;
    VolkDeviceTable vk_api;
    VmaAllocator    vk_allocator = VK_NULL_HANDLE;

    ObjectPool<Buffer, kMaxNumBuffers> buffer_pool;
};

Handle<Buffer> Device::malloc(size_t bytes, MEMORY memory) {
    constexpr size_t kDeviceSize = sizeof(Device::Impl);

    return malloc(bytes, 64, memory);
}

Handle<Buffer> Device::malloc(size_t bytes, size_t align, MEMORY memory) {
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
    chk(vmaCreateBuffer(impl->vk_allocator,
                        &create_info,
                        &alloc_info,
                        &vk_buffer,
                        &vk_allocation,
                        nullptr));

    const uint32_t buffer_idx     = impl->buffer_pool.get();
    impl->buffer_pool[buffer_idx] = {
        .vk_buffer     = vk_buffer,
        .vk_allocation = vk_allocation,
    };
    return {.h = buffer_idx};
}

void Device::free(Handle<Buffer> buffer) {
    auto& b = impl->buffer_pool[buffer.h];
    vmaDestroyBuffer(impl->vk_allocator, b.vk_buffer, b.vk_allocation);
}

GpuPtr Device::getDevicePointer(Handle<Buffer> buffer) {
    VkBufferDeviceAddressInfo addr_info{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = impl->buffer_pool[buffer.h].vk_buffer,
    };
    return impl->vk_api.vkGetBufferDeviceAddress(impl->device, &addr_info);
}

// MARK: Sempahores

Handle<Semaphore> Device::createSemaphore(uint64_t initValue) {
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
    chk(impl->vk_api.vkCreateSemaphore(impl->device, &timeline_create_info, nullptr, &s));

    // For semaphores, we don't need anything beyond the handle, so we just return the vk handle
    // directly.

    return {.h = reinterpret_cast<uintptr_t>(s)};
}

void Device::waitSemaphore(Handle<Semaphore> sema, uint64_t value) {
    VkSemaphore         s = reinterpret_cast<VkSemaphore>(sema.h);
    VkSemaphoreWaitInfo wait_info{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext          = nullptr,
        .flags          = 0,
        .semaphoreCount = 1,
        .pSemaphores    = &s,
        .pValues        = &value,
    };
    chk(impl->vk_api.vkWaitSemaphores(impl->device, &wait_info, UINT64_MAX));
}

void Device::destroySemaphore(Handle<Semaphore> sema) {
    impl->vk_api.vkDestroySemaphore(impl->device, reinterpret_cast<VkSemaphore>(sema.h), nullptr);
}

void Device::chk(uint64_t result) {
    if (result == VK_SUCCESS) { return; }

    // TODO: Report the error somehow.
}

}  // namespace loon::gpu
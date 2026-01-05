#include "gpu/loon_gpu.h"
#include "vma_usage.h"
#include "volk.h"
#include "vulkan/vulkan_core.h"

namespace loon::gpu {

struct Device::Impl {
  VkDevice device;
  VolkDeviceTable vk_api;
  VmaAllocator vk_allocator = VK_NULL_HANDLE;
};

// TODO: Replace with a handle to device object pool
struct GpuBuffer {
  VkBuffer vk_buffer;
  VmaAllocation vk_allocation;
};

GpuBuffer Device::malloc(size_t bytes, MEMORY memory) {
  return malloc(bytes, 64, memory);
}

GpuBuffer Device::malloc(size_t bytes, size_t align, MEMORY memory) {

  constexpr VkBufferUsageFlags kDefaultUsages =
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  VkBufferCreateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .size = bytes,
      .usage = kDefaultUsages,
      .sharingMode =
          VK_SHARING_MODE_EXCLUSIVE, // TODO: Support multiple queues.
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
  };

  VmaAllocationCreateFlags flags = 0;
  switch (memory) {
  case MEMORY_DEFAULT:
    flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT;
    break;
  case MEMORY_GPU:
    flags = 0;
    break;
  case MEMORY_READBACK:
    flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    break;
  }

  VmaAllocationCreateInfo alloc_info{
      .flags = flags,
      .usage = VMA_MEMORY_USAGE_AUTO,
      .requiredFlags = 0,
      .preferredFlags = 0,
      .memoryTypeBits = 0,
      .pool = VK_NULL_HANDLE,
      .pUserData = 0,
      .priority = 0.,
  };

  VkBuffer vk_buffer = nullptr;
  VmaAllocation vk_allocation = nullptr;
  chk(vmaCreateBuffer(impl->vk_allocator, &create_info, &alloc_info, &vk_buffer,
                      &vk_allocation, nullptr));

  return {
      .vk_buffer = vk_buffer,
      .vk_allocation = vk_allocation,
  };
}

void Device::free(GpuBuffer buffer) {
  vmaDestroyBuffer(impl->vk_allocator, buffer.vk_buffer, buffer.vk_allocation);
}

GpuPtr Device::getDevicePointer(GpuBuffer buffer) {
  VkBufferDeviceAddressInfo addr_info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .pNext = nullptr,
      .buffer = buffer.vk_buffer,
  };
  return impl->vk_api.vkGetBufferDeviceAddress(impl->device, &addr_info);
}

// MARK: Sempahores
struct GpuSemaphore {
  VkSemaphore s;
};

GpuSemaphore Device::createSemaphore(uint64_t initValue) {

  VkSemaphoreTypeCreateInfo semaphore_type{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .pNext = nullptr,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
  };

  VkSemaphoreCreateInfo timeline_create_info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &semaphore_type,
      .flags = 0,
  };

  VkSemaphore s = VK_NULL_HANDLE;
  chk(impl->vk_api.vkCreateSemaphore(impl->device, &timeline_create_info,
                                     nullptr, &s));
  return {s};
}

void Device::waitSemaphore(GpuSemaphore sema, uint64_t value) {
  VkSemaphoreWaitInfo wait_info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .pNext = nullptr,
      .flags = 0,
      1,
      .pSemaphores = &sema.s,
      .pValues = &value,
  };
  chk(impl->vk_api.vkWaitSemaphores(impl->device, &wait_info, UINT64_MAX));
}

void Device::destroySemaphore(GpuSemaphore sema) {
  impl->vk_api.vkDestroySemaphore(impl->device, sema.s, nullptr);
}

void Device::chk(uint64_t result) {
  if (result == VK_SUCCESS) {
    return;
  }

  // TODO: Report the error somehow.
}

} // namespace loon::gpu
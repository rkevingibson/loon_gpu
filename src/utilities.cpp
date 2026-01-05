#include "utilities.h"

#include <cstring>

#include "commands.h"
#include "device.h"
#include "objects.h"
#include "volk.h"
#include "webgpu/webgpu.h"
#include "webgpu/webgpu_loon.h"

namespace webgpu {

// MARK: UsageScope

UsageScope::UsageScope(Allocator alloc) : tex_usages(alloc, 64), buffer_usages(alloc, 64) {}

void UsageScope::add(WGPUTextureView tex, ResourceUsage usage, WGPUShaderStage stage) {
    auto& tex_usage = tex_usages.insert({.key = tex->texture, .value = {}}).pair->value;
    // Check for compatible usage?
    tex_usage.usage = static_cast<ResourceUsage>(tex_usage.usage | usage);
    tex_usage.stage |= stage;

    const uint32_t mip_min   = tex->descriptor.baseMipLevel;
    const uint32_t mip_max   = mip_min + tex->descriptor.mipLevelCount - 1;
    const uint32_t array_min = tex->descriptor.baseArrayLayer;
    const uint32_t array_max = array_min + tex->descriptor.arrayLayerCount - 1;

    tex_usage.min_array_layer = std::min(mip_min, tex_usage.min_array_layer);
    tex_usage.max_array_layer = std::max(mip_max, tex_usage.max_array_layer);
    tex_usage.min_array_layer = std::min(array_min, tex_usage.min_array_layer);
    tex_usage.max_array_layer = std::max(array_max, tex_usage.max_array_layer);
}

void UsageScope::add(WGPUBuffer buffer, ResourceUsage usage, WGPUShaderStage stage) {
    auto& buf_usage = buffer_usages.insert({.key = buffer, .value = {}}).pair->value;
    // TODO: Check for compatible usage?
    buf_usage.usage = static_cast<ResourceUsage>(buf_usage.usage | usage);
    buf_usage.stage |= stage;
}

void UsageScope::try_add(WGPUBuffer buffer, ResourceUsage usage, WGPUShaderStage stage) {
    buffer_usages.insert({buffer, {.usage = usage, .stage = stage}});
}

void UsageScope::update_resource_last_usages() {
    for (auto& tex_usage : tex_usages) {
        tex_usage.key->last_submitted_usage = tex_usage.value.usage;
    }
    for (auto& buf_usage : buffer_usages) {
        buf_usage.key->last_submitted_usage = buf_usage.value.usage;
    }
}

void UsageScope::update_first_last_usages(UsageScope& first_usages, UsageScope& last_usages) const {
    for (auto tex_usage : tex_usages) {
        first_usages.tex_usages.insert({tex_usage.key, tex_usage.value});
        last_usages.tex_usages.insert_or_assign(tex_usage.key, tex_usage.value);
    }

    for (auto& buf_usage : buffer_usages) {
        first_usages.buffer_usages.insert({buf_usage.key, buf_usage.value});
        last_usages.buffer_usages.insert_or_assign(buf_usage.key, buf_usage.value);
    }
}



// MARK: CommandPool

CommandPool::CommandPool(WGPUDevice device, Arena* arena) : device{device}, arena{arena} {
    VkCommandPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags
        = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device->get_queue_family(),
    };
    auto& vk_api = device->vk_api;
    WGPU_VK_CHECK(vkCreateCommandPool(device->vk_device, &pool_info, nullptr, &pool));

    constexpr uint32_t          kInitialCapacity = 16;
    VkCommandBufferAllocateInfo alloc_info{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = kInitialCapacity * 2,
    };
    VkCommandBuffer initial_bufs[kInitialCapacity * 2]{0};
    WGPU_VK_CHECK(vkAllocateCommandBuffers(device->vk_device, &alloc_info, initial_bufs));

    const auto new_alloc
        = device->get_allocator().alloc(kInitialCapacity * sizeof(RingBufferEntry));
    ring_buffer = reinterpret_cast<RingBufferEntry*>(new_alloc.ptr);
    for (uint32_t i = 0; i < kInitialCapacity; ++i) {
        WGPUCommandBuffer buf  = device->cmd_buffers.make(device, WGPU_STRING_VIEW_INIT);
        buf->cmd_pool          = this;
        buf->vk_cmd_buffers[0] = initial_bufs[2 * i + 0];
        buf->vk_cmd_buffers[1] = initial_bufs[2 * i + 1],

        ring_buffer[i] = {
            .buffer         = buf,
            .timeline_value = -1,
        };
    }
    capacity = kInitialCapacity;
}

CommandPool::~CommandPool() {
    device->vk_api.vkDestroyCommandPool(device->vk_device, pool, nullptr);
}

WGPUCommandBuffer CommandPool::allocate_command_buffer(int64_t current_timeline_value) {
    std::unique_lock<std::mutex> lock(mutex);
    const uint32_t               idx = read_idx % capacity;

    if (read_idx + capacity == write_idx
        || ring_buffer[idx].timeline_value > current_timeline_value) {
        // Queue is full, need to grow - slow operation :(
        if (!grow()) { return VK_NULL_HANDLE; }
    }

    // Should be guaranteed that we have an command_buffer to return here, but it may require
    // resetting
    if (!ring_buffer[idx].is_reset()) { ring_buffer[idx].buffer->reset(); }

    read_idx++;
    return ring_buffer[idx].buffer;
}

void CommandPool::free_command_buffer(WGPUCommandBuffer buf, int64_t timeline_value) {
    std::unique_lock<std::mutex> lock(mutex);
    // Place it at write idx, I don't think we can ever be out of space without a bug.
    assert(write_idx != read_idx + capacity);

    const uint32_t idx = write_idx % capacity;
    ring_buffer[idx]   = RingBufferEntry{
          .buffer         = buf,
          .timeline_value = timeline_value,
    };
    write_idx++;
}

bool CommandPool::grow() {
    const uint32_t new_capacity = capacity * 2;

    VkCommandBufferAllocateInfo alloc_info{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 2 * capacity,
    };
    VkCommandBuffer* cmd_buffers
        = reinterpret_cast<VkCommandBuffer*>(arena->alloc(sizeof(VkCommandBuffer) * capacity * 2));
    if (!cmd_buffers) { return false; }
    auto cmd_buffers_free = ScopeGuard(
        [=, this]() { arena->free((void*)cmd_buffers, sizeof(VkCommandBuffer) * capacity * 2); });

    auto& vk_api = device->vk_api;
    WGPU_VK_CHECK(vkAllocateCommandBuffers(device->vk_device, &alloc_info, cmd_buffers));
    const auto new_alloc = device->get_allocator().alloc(new_capacity * sizeof(RingBufferEntry));
    if (new_alloc.ptr == nullptr) { return false; }

    // We insert new buffers at the read index, so copy up to that point, and then from
    const uint32_t   read_offset = (read_idx % capacity);
    RingBufferEntry* buffer      = reinterpret_cast<RingBufferEntry*>(new_alloc.ptr);
    memcpy(buffer, ring_buffer, sizeof(RingBufferEntry) * read_offset);
    for (uint32_t idx = 0; idx < capacity; ++idx) {
        WGPUCommandBuffer buf  = device->cmd_buffers.make(device, WGPU_STRING_VIEW_INIT);
        buf->cmd_pool          = this;
        buf->vk_cmd_buffers[0] = cmd_buffers[2 * idx + 0];
        buf->vk_cmd_buffers[1] = cmd_buffers[2 * idx + 1];

        buffer[idx + read_offset] = {
            .buffer         = buf,
            .timeline_value = -1,
        };
    }
    memcpy(buffer + read_offset + capacity,
           ring_buffer + read_offset,
           sizeof(RingBufferEntry) * (capacity - read_offset));
    write_idx += capacity;

    device->get_allocator().free(
        {.ptr = ring_buffer, .len = static_cast<uint32_t>(sizeof(RingBufferEntry) * capacity)});

    capacity    = new_capacity;
    ring_buffer = buffer;

    return true;
}

// MARK: StagingBuffer

StagingBuffer::StagingBuffer(WGPUDevice device, size_t size) {
    WGPUBufferDescriptor descriptor{
        .nextInChain      = nullptr,
        .label            = "Staging Buffer"_wsv,
        .usage            = WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc,
        .size             = size,
        .mappedAtCreation = true,
    };

    buffer = wgpuDeviceCreateBuffer(device, &descriptor);
}

void* StagingBuffer::get_mapped_range(size_t offset, size_t size) {
    return buffer->get_mapped_range(offset, size);
}

void StagingBuffer::unmap(WGPUBuffer target, size_t offset, size_t size) {
    buffer->unmap();

    WGPUDevice device = buffer->device;

    // TODO: May want to have some device-stored list to add copies to the next submitted command
    // encoder, rather than creating our own here - possible multithreading issues?

    WGPUCommandEncoderDescriptor encoder_desc{
        .nextInChain = nullptr,
        .label       = "Staging Buffer Encoder"_wsv,
    };
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(device, &encoder_desc);

    wgpuCommandEncoderCopyBufferToBuffer(command_encoder, buffer, 0, target, offset, size);
    WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(command_encoder, nullptr);

    wgpuQueueSubmit(&device->queue, 1, &cmd_buffer);
    wgpuCommandBufferRelease(cmd_buffer);
    wgpuCommandEncoderRelease(command_encoder);
    wgpuBufferRelease(buffer);
}

// MARK: Descriptor Set Allocator

DescriptorSetAllocator::DescriptorSetAllocator(WGPUDevice device) :
    m_device(device), m_pools(device->get_allocator(), 32) {}

auto DescriptorSetAllocator::alloc(WGPUBindGroupLayout layout) -> DescriptorAllocation {
    auto device = m_device;

    // We don't use all possible values of VkDescriptorType. This is the largest we use.
    constexpr size_t kLargestDescriptorTypeIndex = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    uint32_t         pool_type_counts[kLargestDescriptorTypeIndex + 1]{0};
    uint32_t         descriptor_type_bitset = 0;
    for (auto& entry : layout->entries) {
        const VkDescriptorType type = entry.descriptor_type();
        descriptor_type_bitset |= 1 << type;
        assert(type <= kLargestDescriptorTypeIndex);
        pool_type_counts[type]++;
    }

    VkDescriptorSetAllocateInfo alloc_info{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext              = nullptr,
        .descriptorPool     = VK_NULL_HANDLE,
        .descriptorSetCount = 1,
        .pSetLayouts        = &layout->vk_set_layout,
    };

    auto& poolQueue
        = m_pools
              .insert({
                  descriptor_type_bitset,
                  PoolQueue{.pools = webgpu::SegmentArray<Pool>(m_device->get_allocator())},
              })
              .pair->value;

    auto& vk_api = m_device->vk_api;

    // For each pool, loop over the list of descriptor sets and attempt to allocate.
    // If all fail, create a new set.
    Pool*           pool = nullptr;
    VkDescriptorSet set  = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < poolQueue.pools.size(); ++i) {
        pool = poolQueue.pools.get(i);
        if (pool->sets_remaining == 0) { continue; }

        alloc_info.descriptorPool = pool->vk_pool;
        VkResult alloc_result
            = vk_api.vkAllocateDescriptorSets(device->vk_device, &alloc_info, &set);

        if (alloc_result == VK_SUCCESS) {
            --(pool->sets_remaining);
            break;
        }
    }

    if (set == VK_NULL_HANDLE) {
        constexpr uint32_t kMaxSetsPerPool = 32;

        auto*                                     arena = device->get_thread_local_arena();
        webgpu::ArenaVector<VkDescriptorPoolSize> pool_sizes(arena);
        for (size_t i = 0; i <= kLargestDescriptorTypeIndex; ++i) {
            constexpr uint32_t kMinDescriptorCount = 64;
            if (pool_type_counts[i] != 0) {
                pool_sizes.push(VkDescriptorPoolSize{
                    .type            = VkDescriptorType(i),
                    .descriptorCount = pool_type_counts[i] > kMinDescriptorCount
                                           ? pool_type_counts[i]
                                           : kMinDescriptorCount,
                });
            }
        }

        VkDescriptorPoolCreateInfo info{
            .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext         = nullptr,
            .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets       = kMaxSetsPerPool,
            .poolSizeCount = pool_sizes.size(),
            .pPoolSizes    = pool_sizes.data(),
        };

        VkDescriptorPool vk_pool = VK_NULL_HANDLE;
        WGPU_VK_CHECK(vkCreateDescriptorPool(m_device->vk_device, &info, nullptr, &vk_pool));

        pool = &poolQueue.pools.emplace_back(Pool{
            .vk_pool        = vk_pool,
            .sets_remaining = kMaxSetsPerPool - 1,
        });

        alloc_info.descriptorPool = vk_pool;
        WGPU_VK_CHECK(vkAllocateDescriptorSets(device->vk_device, &alloc_info, &set));
    }

    return {.pool = pool, .set = set};
}

void DescriptorSetAllocator::DescriptorAllocation::free(WGPUDevice device) {
    auto& vk_api = device->vk_api;
    WGPU_VK_CHECK(vkFreeDescriptorSets(device->vk_device, pool->vk_pool, 1, &set));
    pool->sets_remaining++;
}

}  // namespace webgpu
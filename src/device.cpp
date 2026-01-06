#include "device.h"

#include <vulkan/vulkan_core.h>

#include <algorithm>

#include "commands.h"
#include "containers.h"
#include "instance.h"
#include "utilities.h"
#include "vma_usage.h"
#include "wgpu_to_vk.h"

namespace webgpu {
// MARK: ErrorStack

struct CapturedError {
    WGPUErrorType  type{WGPUErrorType_NoError};
    WGPUStringView message = WGPU_STRING_VIEW_INIT;
};

struct ErrorScope {
    WGPUErrorFilter filter{WGPUErrorFilter_Force32};
    CapturedError   error;
};

class ErrorStack {
   public:
    void          initialize(const WGPUUncapturedErrorCallbackInfo& cb_info);
    void          error(WGPUDevice device, WGPUErrorType type, WGPUStringView error_string);
    void          push_scope(WGPUDevice device, WGPUErrorFilter filter);
    CapturedError pop_scope(WGPUDevice device);

    static void default_uncaptured_error_callback(WGPUDevice const*,
                                                  WGPUErrorType,
                                                  WGPUStringView,
                                                  WGPU_NULLABLE void*,
                                                  WGPU_NULLABLE void*) noexcept {}

   private:
    static constexpr int32_t    kMaxNumErrorScopes = 1024;
    ErrorScope                  error_scopes[kMaxNumErrorScopes];
    int32_t                     error_scopes_count{0};
    WGPUUncapturedErrorCallback uncaptured_error_callback{default_uncaptured_error_callback};
    void*                       uncaptured_error_userdata1{nullptr};
    void*                       uncaptured_error_userdata2{nullptr};
};

void ErrorStack::initialize(const WGPUUncapturedErrorCallbackInfo& cb_info) {
    if (cb_info.callback) {
        uncaptured_error_callback  = cb_info.callback;
        uncaptured_error_userdata1 = cb_info.userdata1;
        uncaptured_error_userdata2 = cb_info.userdata2;
    }
}

void ErrorStack::error(WGPUDevice device, WGPUErrorType type, WGPUStringView error_string) {
    WGPUErrorFilter target_filter = WGPUErrorFilter_Force32;
    switch (type) {
        case WGPUErrorType_Validation: target_filter = WGPUErrorFilter_Validation; break;
        case WGPUErrorType_OutOfMemory: target_filter = WGPUErrorFilter_OutOfMemory; break;
        case WGPUErrorType_Internal: target_filter = WGPUErrorFilter_Internal; break;
        default: break;
    }

    int32_t stack_index = error_scopes_count - 1;
    while (stack_index >= 0) {
        if (error_scopes[stack_index].filter == target_filter) {
            // We capture the first error we see for a scope, that's it. Other errors are
            // dropped.
            if (error_scopes[stack_index].error.type == WGPUErrorType_NoError) {
                error_scopes[stack_index].error.type    = type;
                error_scopes[stack_index].error.message = error_string;
            }
            return;
        }
        --stack_index;
    }

    // If we've reached here, no error scope is valid so try the uncaptured error callback
    uncaptured_error_callback(&device,
                              type,
                              error_string,
                              uncaptured_error_userdata1,
                              uncaptured_error_userdata2);
}

void ErrorStack::push_scope(WGPUDevice device, WGPUErrorFilter filter) {
    if (error_scopes_count == kMaxNumErrorScopes) {
        error(device, WGPUErrorType_OutOfMemory, "Maximum number of error scopes exceeded"_wsv);
        return;
    }
    error_scopes[error_scopes_count].error.type = WGPUErrorType_NoError;
    error_scopes[error_scopes_count].filter     = filter;
    ++error_scopes_count;
}

CapturedError ErrorStack::pop_scope(WGPUDevice device) {
    if (error_scopes_count == 0) {
        error(device, WGPUErrorType_Validation, "Popping an empty error stack"_wsv);
        return CapturedError{};
    }
    // Need to do something with the error:
    const auto& scope = error_scopes[error_scopes_count];
    --error_scopes_count;
    return scope.error;
}

}  // namespace webgpu

// MARK: Device

struct WGPUDeviceImpl::ThreadLocalState {
    constexpr static size_t kArenaSize = 64ll * 1024;
    loon::gpu::ErrorStack   error_stack;
    loon::gpu::Allocator    allocator;
    WGPULoonMemoryBlock     arena_memory;
    loon::gpu::Arena        arena;
    loon::gpu::CommandPool  command_pool;

    ThreadLocalState(const loon::gpu::Allocator& alloc, WGPUDeviceImpl* device);
    ~ThreadLocalState() { allocator.free(arena_memory); }
};

WGPUDeviceImpl::WGPUDeviceImpl() = default;


WGPURequestDeviceStatus WGPUDeviceImpl::initialize(WGPUAdapter                 adapter,
                                                   const WGPUDeviceDescriptor* descriptor) {
    float                   queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = 0,
        .queueFamilyIndex = adapter->queue_family,
        .queueCount       = 1,
        .pQueuePriorities = &queue_priority,
    };

    // TODO: Check required limits against our limits.
    VkPhysicalDeviceVulkan13Features vulkan_13_features{};
    vulkan_13_features.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan_13_features.pNext            = nullptr;
    vulkan_13_features.dynamicRendering = true;
    vulkan_13_features.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features vulkan_12_features{};
    vulkan_12_features.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan_12_features.pNext             = &vulkan_13_features;
    vulkan_12_features.timelineSemaphore = true;

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
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &queue_create_info,
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = loon::gpu::kRequiredDeviceExtensionsCount,
        .ppEnabledExtensionNames = loon::gpu::kRequiredDeviceExtensions,
        .pEnabledFeatures        = nullptr,
    };

    VkResult result
        = vkCreateDevice(adapter->vk_physical_device, &create_info, nullptr, &vk_device);
    if (result != VK_SUCCESS) { return WGPURequestDeviceStatus_Error; }

    this->m_adapter  = adapter;
    this->m_instance = adapter->instance;
    volkLoadDeviceTable(&vk_api, vk_device);

    VmaAllocatorCreateInfo vma_create_info{
        .flags                          = 0,
        .physicalDevice                 = adapter->vk_physical_device,
        .device                         = vk_device,
        .preferredLargeHeapBlockSize    = 0,
        .pAllocationCallbacks           = nullptr,
        .pDeviceMemoryCallbacks         = nullptr,
        .pHeapSizeLimit                 = nullptr,
        .pVulkanFunctions               = nullptr,
        .instance                       = m_instance->get_vk_instance(),
        .vulkanApiVersion               = VK_API_VERSION_1_3,
        .pTypeExternalMemoryHandleTypes = nullptr,
    };
    VmaVulkanFunctions vulkan_functions;
    vmaImportVulkanFunctionsFromVolk(&vma_create_info, &vulkan_functions);
    vma_create_info.pVulkanFunctions = &vulkan_functions;

    vma_create_info.flags            = 0;
    vma_create_info.vulkanApiVersion = VK_API_VERSION_1_3;
    vma_create_info.physicalDevice   = adapter->vk_physical_device;
    vmaCreateAllocator(&vma_create_info, &vk_allocator);

    this->queue.initialize(this, get_queue_family());
    m_allocator          = m_instance->get_allocator();
    m_shader_modules     = loon::gpu::ObjectList<WGPUShaderModuleImpl>(m_allocator);
    m_render_pipelines   = loon::gpu::ObjectList<WGPURenderPipelineImpl>(m_allocator);
    m_bind_group_layouts = loon::gpu::ObjectList<WGPUBindGroupLayoutImpl>(m_allocator);
    m_bind_groups        = loon::gpu::ObjectList<WGPUBindGroupImpl>(m_allocator);
    m_pipeline_layouts   = loon::gpu::ObjectList<WGPUPipelineLayoutImpl>(m_allocator);
    m_textures           = loon::gpu::ObjectList<WGPUTextureImpl>(m_allocator);
    m_texture_views      = loon::gpu::ObjectList<WGPUTextureViewImpl>(m_allocator);
    m_cmd_encoders       = loon::gpu::ObjectList<WGPUCommandEncoderImpl>(m_allocator);
    m_render_passes      = loon::gpu::ObjectList<WGPURenderPassEncoderImpl>(m_allocator);
    m_compute_passes     = loon::gpu::ObjectList<WGPUComputePassEncoderImpl>(m_allocator);
    this->cmd_buffers    = loon::gpu::ObjectList<WGPUCommandBufferImpl>(m_allocator);
    m_error_callback     = descriptor->uncapturedErrorCallbackInfo;
    m_tls_key            = loon::gpu::tls_alloc([](void* data) {
        auto state = reinterpret_cast<WGPUDeviceImpl::ThreadLocalState*>(data);
        state->~ThreadLocalState();
    });

    m_descriptor_set_allocator = loon::gpu::DescriptorSetAllocator(this);
    this->label.set(m_allocator, descriptor->label);
    add_ref();

    return WGPURequestDeviceStatus_Success;
}

WGPUDeviceImpl::~WGPUDeviceImpl() {
    auto device = this;
    WGPU_VK_CHECK(vkDeviceWaitIdle(vk_device));

    queue.reset();


    cmd_buffers
        .clear();  // Command buffers hold references to other stuff, so should be cleared first.
    m_shader_modules.clear();
    m_render_pipelines.clear();
    m_bind_group_layouts.clear();
    m_pipeline_layouts.clear();
    m_texture_views.clear();
    m_textures.clear();
    m_buffers.clear();
    m_render_passes.clear();
    m_compute_passes.clear();
    m_cmd_encoders.clear();
    loon::gpu::tls_free(m_tls_key);

    vmaDestroyAllocator(vk_allocator);

    vk_api.vkDestroyDevice(vk_device, nullptr);
}

void WGPUDeviceImpl::add_ref() {
    m_refcount.add();
}

void WGPUDeviceImpl::release() {
    if (m_refcount.release()) { m_adapter->free_device(this); }
}

void WGPUDeviceImpl::destroy() {
    // TODO:
}

void WGPUDeviceImpl::vulkan_error(VkResult res, const char* file, int line) {
    int   error_len = 1 + snprintf(nullptr, 0, "Vulkan error %d - %s:%d", res, file, line);
    auto* arena     = get_thread_local_arena();
    auto  str_ptr   = (char*)arena->alloc(error_len);
    snprintf(str_ptr, error_len, "Vulkan error %d - %s:%d", res, file, line);
    error(WGPUErrorType_Internal,
          WGPUStringView{
              .data   = str_ptr,
              .length = static_cast<size_t>(error_len - 1),
          });

    arena->free(str_ptr, error_len);
}

void WGPUDeviceImpl::error(WGPUErrorType type, WGPUStringView msg) {
    get_thread_local_state()->error_stack.error(this, type, msg);
}

void WGPUDeviceImpl::push_error_scope(WGPUErrorFilter filter) {
    auto thread_local_state = get_thread_local_state();
    thread_local_state->error_stack.push_scope(this, filter);
}

WGPUFuture WGPUDeviceImpl::pop_error_scope(WGPUPopErrorScopeCallbackInfo callback_info) {
    auto thread_local_state = get_thread_local_state();
    thread_local_state->error_stack.pop_scope(this);
    // TODO: Implement
    return WGPU_FUTURE_INIT;
}

// MARK: Resource Management

WGPUBuffer WGPUDeviceImpl::create_buffer(WGPUBufferDescriptor const* descriptor) {
    if (!loon::gpu::validate(this, descriptor)) { return nullptr; }

    // TODO: Need to support mappedAtCreation even if the buffer is not otherwise mappable. How do
    // we do that?

    const bool mappable = (descriptor->usage & WGPUBufferUsage_MapRead)
                          | (descriptor->usage & WGPUBufferUsage_MapWrite);

    auto usage = loon::gpu::bridge_buffer_usage(descriptor->usage);
    if (!mappable && descriptor->mappedAtCreation) {
        // If we can't map, we'll need to copy from a staging buffer.
        usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    VkBufferCreateInfo create_info{
        .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .size                  = descriptor->size,
        .usage                 = usage,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
    };

    VmaAllocationCreateInfo alloc_info{
        .flags
        = mappable ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT : VmaAllocationCreateFlags(0),
        .usage          = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags  = 0,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool           = VK_NULL_HANDLE,
        .pUserData      = 0,
        .priority       = 0.,
    };

    VkBuffer      vk_buffer;
    VmaAllocation vk_allocation;
    auto          device = this;
    WGPU_VMA_CHECK(vmaCreateBuffer(vk_allocator,
                                   &create_info,
                                   &alloc_info,
                                   &vk_buffer,
                                   &vk_allocation,
                                   nullptr));

    void*                    mapping_ptr = nullptr;
    loon::gpu::StagingBuffer staging_buffer{};
    if (descriptor->mappedAtCreation && mappable) {
        vmaMapMemory(vk_allocator, vk_allocation, &mapping_ptr);
    } else if (descriptor->mappedAtCreation && !mappable) {
        staging_buffer = loon::gpu::StagingBuffer(device, descriptor->size);
        mapping_ptr    = staging_buffer.get_mapped_range(0, descriptor->size);
    }

    auto buffer            = m_buffers.make(this, descriptor->label);
    buffer->internal_state = descriptor->mappedAtCreation
                                 ? (mappable ? WGPUBufferImpl::InternalState::Unavailable
                                             : WGPUBufferImpl::InternalState::Init)
                                 : WGPUBufferImpl::InternalState::Available;
    buffer->size = descriptor->size, buffer->usage = descriptor->usage;
    buffer->map_state
        = descriptor->mappedAtCreation ? WGPUBufferMapState_Mapped : WGPUBufferMapState_Unmapped;
    buffer->vk_buffer     = vk_buffer;
    buffer->vk_allocation = vk_allocation;
    vmaGetAllocationInfo(vk_allocator, vk_allocation, &buffer->vk_allocation_info);
    buffer->mapping = {
        .ptr            = mapping_ptr,
        .map_mode       = descriptor->mappedAtCreation ? WGPUMapMode_Write : WGPUMapMode_None,
        .offset         = 0,
        .size           = descriptor->mappedAtCreation ? descriptor->size : 0,
        .staging_buffer = staging_buffer,
    };


    return loon::gpu::return_with_ownership(buffer);
}

WGPUBindGroupLayout WGPUDeviceImpl::create_bind_group_layout(
    WGPUBindGroupLayoutDescriptor const* descriptor) {
    if (!loon::gpu::validate(this, descriptor)) { return nullptr; }

    const size_t num_bindings = descriptor->entryCount;

    loon::gpu::Arena* arena    = get_thread_local_arena();
    auto              bindings = reinterpret_cast<VkDescriptorSetLayoutBinding*>(
        arena->alloc(num_bindings * sizeof(VkDescriptorSetLayoutBinding)));
    auto arena_free = loon::gpu::ScopeGuard([arena, num_bindings, bindings]() {
        arena->free(bindings, sizeof(VkDescriptorSetLayoutBinding) * num_bindings);
    });
    loon::gpu::Vector<WGPUBindGroupLayoutImpl::LayoutEntry> entries(m_allocator);
    uint32_t                                                dynamic_offset_count = 0;

    if (bindings == nullptr) {
        error(WGPUErrorType_OutOfMemory,
              "Failed to allocate memory for VkDescriptorSetLayoutBinding."_wsv);
        return nullptr;
    }

    for (size_t i = 0; i < num_bindings; ++i) {
        auto&                                entry = descriptor->entries[i];
        WGPUBindGroupLayoutImpl::LayoutEntry stored_entry{};

        VkDescriptorType descriptor_type  = VK_DESCRIPTOR_TYPE_MAX_ENUM;
        uint32_t         descriptor_count = 1;

        if (entry.buffer.type != WGPUBufferBindingType_BindingNotUsed) {
            switch (entry.buffer.type) {
                case WGPUBufferBindingType_Undefined:
                case WGPUBufferBindingType_Uniform:
                    descriptor_type = entry.buffer.hasDynamicOffset
                                          ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                                          : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    break;
                case WGPUBufferBindingType_Storage:
                case WGPUBufferBindingType_ReadOnlyStorage:  // TODO: Probably want to mark this
                                                             // as read-only at some point for
                                                             // synchronization tracking.
                    descriptor_type = entry.buffer.hasDynamicOffset
                                          ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                                          : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    break;
                default:
                    error(WGPUErrorType_Internal, "Invalid buffer binding type"_wsv);
                    return nullptr;
            }

            stored_entry = {
                .binding    = static_cast<uint16_t>(entry.binding),
                .entry_type = WGPUBindGroupLayoutImpl::BindingType::kBuffer,
                .visibility = entry.visibility,
                .buffer     = entry.buffer,
            };
            dynamic_offset_count += bool(entry.buffer.hasDynamicOffset);
        } else if (entry.sampler.type != WGPUSamplerBindingType_BindingNotUsed) {
            descriptor_type = VK_DESCRIPTOR_TYPE_SAMPLER;
            stored_entry    = {
                   .binding    = static_cast<uint16_t>(entry.binding),
                   .entry_type = WGPUBindGroupLayoutImpl::BindingType::kSampler,
                   .visibility = entry.visibility,
                   .sampler    = entry.sampler,
            };
        } else if (entry.texture.sampleType != WGPUTextureSampleType_BindingNotUsed) {
            // TODO: Additional info for validation
            descriptor_type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            stored_entry    = {
                   .binding    = static_cast<uint16_t>(entry.binding),
                   .entry_type = WGPUBindGroupLayoutImpl::BindingType::kTexture,
                   .visibility = entry.visibility,
                   .texture    = entry.texture,
            };
        } else if (entry.storageTexture.access != WGPUStorageTextureAccess_BindingNotUsed) {
            descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            stored_entry    = {
                   .binding         = static_cast<uint16_t>(entry.binding),
                   .entry_type      = WGPUBindGroupLayoutImpl::BindingType::kStorageTexture,
                   .visibility      = entry.visibility,
                   .storage_texture = entry.storageTexture,
            };
        }

        entries.emplace_back(std::move(stored_entry));

        if (descriptor_type == VK_DESCRIPTOR_TYPE_MAX_ENUM) {
            error(WGPUErrorType_Internal,
                  "Invalid bind group layout entry - no binding specified"_wsv);
        }

        bindings[i] = VkDescriptorSetLayoutBinding{
            .binding            = entry.binding,
            .descriptorType     = descriptor_type,
            .descriptorCount    = descriptor_count,
            .stageFlags         = loon::gpu::bridge_shader_stage(entry.visibility),
            .pImmutableSamplers = nullptr,
        };
    }

    VkDescriptorSetLayoutCreateInfo create_info{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext        = nullptr,
        .flags        = 0,
        .bindingCount = static_cast<uint32_t>(num_bindings),
        .pBindings    = bindings,
    };
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    auto                  device     = this;
    WGPU_VK_CHECK(vkCreateDescriptorSetLayout(vk_device, &create_info, nullptr, &set_layout));

    auto result = m_bind_group_layouts.make(this, descriptor->label);
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.binding < b.binding;
    });
    result->entries              = std::move(entries);
    result->dynamic_offset_count = dynamic_offset_count;
    result->vk_set_layout        = set_layout;
    result->entry_map
        = loon::gpu::HashTable<uint16_t, uint16_t>(get_allocator(), entries.size() * 2);
    for (uint32_t i = 0; i < result->entries.size(); ++i) {
        result->entry_map.insert_or_assign(result->entries[i].binding, i);
    }
    WGPUBindGroupEntry test;
    return loon::gpu::return_with_ownership(result);
}

WGPUBindGroup WGPUDeviceImpl::create_bind_group(WGPUBindGroupDescriptor const* descriptor) {
    // TODO: validation
    WGPUBindGroup bind_group = m_bind_groups.make(this, descriptor->label);
    descriptor->layout->add_ref_internal();
    bind_group->layout = descriptor->layout;
    bind_group->entries
        = loon::gpu::Vector(m_allocator, descriptor->entries, descriptor->entryCount);
    bind_group->used_resources = loon::gpu::UsageScope(m_allocator);

    loon::gpu::ArenaVector<VkWriteDescriptorSet> write_descriptor_sets(get_thread_local_arena());

    bind_group->descriptor = m_descriptor_set_allocator.alloc(bind_group->layout);

    for (const auto& entry : bind_group->entries) {
        // https://www.w3.org/TR/webgpu/#dom-gpudevice-createbindgroup
        const uint16_t binding_idx = (uint16_t)entry.binding;
        // TODO: PERF - should see if we can avoid this hash lookup, replace with binary search or
        // similar.
        const uint16_t layout_entry_idx = bind_group->layout->entry_map.find(binding_idx)->value;
        auto&          layout_binding   = bind_group->layout->entries[layout_entry_idx];


        VkDescriptorImageInfo  image_info{};
        VkDescriptorBufferInfo buffer_info{};


        if (entry.buffer) {
            bind_group->used_resources.add(entry.buffer,
                                           layout_binding.internal_usage(),
                                           layout_binding.visibility);

        } else if (entry.textureView) {
            bind_group->used_resources.add(entry.textureView,
                                           layout_binding.internal_usage(),
                                           layout_binding.visibility);

            image_info = VkDescriptorImageInfo{
                .sampler     = VK_NULL_HANDLE,
                .imageView   = entry.textureView->vk_image_view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
        } else if (entry.sampler) {
            // TODO: Image samplers support
        }

        write_descriptor_sets.push({
            .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext            = nullptr,
            .dstSet           = bind_group->descriptor.set,
            .dstBinding       = binding_idx,
            .dstArrayElement  = 0,
            .descriptorCount  = 1,
            .descriptorType   = layout_binding.descriptor_type(),
            .pImageInfo       = entry.textureView ? &image_info : nullptr,
            .pBufferInfo      = entry.buffer ? &buffer_info : nullptr,
            .pTexelBufferView = nullptr,
        });
    }

    vk_api.vkUpdateDescriptorSets(vk_device,
                                  write_descriptor_sets.size(),
                                  write_descriptor_sets.data(),
                                  0,
                                  nullptr);

    return loon::gpu::return_with_ownership(bind_group);
}

WGPUPipelineLayout WGPUDeviceImpl::create_pipeline_layout(
    WGPUPipelineLayoutDescriptor const* descriptor) {
    if (!loon::gpu::validate(this, descriptor)) { return nullptr; }

    auto arena       = get_thread_local_arena();
    auto set_layouts = reinterpret_cast<VkDescriptorSetLayout*>(
        arena->alloc(sizeof(VkDescriptorSetLayout) * descriptor->bindGroupLayoutCount));

    loon::gpu::Stack<WGPUBindGroupLayout, loon::gpu::kMaxBindGroups> bind_group_layouts;

    for (size_t set_idx = 0; set_idx < descriptor->bindGroupLayoutCount; ++set_idx) {
        if (descriptor->bindGroupLayouts[set_idx]
            && descriptor->bindGroupLayouts[set_idx]->entries.size() != 0) {
            set_layouts[set_idx] = descriptor->bindGroupLayouts[set_idx]->vk_set_layout;
            descriptor->bindGroupLayouts[set_idx]->add_ref_internal();
            bind_group_layouts.push(descriptor->bindGroupLayouts[set_idx]);
        } else {
            set_layouts[set_idx] = VK_NULL_HANDLE;
            bind_group_layouts.push(nullptr);
        }
    }

    VkPipelineLayoutCreateInfo create_info{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0,
        .setLayoutCount         = static_cast<uint32_t>(descriptor->bindGroupLayoutCount),
        .pSetLayouts            = set_layouts,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges    = nullptr,
    };

    VkPipelineLayout pipeline_layout;
    auto             device = this;
    WGPU_VK_CHECK(
        vkCreatePipelineLayout(device->vk_device, &create_info, nullptr, &pipeline_layout));
    arena->free((const void*)set_layouts,
                sizeof(VkDescriptorSetLayout) * descriptor->bindGroupLayoutCount);

    auto result                = device->m_pipeline_layouts.make(device, descriptor->label);
    result->bind_group_layouts = std::move(bind_group_layouts);
    result->vk_layout          = pipeline_layout;
    return loon::gpu::return_with_ownership(result);
}

// MARK: Shaders

WGPUShaderModule WGPUDeviceImpl::create_shader_module(
    WGPUShaderModuleDescriptor const* descriptor) {
    if (!descriptor->nextInChain && descriptor->nextInChain->sType != WGPUSType_ShaderSourceSPIRV) {
        error(
            WGPUErrorType_Validation,
            "wgpuDeviceCreateShaderModule - Invalid arguments. Requires a chain with ShaderSource_SPIRV"_wsv);

        return nullptr;
    }

    WGPUShaderSourceSPIRV* source
        = reinterpret_cast<WGPUShaderSourceSPIRV*>(descriptor->nextInChain);

    VkShaderModuleCreateInfo create_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = source->codeSize,
        .pCode    = source->code,
    };
    VkShaderModule vk_module;
    auto           device = this;
    WGPU_VK_CHECK(vkCreateShaderModule(device->vk_device, &create_info, nullptr, &vk_module));
    auto module       = device->m_shader_modules.make(device, descriptor->label);
    module->vk_module = vk_module;

    return loon::gpu::return_with_ownership(module);
}

WGPURenderPipeline WGPUDeviceImpl::create_render_pipeline(
    WGPURenderPipelineDescriptor const* descriptor) {
    if (!loon::gpu::validate(this, descriptor)) { return nullptr; }

    loon::gpu::Stack<VkPipelineShaderStageCreateInfo, 2> stages;

    // Vertex state:
    loon::gpu::Stack<VkVertexInputBindingDescription, loon::gpu::kMaxVertexBuffers>
        binding_descriptions;
    loon::gpu::Stack<VkVertexInputAttributeDescription, loon::gpu::kMaxVertexInputAttributes>
        vertex_attributes;

    auto vertex_entry_point = get_temp_null_terminated_string(descriptor->vertex.entryPoint);
    stages.push({
        .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext               = nullptr,
        .flags               = 0,
        .stage               = VK_SHADER_STAGE_VERTEX_BIT,
        .module              = descriptor->vertex.module->vk_module,
        .pName               = vertex_entry_point,
        .pSpecializationInfo = nullptr,  // TODO: Support specialization constants.
    });

    const size_t vertex_buffer_count = descriptor->vertex.bufferCount;
    for (size_t buffer_idx = 0; buffer_idx < vertex_buffer_count; ++buffer_idx) {
        const WGPUVertexBufferLayout& buffer_layout = descriptor->vertex.buffers[buffer_idx];

        if (buffer_layout.attributeCount == 0
            && buffer_layout.stepMode == WGPUVertexStepMode_Undefined) {
            // Empty vertex buffer slot, skip it.
            continue;
        }

        VkVertexInputRate inputRate = VK_VERTEX_INPUT_RATE_MAX_ENUM;
        switch (buffer_layout.stepMode) {
            case WGPUVertexStepMode_Undefined:
            case WGPUVertexStepMode_Vertex: inputRate = VK_VERTEX_INPUT_RATE_VERTEX; break;
            case WGPUVertexStepMode_Instance: inputRate = VK_VERTEX_INPUT_RATE_INSTANCE; break;
            default: break;
        }

        binding_descriptions.push({
            .binding   = static_cast<uint32_t>(buffer_idx),
            .stride    = static_cast<uint32_t>(buffer_layout.arrayStride),
            .inputRate = inputRate,
        });

        for (size_t attr_idx = 0; attr_idx < buffer_layout.attributeCount; ++attr_idx) {
            const auto& attribute = buffer_layout.attributes[attr_idx];

            vertex_attributes.push({
                .location = attribute.shaderLocation,
                .binding  = static_cast<uint32_t>(buffer_idx),
                .format   = loon::gpu::bridge(attribute.format),
                .offset   = static_cast<uint32_t>(attribute.offset),
            });
        }
    }

    VkPipelineVertexInputStateCreateInfo vertex_input_state{
        .sType                         = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext                         = nullptr,
        .flags                         = 0,
        .vertexBindingDescriptionCount = binding_descriptions.size(),
        .pVertexBindingDescriptions    = binding_descriptions.data(),
        .vertexAttributeDescriptionCount = vertex_attributes.size(),
        .pVertexAttributeDescriptions    = vertex_attributes.data(),
    };

    // Primitive state:
    VkPipelineInputAssemblyStateCreateInfo input_assembly_state{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0,
        .topology               = loon::gpu::bridge(descriptor->primitive.topology),
        .primitiveRestartEnable = false,
    };


    // Depth-stencil state:
    VkPipelineDepthStencilStateCreateInfo depth_stencil_state;
    VkFormat                              depth_attachment_format   = VK_FORMAT_UNDEFINED;
    VkFormat                              stencil_attachment_format = VK_FORMAT_UNDEFINED;
    if (descriptor->depthStencil) {
        depth_stencil_state                 = loon::gpu::bridge(*descriptor->depthStencil);
        const VkFormat depth_stencil_format = loon::gpu::bridge(descriptor->depthStencil->format);
        switch (descriptor->depthStencil->format) {
            case WGPUTextureFormat_Stencil8: {  // Stencil-only
                stencil_attachment_format = depth_stencil_format;
                break;
            }
            case WGPUTextureFormat_Depth16Unorm:  // Depth-only
            case WGPUTextureFormat_Depth24Plus:
            case WGPUTextureFormat_Depth32Float: {
                depth_attachment_format = depth_stencil_format;
                break;
            }
            case WGPUTextureFormat_Depth32FloatStencil8:  // Stencil and depth
            case WGPUTextureFormat_Depth24PlusStencil8: {
                depth_attachment_format   = depth_stencil_format;
                stencil_attachment_format = depth_stencil_format;
                break;
            }
            default: break;
        }
    }

    // Color blend state
    loon::gpu::Stack<VkPipelineColorBlendAttachmentState, loon::gpu::kMaxColorAttachments>
                                                                color_blend_attachment_states{};
    loon::gpu::Stack<VkFormat, loon::gpu::kMaxColorAttachments> color_attachment_formats{};

    VkPipelineColorBlendStateCreateInfo color_blend_state{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0,
        .logicOpEnable   = false,
        .logicOp         = VK_LOGIC_OP_NO_OP,
        .attachmentCount = 0,
        .pAttachments    = nullptr,
        .blendConstants  = {1.f, 1.f, 1.f, 1.f},
    };

    const char* fragment_entry_point = nullptr;
    if (descriptor->fragment) {
        fragment_entry_point = get_temp_null_terminated_string(descriptor->fragment->entryPoint);
        stages.push({
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext               = nullptr,
            .flags               = 0,
            .stage               = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module              = descriptor->fragment->module->vk_module,
            .pName               = descriptor->fragment->entryPoint.data,
            .pSpecializationInfo = nullptr,  // TODO: Support specialization constants.
        });

        for (size_t target_idx = 0; target_idx < descriptor->fragment->targetCount; ++target_idx) {
            const auto& target = descriptor->fragment->targets[target_idx];

            const auto attachment_state = loon::gpu::bridge(target);
            color_blend_attachment_states.push(attachment_state);
            color_attachment_formats.push(loon::gpu::bridge(target.format));
        }

        color_blend_state.attachmentCount = color_blend_attachment_states.size();
        color_blend_state.pAttachments    = color_blend_attachment_states.data();
    }

    // Rasterization state:

    VkPipelineRasterizationStateCreateInfo rasterization_state{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = 0,
        .depthClampEnable        = descriptor->primitive.unclippedDepth,
        .rasterizerDiscardEnable = false,
        .polygonMode             = VK_POLYGON_MODE_FILL,
        .cullMode                = loon::gpu::bridge(descriptor->primitive.cullMode),
        .frontFace               = loon::gpu::bridge(descriptor->primitive.frontFace),
        .depthBiasEnable
        = descriptor->depthStencil ? (descriptor->depthStencil->depthBias != 0) : false,
        .depthBiasConstantFactor = 0,
        .depthBiasClamp = descriptor->depthStencil ? (descriptor->depthStencil->depthBiasClamp) : 0,
        .depthBiasSlopeFactor
        = descriptor->depthStencil ? (descriptor->depthStencil->depthBiasSlopeScale) : 0,
        .lineWidth = 1.f,
    };

    // Multisample state:

    VkSampleMask                         sample_mask = descriptor->multisample.count;
    VkPipelineMultisampleStateCreateInfo multisample_state{
        .sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .rasterizationSamples  = loon::gpu::bridge_sample_count(descriptor->multisample.count),
        .sampleShadingEnable   = false,
        .minSampleShading      = 1.0f,
        .pSampleMask           = &sample_mask,
        .alphaToCoverageEnable = descriptor->multisample.alphaToCoverageEnabled,
        .alphaToOneEnable      = false,  // TODO: What does WebGPU spec say?
    };


    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT,
        VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS,

        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
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
        .stageCount          = stages.size(),
        .pStages             = stages.data(),
        .pVertexInputState   = &vertex_input_state,
        .pInputAssemblyState = &input_assembly_state,
        .pTessellationState  = nullptr,
        .pViewportState      = nullptr,  // Viewport state is dynamic
        .pRasterizationState = &rasterization_state,
        .pMultisampleState   = &multisample_state,
        .pDepthStencilState  = descriptor->depthStencil ? &depth_stencil_state : nullptr,
        .pColorBlendState    = descriptor->fragment ? &color_blend_state : nullptr,
        .pDynamicState       = &dynamic_state,
        .layout              = descriptor->layout ? descriptor->layout->vk_layout : VK_NULL_HANDLE,
        .renderPass          = VK_NULL_HANDLE,
        .subpass             = 0,
        .basePipelineHandle  = VK_NULL_HANDLE,
        .basePipelineIndex   = 0,
    };
    VkPipeline vk_pipeline;
    auto       device = this;
    WGPU_VK_CHECK(vkCreateGraphicsPipelines(device->vk_device,
                                            VK_NULL_HANDLE,
                                            1,
                                            &create_info,
                                            nullptr,
                                            &vk_pipeline));


    free_temp_string(fragment_entry_point);
    free_temp_string(vertex_entry_point);

    auto pipeline         = device->m_render_pipelines.make(device, descriptor->label);
    pipeline->vk_pipeline = vk_pipeline;
    descriptor->layout->add_ref_internal();
    pipeline->pipeline_layout = descriptor->layout;
    return loon::gpu::return_with_ownership(pipeline);
}


// MARK: Commands

WGPUCommandEncoder WGPUDeviceImpl::create_command_encoder(
    WGPU_NULLABLE WGPUCommandEncoderDescriptor const* descriptor) {
    auto encoder            = m_cmd_encoders.make(this, descriptor ? descriptor->label : ""_wsv);
    encoder->commands_mixin = loon::gpu::CommandsMixin(m_allocator);
    return loon::gpu::return_with_ownership(encoder);
}

WGPUCommandBuffer WGPUDeviceImpl::allocate_command_buffer() {
    const uint64_t timeline_value = queue.get_current_timeline_value();
    return get_thread_local_state()->command_pool.allocate_command_buffer(
        static_cast<int64_t>(timeline_value));
}


void WGPUDeviceImpl::free(WGPUBuffer buf) {
    m_buffers.free(buf);
}


void WGPUDeviceImpl::free(WGPUShaderModule sm) {
    m_shader_modules.free(sm);
}

void WGPUDeviceImpl::free(WGPURenderPipeline rp) {
    m_render_pipelines.free(rp);
}

void WGPUDeviceImpl::free(WGPUComputePipeline cp) {
    m_compute_pipelines.free(cp);
}

void WGPUDeviceImpl::free(WGPUBindGroupLayout bgl) {
    m_bind_group_layouts.free(bgl);
}

void WGPUDeviceImpl::free(WGPUPipelineLayout pl) {
    m_pipeline_layouts.free(pl);
}

void WGPUDeviceImpl::free(WGPUTexture tex) {
    m_textures.free(tex);
}

void WGPUDeviceImpl::free(WGPUTextureView tv) {
    m_texture_views.free(tv);
}

void WGPUDeviceImpl::free(WGPUCommandEncoder ce) {
    m_cmd_encoders.free(ce);
}

void WGPUDeviceImpl::free(WGPURenderPassEncoder rp) {
    for (auto& attachment : rp->color_attachments) {
        if (attachment.view && attachment.view->release_internal()) { free(attachment.view); }
    }

    if (rp->has_depth_stencil_attachment) {
        if (rp->depth_stencil_attachment.view->release_internal()) {
            free(rp->depth_stencil_attachment.view);
        }
    }

    m_render_passes.free(rp);
}

void WGPUDeviceImpl::free(WGPUComputePassEncoder cp) {
    m_compute_passes.free(cp);
}

void WGPUDeviceImpl::free(WGPUCommandBuffer buffer) {
    // WGPUCommandBuffer is special - it doesn't get destroyed right away, but instead added to a
    // pending list, and will be freed once its safe to do so.
    get_thread_local_state()->command_pool.free_command_buffer(buffer,
                                                               buffer->submitted_timeline_value);
}

void WGPUDeviceImpl::free(WGPUBindGroup) {}
void WGPUDeviceImpl::free(WGPUQuerySet) {}

uint32_t WGPUDeviceImpl::get_queue_family() const {
    return m_adapter->queue_family;
}

// MARK: Thread local state

WGPUDeviceImpl::ThreadLocalState::ThreadLocalState(const loon::gpu::Allocator& alloc,
                                                   WGPUDeviceImpl*             device) :
    allocator{alloc},
    arena_memory{allocator.alloc(kArenaSize)},
    arena(arena_memory.ptr, arena_memory.len),
    command_pool(device, &arena) {
    error_stack.initialize(device->m_error_callback);
}

WGPUDeviceImpl::ThreadLocalState* WGPUDeviceImpl::get_thread_local_state() {
    auto state = reinterpret_cast<ThreadLocalState*>(loon::gpu::tls_get_data(m_tls_key));
    if (state == nullptr) {
        state = new ThreadLocalState(m_allocator, this);
        loon::gpu::tls_set_data(m_tls_key, state);
    }
    return state;
}

loon::gpu::Arena* WGPUDeviceImpl::get_thread_local_arena() {
    return &get_thread_local_state()->arena;
}

const char* WGPUDeviceImpl::get_temp_null_terminated_string(WGPUStringView msg) {
    if (msg.length != WGPU_STRLEN) {
        char* str = (char*)get_thread_local_arena()->alloc(msg.length + 1);
        if (str) {
            memcpy(str, msg.data, msg.length);
            str[msg.length] = '\0';
        }
        return str;
    } else {
        return msg.data;
    }
}

void WGPUDeviceImpl::free_temp_string(const char* str) {
    auto arena = get_thread_local_arena();
    if (str && arena->owns(str)) { arena->free(str, strlen(str) + 1); }
}

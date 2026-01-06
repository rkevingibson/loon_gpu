#include "objects.h"

#include <vulkan/vulkan_core.h>

#include "device.h"
#include "instance.h"
#include "validation.h"
#include "vma_usage.h"
#include "wgpu_to_vk.h"


// MARK: Adapter

WGPUAdapterImpl::~WGPUAdapterImpl() {}

void WGPUAdapterImpl::add_ref() {
    refcount.add();
}

void WGPUAdapterImpl::release() {
    if (refcount.release() && can_destroy()) { instance->free_adapter(this); }
}

bool WGPUAdapterImpl::can_destroy() {
    return refcount.count() == 0 && loon::gpu::atomic_load(&device_count) == 0;
}

WGPUFuture WGPUAdapterImpl::request_device(WGPU_NULLABLE WGPUDeviceDescriptor const* descriptor,
                                           WGPURequestDeviceCallbackInfo             callbackInfo) {
    auto device_alloc = instance->get_allocator().alloc(sizeof(WGPUDeviceImpl));
    auto device       = ::new (device_alloc.ptr) WGPUDeviceImpl;
    auto status       = device->initialize(this, descriptor);
    if (status != WGPURequestDeviceStatus_Success) {
        device->~WGPUDeviceImpl();
        device = nullptr;
    }

    loon::gpu::CallbackData* cb;
    WGPUFuture               future = instance->create_future(&cb);
    *cb                  = {
                         .callback       = reinterpret_cast<WGPUProc>(callbackInfo.callback),
                         .mode           = callbackInfo.mode,
                         .userdata1      = callbackInfo.userdata1,
                         .userdata2      = callbackInfo.userdata2,
                         .message        = WGPU_STRING_VIEW_INIT,
                         .type           = loon::gpu::CallbackType::RequestDevice,
                         .request_device = {.status = status, .device = device,},
    };

    instance->set_future_ready(future);
    return future;
}

void WGPUAdapterImpl::free_device(WGPUDevice device) {
    device->~WGPUDeviceImpl();
    instance->get_allocator().free({.ptr = device, .len = sizeof(WGPUDeviceImpl)});
    loon::gpu::atomic_fetch_add(&device_count, -1);
    if (can_destroy()) { instance->free_adapter(this); }
}

// MARK: Surface

WGPUSurfaceImpl::~WGPUSurfaceImpl() {
    if (is_configured()) {
        device->vk_api.vkDestroySwapchainKHR(device->vk_device, vk_swapchain, nullptr);
    }
    vkDestroySurfaceKHR(instance->get_vk_instance(), vk_surface, nullptr);
}

void WGPUSurfaceImpl::add_ref() {
    refcount.add();
}

void WGPUSurfaceImpl::release() {
    if (refcount.release()) { instance->free_surface(this); }
}

void WGPUSurfaceImpl::configure(WGPUSurfaceConfiguration const* config) {
    if (!loon::gpu::validate(this, config)) {
        return;  // No way of error reporting for this function?
    }

    // Create the VkSwapchain based on the configuration

    VkPhysicalDevice         physical_device = config->device->m_adapter->vk_physical_device;
    VkSurfaceCapabilitiesKHR vk_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, vk_surface, &vk_capabilities);

    uint32_t image_count = vk_capabilities.minImageCount + 1;
    if (vk_capabilities.maxImageCount > 0 && image_count > vk_capabilities.maxImageCount) {
        image_count = vk_capabilities.maxImageCount;
    }

    const auto extent = VkExtent2D{.width = config->width, .height = config->height};

    VkSwapchainCreateInfoKHR swapchain_info{
        .sType         = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext         = nullptr,
        .flags         = 0,
        .surface       = vk_surface,
        .minImageCount = image_count,
        .imageFormat   = loon::gpu::bridge(config->format),
        .imageColorSpace
        = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,  // TODO: Colorspace support in webgpu?
        .imageExtent           = extent,
        .imageArrayLayers      = 1,
        .imageUsage            = loon::gpu::bridge_usage_flags(config->usage),
        .imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE,  // We only support one queue.
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .preTransform          = vk_capabilities.currentTransform,
        .compositeAlpha        = loon::gpu::bridge_composite_alpha_mode(config->alphaMode),
        .presentMode           = loon::gpu::bridge(config->presentMode),
        .clipped               = true,
        .oldSwapchain          = vk_swapchain,
    };

    device             = config->device;
    const auto& vk_api = device->vk_api;

    VkSwapchainKHR swapchain;
    WGPU_VK_CHECK(
        vkCreateSwapchainKHR(config->device->vk_device, &swapchain_info, nullptr, &swapchain));

    image_count = 0;
    WGPU_VK_CHECK(
        vkGetSwapchainImagesKHR(config->device->vk_device, swapchain, &image_count, nullptr));

    if (image_count > WGPUSurfaceImpl::kMaxSwapchainImages) {
        config->device->error(WGPUErrorType_Internal, "Swapchain creating too many images"_wsv);
        return;
    }

    loon::gpu::Stack<VkImage, WGPUSurfaceImpl::kMaxSwapchainImages> images;
    images.resize(image_count);
    WGPU_VK_CHECK(
        vkGetSwapchainImagesKHR(config->device->vk_device, swapchain, &image_count, images.data()));

    loon::gpu::Stack<WGPUTexture, WGPUSurfaceImpl::kMaxSwapchainImages> textures;
    for (size_t i = 0; i < images.size(); ++i) {
        auto texture      = config->device->m_textures.make(config->device, "Swapchain image"_wsv);
        texture->vk_image = images[i];
        texture->width    = config->width;
        texture->height   = config->height;
        texture->depth_or_array_layers = 1;
        texture->mip_level_count       = 1;
        texture->sample_count          = 1;
        texture->dimension             = WGPUTextureDimension_2D;
        texture->format                = config->format;
        texture->usages                = config->usage;
        texture->is_surface_image      = true;

        texture->add_ref_internal();
        textures.push(texture);
    }

    // TODO: What to do with viewFormats?
    device           = config->device;
    vk_swapchain     = swapchain;
    swapchain_images = textures;
    swapchain_format = loon::gpu::bridge(config->format);
    swapchain_extent = extent;
    return;
}

void WGPUSurfaceImpl::unconfigure() {
    if (is_configured()) {
        device->vk_api.vkDestroySwapchainKHR(device->vk_device, vk_swapchain, nullptr);
        device       = nullptr;
        vk_swapchain = VK_NULL_HANDLE;
        swapchain_images.resize(0);
        current_frame = nullptr;
    }
}

WGPUStatus WGPUSurfaceImpl::get_capabilities(WGPUAdapter              adapter,
                                             WGPUSurfaceCapabilities* capabilities) {
    if (!loon::gpu::validate(this, adapter, capabilities)) { return WGPUStatus_Error; }

    // Surface formats:
    uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(adapter->vk_physical_device,
                                         vk_surface,
                                         &format_count,
                                         nullptr);

    // TODO: an instance-specific arena would be useful for these.
    auto format_block
        = adapter->instance->get_allocator().alloc(sizeof(VkSurfaceFormatKHR) * format_count);
    auto* vk_formats = reinterpret_cast<VkSurfaceFormatKHR*>(format_block.ptr);
    vkGetPhysicalDeviceSurfaceFormatsKHR(adapter->vk_physical_device,
                                         vk_surface,
                                         &format_count,
                                         vk_formats);

    // Present modes:
    uint32_t present_mode_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(adapter->vk_physical_device,
                                              vk_surface,
                                              &present_mode_count,
                                              nullptr);
    auto present_mode_block
        = adapter->instance->get_allocator().alloc(sizeof(VkPresentModeKHR) * present_mode_count);
    auto* vk_modes = reinterpret_cast<VkPresentModeKHR*>(present_mode_block.ptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(adapter->vk_physical_device,
                                              vk_surface,
                                              &present_mode_count,
                                              vk_modes);

    VkSurfaceCapabilitiesKHR vk_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(adapter->vk_physical_device,
                                              vk_surface,
                                              &vk_capabilities);
    const auto alpha_modes
        = loon::gpu::bridge_composite_alpha_mode(vk_capabilities.supportedCompositeAlpha);

    // Need to copy to output:
    const auto output_alloc_size = sizeof(loon::gpu::Allocator)
                                   + sizeof(WGPUTextureFormat) * format_count
                                   + sizeof(WGPUPresentMode) * present_mode_count
                                   + sizeof(WGPUCompositeAlphaMode) * alpha_modes.size();
    auto alloc_block = adapter->instance->get_allocator().alloc(output_alloc_size);
    // This is an odd line of code - basically copying the allocator into its own allocation, so
    // that it can be freed properly later.
    auto allocator_ptr
        = ::new (alloc_block.ptr) loon::gpu::Allocator(adapter->instance->get_allocator());
    auto formats = reinterpret_cast<WGPUTextureFormat*>(&allocator_ptr[1]);
    for (size_t i = 0; i < format_count; ++i) {
        formats[i] = loon::gpu::bridge(vk_formats[i].format);
    }
    auto present_modes = reinterpret_cast<WGPUPresentMode*>(&formats[format_count]);
    for (size_t i = 0; i < present_mode_count; ++i) {
        present_modes[i] = loon::gpu::bridge(vk_modes[i]);
    }
    auto alpha_modes_ptr
        = reinterpret_cast<WGPUCompositeAlphaMode*>(&present_modes[present_mode_count]);
    for (size_t i = 0; i < alpha_modes.size(); ++i) { alpha_modes_ptr[i] = alpha_modes[i]; }

    adapter->instance->get_allocator().free(present_mode_block);
    adapter->instance->get_allocator().free(format_block);

    capabilities->usages      = loon::gpu::bridge_usage_flags(vk_capabilities.supportedUsageFlags);
    capabilities->formatCount = format_count;
    capabilities->formats     = formats;
    capabilities->presentModeCount = present_mode_count;
    capabilities->presentModes     = present_modes;
    capabilities->alphaModeCount   = alpha_modes.size();
    capabilities->alphaModes       = alpha_modes_ptr;

    return WGPUStatus_Success;
}

void WGPUSurfaceImpl::get_current_texture(WGPUSurfaceTexture* surface_texture) {
    if (!is_configured() || current_frame != nullptr) {
        surface_texture->status = WGPUSurfaceGetCurrentTextureStatus_Error;
        return;
    }

    // Synchronization: need to get the device queue, and wait on the timeline semaphore.
    const auto& vk_api            = device->vk_api;
    auto&       queue             = device->queue;
    VkSemaphore acquire_semaphore = queue.wait_for_current_swapchain_image();

    uint32_t image_idx = 0;
    VkResult result    = vk_api.vkAcquireNextImageKHR(device->vk_device,
                                                   vk_swapchain,
                                                   UINT64_MAX,
                                                   acquire_semaphore,
                                                   VK_NULL_HANDLE,
                                                   &image_idx);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        surface_texture->status = WGPUSurfaceGetCurrentTextureStatus_Outdated;
    } else if (result == VK_SUBOPTIMAL_KHR) {
        surface_texture->status = WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal;
    } else if (result == VK_SUCCESS) {
        surface_texture->status = WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal;
    } else {
        device->vulkan_error(result, __FILE__, __LINE__);
        surface_texture->status = WGPUSurfaceGetCurrentTextureStatus_Error;
    }

    auto tex = swapchain_images[image_idx];
    tex->add_ref();
    surface_texture->texture = tex;
    current_frame            = tex;
    current_idx              = image_idx;
}

WGPUStatus WGPUSurfaceImpl::present() {
    if (!is_configured() || current_frame == nullptr) { return WGPUStatus_Error; }

    auto       presenting_texture = swapchain_images[current_idx];
    WGPUStatus status = device->queue.present(presenting_texture, vk_swapchain, current_idx);
    current_frame     = nullptr;
    return status;
}

// MARK: ObjectBase

WGPUObjectBase::WGPUObjectBase(WGPUDevice device, WGPUStringView l) :
    device{device}, label(device->get_allocator(), l) {}

void WGPUObjectBase::set_label(WGPUStringView str) {
    label.set(device->get_allocator(), str);
}

// MARK: ShaderModule
WGPUShaderModuleImpl::~WGPUShaderModuleImpl() {
    if (vk_module) { device->vk_api.vkDestroyShaderModule(device->vk_device, vk_module, nullptr); }
}

void swap(WGPUShaderModuleImpl& a, WGPUShaderModuleImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(a.vk_module, b.vk_module);
}

// MARK: PipelineBase
WGPUPipelineBaseImpl::~WGPUPipelineBaseImpl() {
    if (pipeline_layout) { loon::gpu::release_internal(pipeline_layout); }
}

void swap(WGPUPipelineBaseImpl& a, WGPUPipelineBaseImpl& b) {
    using std::swap;
    swap(a.pipeline_layout, b.pipeline_layout);
}

WGPUBindGroupLayout WGPUPipelineBaseImpl::get_bind_group_layout(uint32_t index) {
    return pipeline_layout->bind_group_layouts[index];
}

// MARK: RenderPipeline
WGPURenderPipelineImpl::~WGPURenderPipelineImpl() {
    if (vk_pipeline) { device->vk_api.vkDestroyPipeline(device->vk_device, vk_pipeline, nullptr); }
}

void swap(WGPURenderPipelineImpl& a, WGPURenderPipelineImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(static_cast<WGPUPipelineBaseImpl&>(a), static_cast<WGPUPipelineBaseImpl&>(b));
    swap(a.vk_pipeline, b.vk_pipeline);
}


// MARK: ComputePipeline
WGPUComputePipelineImpl::~WGPUComputePipelineImpl() {
    if (vk_pipeline) { device->vk_api.vkDestroyPipeline(device->vk_device, vk_pipeline, nullptr); }
}

void swap(WGPUComputePipelineImpl& a, WGPUComputePipelineImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(static_cast<WGPUPipelineBaseImpl&>(a), static_cast<WGPUPipelineBaseImpl&>(b));
    swap(a.vk_pipeline, b.vk_pipeline);
}

// MARK: BindGroupLayout
WGPUBindGroupLayoutImpl::~WGPUBindGroupLayoutImpl() {
    if (vk_set_layout) {
        device->vk_api.vkDestroyDescriptorSetLayout(device->vk_device, vk_set_layout, nullptr);
    }
}

bool compatible(const WGPUBindGroupLayoutImpl& a, const WGPUBindGroupLayoutImpl& b) {
    // https://www.w3.org/TR/webgpu/#bind-group-compatibility
    // Note: because we keep the entries array sorted, compatibility is just if the entries vectors
    // are equivalent
    return a.exclusive_pipeline == b.exclusive_pipeline && a.entries.size() == b.entries.size()
           && std::equal(a.entries.begin(), a.entries.end(), b.entries.begin());
}

bool operator==(const WGPUBindGroupLayoutImpl::LayoutEntry& a,
                const WGPUBindGroupLayoutImpl::LayoutEntry& b) {
    if (a.binding == b.binding && a.entry_type == b.entry_type && a.visibility && b.visibility) {
        switch (a.entry_type) {
            case WGPUBindGroupLayoutImpl::BindingType::kBuffer:
                return a.buffer.type == b.buffer.type
                       && a.buffer.hasDynamicOffset == b.buffer.hasDynamicOffset
                       && a.buffer.minBindingSize == b.buffer.minBindingSize;
            case WGPUBindGroupLayoutImpl::BindingType::kSampler:
                return a.sampler.type == b.sampler.type;
            case WGPUBindGroupLayoutImpl::BindingType::kTexture:
                return a.texture.sampleType == b.texture.sampleType
                       && a.texture.viewDimension == b.texture.viewDimension
                       && a.texture.multisampled == b.texture.multisampled;
            case WGPUBindGroupLayoutImpl::BindingType::kStorageTexture:
                return a.storage_texture.access == b.storage_texture.access
                       && a.storage_texture.format == b.storage_texture.format
                       && a.storage_texture.viewDimension == b.storage_texture.viewDimension;
        }
    }
    return false;
}

loon::gpu::ResourceUsage WGPUBindGroupLayoutImpl::LayoutEntry::internal_usage() const {
    using loon::gpu::ResourceUsage;
    switch (entry_type) {
        case BindingType::kBuffer: {
            switch (buffer.type) {
                case WGPUBufferBindingType_Uniform: return ResourceUsage::kUsageConstant;
                case WGPUBufferBindingType_Storage: return ResourceUsage::kUsageStorage;
                case WGPUBufferBindingType_ReadOnlyStorage: return ResourceUsage::kUsageStorageRead;
                default: break; ;
            }
        }
        case BindingType::kSampler:
        case BindingType::kTexture: return ResourceUsage::kUsageConstant;
        case BindingType::kStorageTexture: {
            switch (storage_texture.access) {
                case WGPUStorageTextureAccess_WriteOnly: return ResourceUsage::kUsageStorage;
                case WGPUStorageTextureAccess_ReadOnly: return ResourceUsage::kUsageStorageRead;
                case WGPUStorageTextureAccess_ReadWrite: return ResourceUsage::kUsageStorage;
                default: break;
            }
        };
    }
    return ResourceUsage::kUsageUndefined;
}


VkDescriptorType WGPUBindGroupLayoutImpl::LayoutEntry::descriptor_type() const {
    switch (entry_type) {
        case BindingType::kBuffer:
            switch (buffer.type) {
                case WGPUBufferBindingType_Uniform:
                    return buffer.hasDynamicOffset ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                                                   : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                case WGPUBufferBindingType_Storage: [[fallthrough]];
                case WGPUBufferBindingType_ReadOnlyStorage:
                    return buffer.hasDynamicOffset ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                                                   : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                default: break;
            }
        case BindingType::kSampler: return VK_DESCRIPTOR_TYPE_SAMPLER;
        case BindingType::kTexture: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case BindingType::kStorageTexture: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    assert(false && "Invalid descriptor type");
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

void swap(WGPUBindGroupLayoutImpl& a, WGPUBindGroupLayoutImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(a.vk_set_layout, b.vk_set_layout);
}

// MARK: BindGroup

WGPUBindGroupImpl::~WGPUBindGroupImpl() {
    if (descriptor.set != VK_NULL_HANDLE) { descriptor.free(device); }
}

void swap(WGPUBindGroupImpl& a, WGPUBindGroupImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(a.layout, b.layout);
    swap(a.entries, b.entries);
    swap(a.used_resources, b.used_resources);
    swap(a.descriptor, b.descriptor);
}


// MARK: PipelineLayout
WGPUPipelineLayoutImpl::~WGPUPipelineLayoutImpl() {
    if (vk_layout) {
        device->vk_api.vkDestroyPipelineLayout(device->vk_device, vk_layout, nullptr);
    }

    for (auto layout : bind_group_layouts) {
        if (layout) { loon::gpu::release_internal(layout); }
    }
}

void swap(WGPUPipelineLayoutImpl& a, WGPUPipelineLayoutImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(a.vk_layout, b.vk_layout);
}

// MARK: Texture

WGPUTextureImpl::~WGPUTextureImpl() {
    if (vk_image && !is_surface_image) {
        device->vk_api.vkDestroyImage(device->vk_device, vk_image, nullptr);
    }
}

void swap(WGPUTextureImpl& a, WGPUTextureImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(a.vk_image, b.vk_image);
    swap(a.width, b.width);
    swap(a.height, b.height);
    swap(a.depth_or_array_layers, b.depth_or_array_layers);
    swap(a.mip_level_count, b.mip_level_count);
    swap(a.sample_count, b.sample_count);
    swap(a.dimension, b.dimension);
    swap(a.format, b.format);
    swap(a.usages, b.usages);
    swap(a.is_surface_image, b.is_surface_image);
    swap(a.last_submitted_usage, b.last_submitted_usage);
}

WGPUTextureView WGPUTextureImpl::create_view(WGPUTextureViewDescriptor const* descriptor) {
    const auto vk_format = loon::gpu::bridge(descriptor->format);
    VkImageViewCreateInfo create_info{.sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                      .pNext      = nullptr,
                                      .flags      = 0,
                                      .image      = vk_image,
                                      .viewType   = loon::gpu::bridge(descriptor->dimension),
                                      .format     = vk_format,
                                      .components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                                                     VK_COMPONENT_SWIZZLE_IDENTITY,
                                                     VK_COMPONENT_SWIZZLE_IDENTITY,
                                                     VK_COMPONENT_SWIZZLE_IDENTITY,},
                                      .subresourceRange = {
                                        .aspectMask = loon::gpu::bridge(descriptor->aspect),
                                        .baseMipLevel = descriptor->baseMipLevel,
                                        .levelCount = descriptor->mipLevelCount,
                                        .baseArrayLayer = descriptor->baseArrayLayer,
                                        .layerCount = descriptor->arrayLayerCount,
                                      },};
    VkImageView image_view = VK_NULL_HANDLE;
    auto&       vk_api     = device->vk_api;
    WGPU_VK_CHECK(vkCreateImageView(device->vk_device, &create_info, nullptr, &image_view));

    auto result = device->m_texture_views.make(device, descriptor->label);
    add_ref_internal();
    result->texture       = this;
    result->descriptor    = *descriptor;
    result->render_extent = physical_extent(descriptor->baseMipLevel);
    result->vk_image_view = image_view;
    result->vk_format     = vk_format;

    return loon::gpu::return_with_ownership(result);
}

WGPUExtent3D WGPUTextureImpl::logical_extent(uint32_t mip_level) const {
    WGPUExtent3D extent{};
    switch (dimension) {
        case WGPUTextureDimension_1D:
            extent.width              = std::max(width >> mip_level, 1u);
            extent.height             = 1;
            extent.depthOrArrayLayers = 1;
            break;
        case WGPUTextureDimension_2D:
            extent.width              = std::max(width >> mip_level, 1u);
            extent.height             = std::max(height >> mip_level, 1u);
            extent.depthOrArrayLayers = depth_or_array_layers;
            break;
        case WGPUTextureDimension_3D:
            extent.width              = std::max(width >> mip_level, 1u);
            extent.height             = std::max(height >> mip_level, 1u);
            extent.depthOrArrayLayers = std::max(depth_or_array_layers >> mip_level, 1u);
            break;
        default: assert(false); break;
    }
    return extent;
}

WGPUExtent3D WGPUTextureImpl::physical_extent(uint32_t mip_level) const {
    WGPUExtent3D extent{};
    const auto   l_extent         = logical_extent(mip_level);
    const auto   texel_blk_width  = loon::gpu::texel_block_width(format);
    const auto   texel_blk_height = loon::gpu::texel_block_height(format);
    const auto   round_up         = [](uint32_t x, uint32_t m) { return ((x + m - 1) / m) * m; };
    switch (dimension) {
        case WGPUTextureDimension_1D:
            extent = {
                .width              = round_up(l_extent.width, texel_blk_width),
                .height             = 1,
                .depthOrArrayLayers = 1,
            };
            break;
        case WGPUTextureDimension_2D:
        case WGPUTextureDimension_3D:
            extent = {
                .width              = round_up(l_extent.width, texel_blk_width),
                .height             = round_up(l_extent.height, texel_blk_height),
                .depthOrArrayLayers = l_extent.depthOrArrayLayers,
            };
            break;
        default: break;
    }
    return extent;
}

// MARK: TextureView

WGPUTextureViewImpl::~WGPUTextureViewImpl() {
    if (vk_image_view) {
        device->vk_api.vkDestroyImageView(device->vk_device, vk_image_view, nullptr);
    }
    if (texture != nullptr) {
        if (texture->release_internal()) { device->free(texture); }
    }
}

void swap(WGPUTextureViewImpl& a, WGPUTextureViewImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(a.texture, b.texture);
    swap(a.descriptor, b.descriptor);
    swap(a.render_extent, b.render_extent);
    swap(a.vk_image_view, b.vk_image_view);
    swap(a.vk_format, b.vk_format);
}

// MARK: Buffer

WGPUBufferImpl::~WGPUBufferImpl() {
    if (vk_buffer) { vmaDestroyBuffer(device->vk_allocator, vk_buffer, vk_allocation); }
}

void swap(WGPUBufferImpl& a, WGPUBufferImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(a.internal_state, b.internal_state);
    swap(a.size, b.size);
    swap(a.usage, b.usage);
    swap(a.map_state, b.map_state);
    swap(a.vk_buffer, b.vk_buffer);
    swap(a.vk_allocation, b.vk_allocation);
    swap(a.mapping, b.mapping);
}

WGPUFuture WGPUBufferImpl::map_async(WGPUMapMode               mode,
                                     size_t                    offset,
                                     size_t                    size,
                                     WGPUBufferMapCallbackInfo callbackInfo) {
    // TODO:
    return WGPU_FUTURE_INIT;
}

void* WGPUBufferImpl::get_mapped_range(size_t offset, size_t size) {
    if (!loon::gpu::validate_buffer_get_mapped_range(this, offset, size)) { return nullptr; }
    return ((char*)mapping.ptr) + offset;
}

void WGPUBufferImpl::unmap() {
    if (mapping.ptr == nullptr) { return; }
    if (internal_state == InternalState::Init) {
        // Mapped using a staging buffer
        mapping.staging_buffer.unmap(this, mapping.offset, mapping.size);
    } else if (internal_state == InternalState::Unavailable) {
        vmaUnmapMemory(device->vk_allocator, vk_allocation);
        vmaFlushAllocation(device->vk_allocator, vk_allocation, mapping.offset, mapping.size);
    }
    mapping        = {0};
    internal_state = InternalState::Available;
}

void WGPUBufferImpl::destroy() {
    // See https://www.w3.org/TR/webgpu/#buffer-destruction
    unmap();

    const auto& vk_api = device->vk_api;

    // TODO: Don't want to do this while any operations are pending on it. Need to defer this until
    // later.
    // vk_api.vkDestroyBuffer(device->vk_device, vk_buffer, nullptr);
}
#pragma once

#include <cstddef>

#include "platform_utils.h"
#include "utilities.h"
#include "validation.h"
#include "vma_usage.h"
#include "volk.h"
#include "webgpu/webgpu.h"

VK_DEFINE_HANDLE(VmaAllocation);

namespace webgpu {
struct CommandsMixin;

template <typename T>
void release(T* ptr) {
    if (ptr->release()) { ptr->device->free(ptr); }
}

template <typename T>
void release_internal(T* ptr) {
    if (ptr->release_internal()) { ptr->device->free(ptr); }
}

}  // namespace webgpu

struct WGPUAdapterImpl {
    ~WGPUAdapterImpl();
    void add_ref();
    void release();

    WGPUFuture request_device(WGPU_NULLABLE WGPUDeviceDescriptor const* descriptor,
                              WGPURequestDeviceCallbackInfo             callbackInfo);
    void       free_device(WGPUDevice device);

    WGPUInstance     instance           = nullptr;
    VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;
    uint32_t         queue_family       = 0;

    uint32_t        feature_count = 0;
    WGPUFeatureName supported_features[WGPUFeatureName_Subgroups];

    WGPULimits limits{};

    char            device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]{'\0'};
    WGPUAdapterType adapter_type      = WGPUAdapterType_Unknown;
    uint32_t        vendor_id         = 0;
    uint32_t        device_id         = 0;
    uint32_t        subgroup_min_size = 0;
    uint32_t        subgroup_max_size = 0;

    webgpu::Label label;

   private:
    bool can_destroy();

    webgpu::ReferenceCount refcount;
    int64_t                device_count = 0;
};

struct WGPUSurfaceImpl {
    ~WGPUSurfaceImpl();
    void add_ref();
    void release();

    void       configure(WGPUSurfaceConfiguration const* config);
    void       unconfigure();
    WGPUStatus get_capabilities(WGPUAdapter adapter, WGPUSurfaceCapabilities* capabilities);

    void       get_current_texture(WGPUSurfaceTexture* surfaceTexture);
    WGPUStatus present();

    WGPUInstance instance   = nullptr;
    VkSurfaceKHR vk_surface = VK_NULL_HANDLE;

    // Only set when configured
    WGPUDevice                                      device              = nullptr;
    VkSwapchainKHR                                  vk_swapchain        = VK_NULL_HANDLE;
    static constexpr uint32_t                       kMaxSwapchainImages = 8;
    webgpu::Stack<WGPUTexture, kMaxSwapchainImages> swapchain_images;
    VkFormat                                        swapchain_format = VK_FORMAT_UNDEFINED;
    VkExtent2D                                      swapchain_extent = {0, 0};

    WGPUTexture current_frame = nullptr;
    uint32_t    current_idx   = 0;

    webgpu::Label label;

    bool is_configured() const { return device != nullptr; }

   private:
    webgpu::ReferenceCount refcount;
};

struct WGPUObjectBase {
    WGPUDevice    device = nullptr;
    webgpu::Label label;
    bool          valid = true;
    int64_t       reference_count{0};

    WGPUObjectBase() = default;
    WGPUObjectBase(WGPUDevice device, WGPUStringView l = WGPU_STRING_VIEW_INIT);

    static constexpr int64_t kInternalReferenceIncrement = 1ll << 32;
    static constexpr int64_t kExternalReferenceCountMask = kInternalReferenceIncrement - 1;

    void               add_ref() { webgpu::atomic_fetch_add(&reference_count, 1); }
    [[nodiscard]] bool release() {
        const auto old_count = webgpu::atomic_fetch_add(&reference_count, -1);
        assert((old_count & kExternalReferenceCountMask) != 0);
        return (old_count - 1) == 0;
    }

    [[nodiscard]] bool free_external_refs() {
        auto    expected = webgpu::atomic_load(&reference_count);
        int64_t desired  = expected & ~kExternalReferenceCountMask;  // Zero out external references
        while (!webgpu::atomic_compare_exchange(&reference_count, &desired, expected)) {
            desired = expected & ~kExternalReferenceCountMask;  // Zero out external references
        }
        return desired == 0;
    }
    void add_ref_internal() {
        webgpu::atomic_fetch_add(&reference_count, kInternalReferenceIncrement);
    }
    [[nodiscard]] bool release_internal() {
        const auto old_count
            = webgpu::atomic_fetch_add(&reference_count, -kInternalReferenceIncrement);
        assert((old_count >> 32) != 0);
        return ((old_count - kInternalReferenceIncrement) == 0);
    }

    void set_label(WGPUStringView str);

    friend void swap(WGPUObjectBase& a, WGPUObjectBase& b) {
        using std::swap;
        swap(a.device, b.device);
        swap(a.label, b.label);
        swap(a.valid, b.valid);
        swap(a.reference_count, b.reference_count);
    }
};

#define WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(Type)                                                 \
    Type() = default;                                                                              \
    Type(WGPUDevice device, WGPUStringView label) : WGPUObjectBase(device, label) {}               \
    ~Type();                                                                                       \
    Type(Type&& other) : Type() {                                                                  \
        swap(*this, other);                                                                        \
    }                                                                                              \
    Type& operator=(Type&& other) {                                                                \
        swap(*this, other);                                                                        \
        return *this;                                                                              \
    }                                                                                              \
    friend void swap(Type& a, Type& b);


struct WGPUShaderModuleImpl : WGPUObjectBase {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPUShaderModuleImpl);

    VkShaderModule vk_module = VK_NULL_HANDLE;
};

struct WGPUPipelineBaseImpl {
    WGPUPipelineBaseImpl() = default;
    ~WGPUPipelineBaseImpl();
    WGPUPipelineBaseImpl(WGPUPipelineBaseImpl&& other) : WGPUPipelineBaseImpl() {
        swap(*this, other);
    }
    WGPUPipelineBaseImpl& operator=(WGPUPipelineBaseImpl&& other) {
        swap(*this, other);
        return *this;
    }
    friend void swap(WGPUPipelineBaseImpl& a, WGPUPipelineBaseImpl& b);

    WGPUBindGroupLayout get_bind_group_layout(uint32_t index);

    WGPUPipelineLayout pipeline_layout = nullptr;
};
using WGPUPipelineBase = WGPUPipelineBaseImpl*;

struct WGPURenderPipelineImpl : WGPUObjectBase, WGPUPipelineBaseImpl {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPURenderPipelineImpl);

    VkPipeline vk_pipeline = VK_NULL_HANDLE;
};

struct WGPUComputePipelineImpl : WGPUObjectBase, WGPUPipelineBaseImpl {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPUComputePipelineImpl);

    VkPipeline vk_pipeline = VK_NULL_HANDLE;
};

struct WGPUBindGroupLayoutImpl : WGPUObjectBase {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPUBindGroupLayoutImpl);

    friend bool compatible(const WGPUBindGroupLayoutImpl& a, const WGPUBindGroupLayoutImpl& b);

    enum class BindingType : uint8_t {
        kBuffer,
        kSampler,
        kTexture,
        kStorageTexture,
    };

    struct LayoutEntry {
        uint16_t        binding;
        BindingType     entry_type;
        WGPUShaderStage visibility;
        union {
            WGPUBufferBindingLayout         buffer;
            WGPUSamplerBindingLayout        sampler;
            WGPUTextureBindingLayout        texture;
            WGPUStorageTextureBindingLayout storage_texture;
        };

        friend bool           operator==(const LayoutEntry& a, const LayoutEntry& b);
        webgpu::ResourceUsage internal_usage() const;

        VkDescriptorType descriptor_type() const;
    };

    webgpu::Vector<LayoutEntry>           entries;
    uint32_t                              dynamic_offset_count = 0;
    WGPUPipelineBase                      exclusive_pipeline   = nullptr;
    VkDescriptorSetLayout                 vk_set_layout        = VK_NULL_HANDLE;
    webgpu::HashTable<uint16_t, uint16_t> entry_map;  // Map from binding index to index in entries
};

struct WGPUBindGroupImpl : WGPUObjectBase {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPUBindGroupImpl);

    WGPUBindGroupLayout                                  layout = nullptr;
    webgpu::Vector<WGPUBindGroupEntry>                   entries;
    webgpu::UsageScope                                   used_resources;
    webgpu::DescriptorSetAllocator::DescriptorAllocation descriptor;
    VkDescriptorPool                                     vk_pool = VK_NULL_HANDLE;
};

struct WGPUPipelineLayoutImpl : WGPUObjectBase {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPUPipelineLayoutImpl);

    webgpu::Stack<WGPUBindGroupLayout, webgpu::kMaxBindGroups> bind_group_layouts;
    VkPipelineLayout                                           vk_layout = VK_NULL_HANDLE;
};

struct WGPUTextureImpl : WGPUObjectBase {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPUTextureImpl);

    WGPUTextureView create_view(WGPUTextureViewDescriptor const* descriptor);

    WGPUExtent3D logical_extent(uint32_t mip_level) const;
    WGPUExtent3D physical_extent(uint32_t mip_level) const;

    VkImage               vk_image              = VK_NULL_HANDLE;
    uint32_t              width                 = 0;
    uint32_t              height                = 0;
    uint32_t              depth_or_array_layers = 0;
    uint32_t              mip_level_count       = 0;
    uint32_t              sample_count          = 0;
    WGPUTextureDimension  dimension             = WGPUTextureDimension_Undefined;
    WGPUTextureFormat     format                = WGPUTextureFormat_Undefined;
    WGPUTextureUsage      usages                = 0;
    bool                  is_surface_image      = false;
    webgpu::ResourceUsage last_submitted_usage  = webgpu::ResourceUsage::kUsageUndefined;
    // TODO: View formats array. Do I need it for anything other than validation?
};

struct WGPUTextureViewImpl : WGPUObjectBase {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPUTextureViewImpl);

    WGPUTexture               texture       = nullptr;
    WGPUTextureViewDescriptor descriptor    = {};
    WGPUExtent3D              render_extent = {};
    VkImageView               vk_image_view = VK_NULL_HANDLE;
    VkFormat                  vk_format     = VK_FORMAT_UNDEFINED;
};

struct WGPUBufferImpl : WGPUObjectBase {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPUBufferImpl);

    WGPUFuture map_async(WGPUMapMode               mode,
                         size_t                    offset,
                         size_t                    size,
                         WGPUBufferMapCallbackInfo callbackInfo);
    void*      get_mapped_range(size_t offset, size_t size);
    void       unmap();
    void       destroy();

    enum class InternalState {
        Available,
        Init,
        Unavailable,
        Destroyed,
    };

    struct ActiveBufferMapping {
        void*       ptr      = nullptr;
        WGPUMapMode map_mode = WGPUMapMode_None;
        uint64_t    offset   = 0;
        uint64_t    size     = 0;

        // Null unless internal_state == Init;
        webgpu::StagingBuffer staging_buffer = {};
    };

    InternalState      internal_state;
    uint64_t           size      = 0;
    WGPUBufferUsage    usage     = WGPUBufferUsage_None;
    WGPUBufferMapState map_state = WGPUBufferMapState_Unmapped;

    VkBuffer            vk_buffer          = VK_NULL_HANDLE;
    VmaAllocation       vk_allocation      = VK_NULL_HANDLE;
    VmaAllocationInfo   vk_allocation_info = {};
    ActiveBufferMapping mapping            = {};

    webgpu::ResourceUsage last_submitted_usage = webgpu::ResourceUsage::kUsageUndefined;
};

struct WGPUQuerySetImpl : WGPUObjectBase {};

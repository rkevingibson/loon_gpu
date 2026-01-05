#include "webgpu/webgpu.h"

#include <cassert>
#include <cstdio>
#include <cstring>

#include "commands.h"
#include "device.h"
#include "instance.h"
#include "objects.h"
#include "utilities.h"
#include "validation.h"
#include "volk.h"
#include "webgpu/webgpu_loon.h"
#include "webgpu/webgpu_procs.h"

#define WGPU_STRINGIFY_IMPL(x) #x
#define WGPU_STRINGIFY(x)      WGPU_STRINGIFY_IMPL(x)

// TODO: Fix this once we're done implementing
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"

// MARK: Global Functions

WGPUInstance wgpuCreateInstance(WGPU_NULLABLE WGPUInstanceDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    WGPULoonInstanceConfiguration* alloc_info = nullptr;
    if (descriptor->nextInChain
        && descriptor->nextInChain->sType
               == static_cast<WGPUSType>(WGPUSType_LoonInstanceConfiguration)) {
        alloc_info = reinterpret_cast<WGPULoonInstanceConfiguration*>(descriptor->nextInChain);
    }

    return WGPUInstanceImpl::create(alloc_info);
}

void wgpuGetInstanceFeatures(WGPUSupportedInstanceFeatures* features) {
    (void)features;
}

void wgpuSupportedInstanceFeaturesFreeMembers(
    WGPUSupportedInstanceFeatures supportedInstanceFeatures) {
    (void)supportedInstanceFeatures;
}

WGPUStatus wgpuGetInstanceLimits(WGPUInstanceLimits* limits) {
    (void)limits;
    return WGPUStatus_Error;
}

WGPUBool wgpuHasInstanceFeature(WGPUInstanceFeatureName feature) {
    switch (feature) {
        case WGPUInstanceFeatureName_TimedWaitAny:
        case WGPUInstanceFeatureName_ShaderSourceSPIRV:
        case WGPUInstanceFeatureName_MultipleDevicesPerAdapter: return true;
        case WGPUInstanceFeatureName_Force32: break;
    }

    return false;
}

constexpr bool operator==(const WGPUStringView& a, const WGPUStringView& b) noexcept {
    if (a.data == nullptr || b.data == nullptr) { return a.data == b.data; }

    const size_t a_len = a.length == WGPU_STRLEN ? strlen(a.data) : a.length;
    const size_t b_len = b.length == WGPU_STRLEN ? strlen(b.data) : b.length;

    return a_len == b_len && memcmp(a.data, b.data, a_len) == 0;
}

WGPUProc wgpuGetProcAddress(WGPUStringView proc_name) {
    // Function names are sorted alphabetically to allow for binary search, but for now just
    // linearly search.
    struct ProcEntry {
        WGPUStringView name;
        WGPUProc       proc;
    };
    static const ProcEntry function_list[] = {
#define WGPU_FN_NAME_AS_SLICE(name) WGPU_STRCAT(WGPU_STRINGIFY(WGPU_STRCAT(wgpu, name)), _wsv)
#define X_WGPU_FUNCTION(_, x)                                                                      \
    {                                                                                              \
        .name = WGPU_FN_NAME_AS_SLICE(x),                                                          \
        .proc = (WGPUProc)WGPU_STRCAT(wgpu, x),                                                    \
    },
        WGPU_FUNCTION_LIST
#undef X_WGPU_FUNCTION
    };

    for (const auto& entry : function_list) {
        if (entry.name == proc_name) { return entry.proc; }
    }

    return nullptr;
}

void wgpuLoonGetFunctionTable(WGPULoonFunctionTable* table) {
    if (table != NULL) {
        *table = WGPULoonFunctionTable{
#define X_WGPU_FUNCTION(snake_name, pascal_name) .snake_name = WGPU_STRCAT(wgpu, pascal_name),
            WGPU_FUNCTION_LIST
#undef X_WGPU_FUNCTION
        };
    }
}

// MARK: Instance methods

/**
 * Creates a @ref WGPUSurface, see @ref Surface-Creation for more details.
 *
 * @param descriptor
 * The description of the @ref WGPUSurface to create.
 *
 * @returns
 * A new @ref WGPUSurface for this descriptor (or an error @ref WGPUSurface).
 * This value is @ref ReturnedWithOwnership.
 */
WGPUSurface wgpuInstanceCreateSurface(WGPUInstance                 instance,
                                      WGPUSurfaceDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return instance->create_surface(descriptor);
}

/**
 * Get the list of @ref WGPUWGSLLanguageFeatureName values supported by the instance.
 */
void wgpuInstanceGetWGSLLanguageFeatures(WGPUInstance, WGPUSupportedWGSLLanguageFeatures* features)
    WGPU_FUNCTION_ATTRIBUTE {
    if (!features) return;
    features->featureCount = 0;
    features->features     = nullptr;
}

WGPUBool wgpuInstanceHasWGSLLanguageFeature(WGPUInstance,
                                            WGPUWGSLLanguageFeatureName) WGPU_FUNCTION_ATTRIBUTE {
    return WGPUBool{false};
}

/**
 * Processes asynchronous events on this `WGPUInstance`, calling any callbacks for asynchronous
 * operations created with @ref WGPUCallbackMode_AllowProcessEvents.
 *
 * See @ref Process-Events for more information.
 */
void wgpuInstanceProcessEvents(WGPUInstance instance) WGPU_FUNCTION_ATTRIBUTE {
    instance->process_ready_futures();
}

/**
 * Wait for at least one WGPUFuture in `futures` to complete, and call callbacks of the respective
 * completed asynchronous operations.
 *
 * See @ref Wait-Any for more information.
 */
WGPUWaitStatus wgpuInstanceWaitAny(WGPUInstance                      instance,
                                   size_t                            futureCount,
                                   WGPU_NULLABLE WGPUFutureWaitInfo* futures,
                                   uint64_t timeoutNS) WGPU_FUNCTION_ATTRIBUTE {
    return instance->wait_on_futures(futureCount, futures, timeoutNS);
}



WGPUFuture wgpuInstanceRequestAdapter(WGPUInstance                                   instance,
                                      WGPU_NULLABLE WGPURequestAdapterOptions const* options,
                                      WGPURequestAdapterCallbackInfo                 callbackInfo)
    WGPU_FUNCTION_ATTRIBUTE {
    return instance->request_adapter(options, callbackInfo);
}

void wgpuInstanceAddRef(WGPUInstance instance) WGPU_FUNCTION_ATTRIBUTE {
    instance->add_ref();
}
void wgpuInstanceRelease(WGPUInstance instance) WGPU_FUNCTION_ATTRIBUTE {
    instance->release();
}

// MARK: Surface Methods

/**
 * \defgroup WGPUSurfaceMethods WGPUSurface methods
 * \brief Functions whose first argument has type WGPUSurface.
 *
 * @{
 */
/**
 * Configures parameters for rendering to `surface`.
 * Produces a @ref DeviceError for all content-timeline errors defined by the WebGPU specification.
 *
 * See @ref Surface-Configuration for more details.
 *
 * @param config
 * The new configuration to use.
 */
void wgpuSurfaceConfigure(WGPUSurface                     surface,
                          WGPUSurfaceConfiguration const* config) WGPU_FUNCTION_ATTRIBUTE {
    surface->configure(config);
}

/**
 * Provides information on how `adapter` is able to use `surface`.
 * See @ref Surface-Capabilities for more details.
 *
 * @param adapter
 * The @ref WGPUAdapter to get capabilities for presenting to this @ref WGPUSurface.
 *
 * @param capabilities
 * The structure to fill capabilities in.
 * It may contain memory allocations so @ref wgpuSurfaceCapabilitiesFreeMembers must be called to
 * avoid memory leaks. This parameter is @ref ReturnedWithOwnership.
 *
 * @returns
 * Indicates if there was an @ref OutStructChainError.
 */
WGPUStatus wgpuSurfaceGetCapabilities(WGPUSurface              surface,
                                      WGPUAdapter              adapter,
                                      WGPUSurfaceCapabilities* capabilities)
    WGPU_FUNCTION_ATTRIBUTE {
    return surface->get_capabilities(adapter, capabilities);
}
/**
 * Returns the @ref WGPUTexture to render to `surface` this frame along with metadata on the frame.
 * Returns `NULL` and @ref WGPUSurfaceGetCurrentTextureStatus_Error if the surface is not
 * configured.
 *
 * See @ref Surface-Presenting for more details.
 *
 * @param surfaceTexture
 * The structure to fill the @ref WGPUTexture and metadata in.
 */
void wgpuSurfaceGetCurrentTexture(WGPUSurface         surface,
                                  WGPUSurfaceTexture* surfaceTexture) WGPU_FUNCTION_ATTRIBUTE {
    surface->get_current_texture(surfaceTexture);
}

/**
 * Shows `surface`'s current texture to the user.
 * See @ref Surface-Presenting for more details.
 *
 * @returns
 * Returns @ref WGPUStatus_Error if the surface doesn't have a current texture.
 */
WGPUStatus wgpuSurfacePresent(WGPUSurface surface) WGPU_FUNCTION_ATTRIBUTE {
    return surface->present();
}
/**
 * Modifies the label used to refer to `surface`.
 *
 * @param label
 * The new label.
 */
void wgpuSurfaceSetLabel(WGPUSurface surface, WGPUStringView label) WGPU_FUNCTION_ATTRIBUTE {
    surface->label.set(surface->device->get_allocator(), label);
}
/**
 * Removes the configuration for `surface`.
 * See @ref Surface-Configuration for more details.
 */
void wgpuSurfaceUnconfigure(WGPUSurface surface) WGPU_FUNCTION_ATTRIBUTE {
    surface->unconfigure();
}

void wgpuSurfaceAddRef(WGPUSurface surface) WGPU_FUNCTION_ATTRIBUTE {
    surface->add_ref();
}

void wgpuSurfaceRelease(WGPUSurface surface) WGPU_FUNCTION_ATTRIBUTE {
    surface->release();
}
/** @} */



/**
 * \defgroup WGPUSurfaceCapabilitiesMethods WGPUSurfaceCapabilities methods
 * \brief Functions whose first argument has type WGPUSurfaceCapabilities.
 *
 * @{
 */
/**
 * Frees array members of WGPUSurfaceCapabilities which were allocated by the API.
 */
void wgpuSurfaceCapabilitiesFreeMembers(WGPUSurfaceCapabilities surfaceCapabilities)
    WGPU_FUNCTION_ATTRIBUTE {
    const auto output_alloc_size
        = sizeof(webgpu::Allocator) + sizeof(WGPUTextureFormat) * surfaceCapabilities.formatCount
          + sizeof(WGPUPresentMode) * surfaceCapabilities.presentModeCount
          + sizeof(WGPUCompositeAlphaMode) * surfaceCapabilities.alphaModeCount;

    auto allocator = reinterpret_cast<const webgpu::Allocator*>(surfaceCapabilities.formats) - 1;
    allocator->free({(void*)allocator, static_cast<uint32_t>(output_alloc_size)});
}
/** @} */



// MARK: Adapter Methods

void wgpuAdapterGetFeatures(WGPUAdapter            adapter,
                            WGPUSupportedFeatures* features) WGPU_FUNCTION_ATTRIBUTE {
    if (features) {
        features->featureCount = adapter->feature_count;
        features->features     = adapter->supported_features;
    }
}

WGPUStatus wgpuAdapterGetInfo(WGPUAdapter adapter, WGPUAdapterInfo* info) WGPU_FUNCTION_ATTRIBUTE {
    if (info == nullptr) { return WGPUStatus_Error; }

    *info = WGPUAdapterInfo{
        .nextInChain     = nullptr,
        .vendor          = WGPU_STRING_VIEW_INIT,
        .architecture    = WGPU_STRING_VIEW_INIT,
        .device          = WGPUStringView{.data = adapter->device_name, .length = WGPU_STRLEN},
        .description     = WGPU_STRING_VIEW_INIT,
        .backendType     = WGPUBackendType_Vulkan,
        .adapterType     = adapter->adapter_type,
        .vendorID        = adapter->vendor_id,
        .deviceID        = adapter->device_id,
        .subgroupMinSize = adapter->subgroup_min_size,
        .subgroupMaxSize = adapter->subgroup_max_size,
    };
    return WGPUStatus_Success;
}

WGPUStatus wgpuAdapterGetLimits(WGPUAdapter adapter, WGPULimits* limits) WGPU_FUNCTION_ATTRIBUTE {
    if (limits == nullptr) { return WGPUStatus_Error; }
    *limits = adapter->limits;
    return WGPUStatus_Success;
}

WGPUBool wgpuAdapterHasFeature(WGPUAdapter     adapter,
                               WGPUFeatureName feature) WGPU_FUNCTION_ATTRIBUTE {
    for (uint32_t i = 0; i < adapter->feature_count; ++i) {
        if (adapter->supported_features[i] == feature) { return true; }
    }
    return false;
}

WGPUFuture wgpuAdapterRequestDevice(WGPUAdapter                               adapter,
                                    WGPU_NULLABLE WGPUDeviceDescriptor const* descriptor,
                                    WGPURequestDeviceCallbackInfo             callbackInfo)
    WGPU_FUNCTION_ATTRIBUTE {
    return adapter->request_device(descriptor, callbackInfo);
}


void wgpuAdapterAddRef(WGPUAdapter adapter) WGPU_FUNCTION_ATTRIBUTE {
    adapter->add_ref();
}

void wgpuAdapterRelease(WGPUAdapter adapter) WGPU_FUNCTION_ATTRIBUTE {
    adapter->release();
}



/**
 * \defgroup WGPUAdapterInfoMethods WGPUAdapterInfo methods
 * \brief Functions whose first argument has type WGPUAdapterInfo.
 *
 * @{
 */
/**
 * Frees array members of WGPUAdapterInfo which were allocated by the API.
 */
void wgpuAdapterInfoFreeMembers(WGPUAdapterInfo adapterInfo) WGPU_FUNCTION_ATTRIBUTE {
    (void)adapterInfo;
}
/** @} */

// MARK: Device Methods

/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUBindGroup wgpuDeviceCreateBindGroup(WGPUDevice                     device,
                                        WGPUBindGroupDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return device->create_bind_group(descriptor);
}

/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUBindGroupLayout wgpuDeviceCreateBindGroupLayout(WGPUDevice                           device,
                                                    WGPUBindGroupLayoutDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return device->create_bind_group_layout(descriptor);
}

/**
 * TODO
 *
 * If @ref WGPUBufferDescriptor::mappedAtCreation is `true` and the mapping allocation fails,
 * returns `NULL`.
 *
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPU_NULLABLE WGPUBuffer wgpuDeviceCreateBuffer(WGPUDevice                  device,
                                                WGPUBufferDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return device->create_buffer(descriptor);
}
/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUCommandEncoder wgpuDeviceCreateCommandEncoder(
    WGPUDevice                                        device,
    WGPU_NULLABLE WGPUCommandEncoderDescriptor const* descriptor) WGPU_FUNCTION_ATTRIBUTE {
    return device->create_command_encoder(descriptor);
}
/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUComputePipeline wgpuDeviceCreateComputePipeline(WGPUDevice                           device,
                                                    WGPUComputePipelineDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return nullptr;
}

WGPUFuture wgpuDeviceCreateComputePipelineAsync(
    WGPUDevice                                 device,
    WGPUComputePipelineDescriptor const*       descriptor,
    WGPUCreateComputePipelineAsyncCallbackInfo callbackInfo) WGPU_FUNCTION_ATTRIBUTE {
    return WGPU_FUTURE_INIT;
}

/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUPipelineLayout wgpuDeviceCreatePipelineLayout(WGPUDevice                          device,
                                                  WGPUPipelineLayoutDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return device->create_pipeline_layout(descriptor);
}

/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUQuerySet wgpuDeviceCreateQuerySet(WGPUDevice device, WGPUQuerySetDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return nullptr;
}

/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPURenderBundleEncoder wgpuDeviceCreateRenderBundleEncoder(
    WGPUDevice                               device,
    WGPURenderBundleEncoderDescriptor const* descriptor) WGPU_FUNCTION_ATTRIBUTE {
    return nullptr;
}

/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPURenderPipeline wgpuDeviceCreateRenderPipeline(WGPUDevice                          device,
                                                  WGPURenderPipelineDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return device->create_render_pipeline(descriptor);
}

WGPUFuture wgpuDeviceCreateRenderPipelineAsync(
    WGPUDevice                                device,
    WGPURenderPipelineDescriptor const*       descriptor,
    WGPUCreateRenderPipelineAsyncCallbackInfo callbackInfo) WGPU_FUNCTION_ATTRIBUTE {
    return WGPU_FUTURE_INIT;
}

/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUSampler wgpuDeviceCreateSampler(WGPUDevice                                 device,
                                    WGPU_NULLABLE WGPUSamplerDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return nullptr;
}

/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUShaderModule wgpuDeviceCreateShaderModule(WGPUDevice                        device,
                                              WGPUShaderModuleDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return device->create_shader_module(descriptor);
}

/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUTexture wgpuDeviceCreateTexture(WGPUDevice device, WGPUTextureDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return nullptr;
}

void wgpuDeviceDestroy(WGPUDevice device) WGPU_FUNCTION_ATTRIBUTE {
    // Shouldn't free directly, but should somehow trigger a freeing of the device.
    // This is basically just killing any external refs, but allowing internal refs to stick
    // around. Also need to trigger destruction of any device-owned resources.
    device->destroy();
}

/**
 * @param adapterInfo
 * This parameter is @ref ReturnedWithOwnership.
 *
 * @returns
 * Indicates if there was an @ref OutStructChainError.
 */
WGPUStatus wgpuDeviceGetAdapterInfo(WGPUDevice       device,
                                    WGPUAdapterInfo* adapterInfo) WGPU_FUNCTION_ATTRIBUTE {
    return wgpuAdapterGetInfo(device->get_adapter(), adapterInfo);
}

/**
 * Get the list of @ref WGPUFeatureName values supported by the device.
 *
 * @param features
 * This parameter is @ref ReturnedWithOwnership.
 */
void wgpuDeviceGetFeatures(WGPUDevice             device,
                           WGPUSupportedFeatures* features) WGPU_FUNCTION_ATTRIBUTE {}
/**
 * @returns
 * Indicates if there was an @ref OutStructChainError.
 */
WGPUStatus wgpuDeviceGetLimits(WGPUDevice device, WGPULimits* limits) WGPU_FUNCTION_ATTRIBUTE {
    return WGPUStatus_Error;
}
/**
 * @returns
 * The @ref WGPUFuture for the device-lost event of the device.
 */
WGPUFuture wgpuDeviceGetLostFuture(WGPUDevice device) WGPU_FUNCTION_ATTRIBUTE {
    return WGPU_FUTURE_INIT;
}
/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUQueue wgpuDeviceGetQueue(WGPUDevice device) WGPU_FUNCTION_ATTRIBUTE {
    return webgpu::return_with_ownership(&device->queue);
}

WGPUBool wgpuDeviceHasFeature(WGPUDevice device, WGPUFeatureName feature) WGPU_FUNCTION_ATTRIBUTE {
    return false;
}
/**
 * Pops an error scope to the current thread's error scope stack,
 * asynchronously returning the result. See @ref ErrorScopes.
 */
WGPUFuture wgpuDevicePopErrorScope(WGPUDevice device, WGPUPopErrorScopeCallbackInfo callbackInfo)
    WGPU_FUNCTION_ATTRIBUTE {
    return device->pop_error_scope(callbackInfo);
}
/**
 * Pushes an error scope to the current thread's error scope stack.
 * See @ref ErrorScopes.
 */
void wgpuDevicePushErrorScope(WGPUDevice device, WGPUErrorFilter filter) WGPU_FUNCTION_ATTRIBUTE {
    device->push_error_scope(filter);
}

void wgpuDeviceSetLabel(WGPUDevice device, WGPUStringView label) WGPU_FUNCTION_ATTRIBUTE {
    device->label.set(device->get_allocator(), label);
}

void wgpuDeviceAddRef(WGPUDevice device) WGPU_FUNCTION_ATTRIBUTE {
    device->add_ref();
}
void wgpuDeviceRelease(WGPUDevice device) WGPU_FUNCTION_ATTRIBUTE {
    device->release();
}
/** @} */


// MARK: Bind Group Methods

/**
 * \defgroup WGPUBindGroupMethods WGPUBindGroup methods
 * \brief Functions whose first argument has type WGPUBindGroup.
 *
 * @{
 */
void wgpuBindGroupSetLabel(WGPUBindGroup bindGroup, WGPUStringView label) WGPU_FUNCTION_ATTRIBUTE {
    (void)bindGroup;
    (void)label;
}

void wgpuBindGroupAddRef(WGPUBindGroup bindGroup) WGPU_FUNCTION_ATTRIBUTE {
    (void)bindGroup;
}

void wgpuBindGroupRelease(WGPUBindGroup bindGroup) WGPU_FUNCTION_ATTRIBUTE {
    (void)bindGroup;
}
/** @} */



/**
 * \defgroup WGPUBindGroupLayoutMethods WGPUBindGroupLayout methods
 * \brief Functions whose first argument has type WGPUBindGroupLayout.
 *
 * @{
 */
void wgpuBindGroupLayoutSetLabel(WGPUBindGroupLayout bindGroupLayout,
                                 WGPUStringView      label) WGPU_FUNCTION_ATTRIBUTE {
    bindGroupLayout->set_label(label);
}
void wgpuBindGroupLayoutAddRef(WGPUBindGroupLayout bindGroupLayout) WGPU_FUNCTION_ATTRIBUTE {
    bindGroupLayout->add_ref();
}
void wgpuBindGroupLayoutRelease(WGPUBindGroupLayout bindGroupLayout) WGPU_FUNCTION_ATTRIBUTE {
    if (bindGroupLayout->release()) { bindGroupLayout->device->free(bindGroupLayout); }
}
/** @} */



/**
 * \defgroup WGPUBufferMethods WGPUBuffer methods
 * \brief Functions whose first argument has type WGPUBuffer.
 *
 * @{
 */
void wgpuBufferDestroy(WGPUBuffer buffer) WGPU_FUNCTION_ATTRIBUTE {
    buffer->destroy();
}
/**
 * Returns a const pointer to beginning of the mapped range.
 * It must not be written; writing to this range causes undefined behavior.
 * See @ref MappedRangeBehavior for error conditions and guarantees.
 * This function is safe to call inside spontaneous callbacks (see @ref CallbackReentrancy).
 *
 * In Wasm, if `memcpy`ing from this range, prefer using @ref wgpuBufferReadMappedRange
 * instead for better performance.
 *
 * @param offset
 * Byte offset relative to the beginning of the buffer.
 *
 * @param size
 * Byte size of the range to get.
 * If this is @ref WGPU_WHOLE_MAP_SIZE, it defaults to `buffer.size - offset`.
 * The returned pointer is valid for exactly this many bytes.
 */
void const* wgpuBufferGetConstMappedRange(WGPUBuffer buffer,
                                          size_t     offset,
                                          size_t     size) WGPU_FUNCTION_ATTRIBUTE {
    return buffer->get_mapped_range(offset, size);
}

WGPUBufferMapState wgpuBufferGetMapState(WGPUBuffer buffer) WGPU_FUNCTION_ATTRIBUTE {
    return buffer->map_state;
}
/**
 * Returns a mutable pointer to beginning of the mapped range.
 * See @ref MappedRangeBehavior for error conditions and guarantees.
 * This function is safe to call inside spontaneous callbacks (see @ref CallbackReentrancy).
 *
 * In Wasm, if `memcpy`ing into this range, prefer using @ref wgpuBufferWriteMappedRange
 * instead for better performance.
 *
 * @param offset
 * Byte offset relative to the beginning of the buffer.
 *
 * @param size
 * Byte size of the range to get.
 * If this is @ref WGPU_WHOLE_MAP_SIZE, it defaults to `buffer.size - offset`.
 * The returned pointer is valid for exactly this many bytes.
 */
void* wgpuBufferGetMappedRange(WGPUBuffer buffer,
                               size_t     offset,
                               size_t     size) WGPU_FUNCTION_ATTRIBUTE {
    // https://webgpu-native.github.io/webgpu-headers/BufferMapping.html
    if ((buffer->mapping.map_mode & WGPUMapMode_Write) == 0) {
        // TODO: Logging
        return nullptr;
    }
    return buffer->get_mapped_range(offset, size);
}

uint64_t wgpuBufferGetSize(WGPUBuffer buffer) WGPU_FUNCTION_ATTRIBUTE {
    return buffer->size;
}

WGPUBufferUsage wgpuBufferGetUsage(WGPUBuffer buffer) WGPU_FUNCTION_ATTRIBUTE {
    return buffer->usage;
}

/**
 * @param offset
 * Byte offset relative to beginning of the buffer.
 *
 * @param size
 * Byte size of the region to map.
 * If this is @ref WGPU_WHOLE_MAP_SIZE, it defaults to `buffer.size - offset`.
 */
WGPUFuture wgpuBufferMapAsync(WGPUBuffer                buffer,
                              WGPUMapMode               mode,
                              size_t                    offset,
                              size_t                    size,
                              WGPUBufferMapCallbackInfo callbackInfo) WGPU_FUNCTION_ATTRIBUTE {
    return buffer->map_async(mode, offset, size, callbackInfo);
}
/**
 * Copies a range of data from the buffer mapping into the provided destination pointer.
 * See @ref MappedRangeBehavior for error conditions and guarantees.
 * This function is safe to call inside spontaneous callbacks (see @ref CallbackReentrancy).
 *
 * In Wasm, this is more efficient than copying from a mapped range into a `malloc`'d range.
 *
 * @param offset
 * Byte offset relative to the beginning of the buffer.
 *
 * @param data
 * Destination, to read buffer data into.
 *
 * @param size
 * Number of bytes of data to read from the buffer.
 * (Note @ref WGPU_WHOLE_MAP_SIZE is *not* accepted here.)
 *
 * @returns
 * @ref WGPUStatus_Error if the copy did not occur.
 */
WGPUStatus wgpuBufferReadMappedRange(WGPUBuffer buffer, size_t offset, void* data, size_t size)
    WGPU_FUNCTION_ATTRIBUTE {
    // See https://webgpu-native.github.io/webgpu-headers/BufferMapping.html for behavior
    void* src = buffer->get_mapped_range(offset, size);
    if (src == nullptr) { return WGPUStatus_Error; }
    memcpy(data, src, size);
    return WGPUStatus_Success;
}

void wgpuBufferSetLabel(WGPUBuffer buffer, WGPUStringView label) WGPU_FUNCTION_ATTRIBUTE {
    buffer->set_label(label);
}

void wgpuBufferUnmap(WGPUBuffer buffer) WGPU_FUNCTION_ATTRIBUTE {
    buffer->unmap();
}

/**
 * Copies a range of data from the provided source pointer into the buffer mapping.
 * See @ref MappedRangeBehavior for error conditions and guarantees.
 * This function is safe to call inside spontaneous callbacks (see @ref CallbackReentrancy).
 *
 * In Wasm, this is more efficient than copying from a `malloc`'d range into a mapped range.
 *
 * @param offset
 * Byte offset relative to the beginning of the buffer.
 *
 * @param data
 * Source, to write buffer data from.
 *
 * @param size
 * Number of bytes of data to write to the buffer.
 * (Note @ref WGPU_WHOLE_MAP_SIZE is *not* accepted here.)
 *
 * @returns
 * @ref WGPUStatus_Error if the copy did not occur.
 */


WGPUStatus wgpuBufferWriteMappedRange(WGPUBuffer  buffer,
                                      size_t      offset,
                                      void const* data,
                                      size_t      size) WGPU_FUNCTION_ATTRIBUTE {
    // See https://webgpu-native.github.io/webgpu-headers/BufferMapping.html for behavior
    if ((buffer->mapping.map_mode & WGPUMapMode_Write) == 0) {
        // TODO: Logging here
        return WGPUStatus_Error;
    }
    void* dst = buffer->get_mapped_range(offset, size);
    if (dst == nullptr) { return WGPUStatus_Error; }
    memcpy(dst, data, size);
    return WGPUStatus_Success;
}
void wgpuBufferAddRef(WGPUBuffer buffer) WGPU_FUNCTION_ATTRIBUTE {
    buffer->add_ref();
}
void wgpuBufferRelease(WGPUBuffer buffer) WGPU_FUNCTION_ATTRIBUTE {
    webgpu::release(buffer);
}
/** @} */



/**
 * \defgroup WGPUCommandBufferMethods WGPUCommandBuffer methods
 * \brief Functions whose first argument has type WGPUCommandBuffer.
 *
 * @{
 */
void wgpuCommandBufferSetLabel(WGPUCommandBuffer commandBuffer,
                               WGPUStringView    label) WGPU_FUNCTION_ATTRIBUTE {
    commandBuffer->set_label(label);
}

void wgpuCommandBufferAddRef(WGPUCommandBuffer commandBuffer) WGPU_FUNCTION_ATTRIBUTE {
    commandBuffer->add_ref();
}

void wgpuCommandBufferRelease(WGPUCommandBuffer commandBuffer) WGPU_FUNCTION_ATTRIBUTE {
    webgpu::release(commandBuffer);
}
/** @} */



/**
 * \defgroup WGPUCommandEncoderMethods WGPUCommandEncoder methods
 * \brief Functions whose first argument has type WGPUCommandEncoder.
 *
 * @{
 */
/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUComputePassEncoder wgpuCommandEncoderBeginComputePass(
    WGPUCommandEncoder                             commandEncoder,
    WGPU_NULLABLE WGPUComputePassDescriptor const* descriptor) WGPU_FUNCTION_ATTRIBUTE {
    return nullptr;
}
/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPURenderPassEncoder wgpuCommandEncoderBeginRenderPass(WGPUCommandEncoder commandEncoder,
                                                        WGPURenderPassDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return commandEncoder->begin_render_pass(descriptor);
}

void wgpuCommandEncoderClearBuffer(WGPUCommandEncoder commandEncoder,
                                   WGPUBuffer         buffer,
                                   uint64_t           offset,
                                   uint64_t           size) WGPU_FUNCTION_ATTRIBUTE {
    buffer->add_ref_internal();
    commandEncoder->commands_mixin.add(webgpu::Command{
        .clear_buffer = {
            .buffer = buffer,
            .offset = offset,
            .size = size,
        },
        .type = webgpu::CommandType::ClearBuffer,
    });
}

void wgpuCommandEncoderCopyBufferToBuffer(WGPUCommandEncoder commandEncoder,
                                          WGPUBuffer         source,
                                          uint64_t           sourceOffset,
                                          WGPUBuffer         destination,
                                          uint64_t           destinationOffset,
                                          uint64_t           size) WGPU_FUNCTION_ATTRIBUTE {
    source->add_ref_internal();
    destination->add_ref_internal();
    commandEncoder->commands_mixin.add(webgpu::Command{
        .copy_buffer_to_buffer = {.src        = source,
                                  .dst        = destination,
                                  .src_offset = sourceOffset,
                                  .dst_offset = destinationOffset,
                                  .size       = size,},
        .type                  = webgpu::CommandType::CopyBufferToBuffer,
    });
}
void wgpuCommandEncoderCopyBufferToTexture(WGPUCommandEncoder              commandEncoder,
                                           WGPUTexelCopyBufferInfo const*  source,
                                           WGPUTexelCopyTextureInfo const* destination,
                                           WGPUExtent3D const* copySize) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuCommandEncoderCopyTextureToBuffer(WGPUCommandEncoder              commandEncoder,
                                           WGPUTexelCopyTextureInfo const* source,
                                           WGPUTexelCopyBufferInfo const*  destination,
                                           WGPUExtent3D const* copySize) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuCommandEncoderCopyTextureToTexture(WGPUCommandEncoder              commandEncoder,
                                            WGPUTexelCopyTextureInfo const* source,
                                            WGPUTexelCopyTextureInfo const* destination,
                                            WGPUExtent3D const* copySize) WGPU_FUNCTION_ATTRIBUTE {}
/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUCommandBuffer wgpuCommandEncoderFinish(
    WGPUCommandEncoder                               commandEncoder,
    WGPU_NULLABLE WGPUCommandBufferDescriptor const* descriptor) WGPU_FUNCTION_ATTRIBUTE {
    return commandEncoder->finish(descriptor);
}
void wgpuCommandEncoderInsertDebugMarker(WGPUCommandEncoder commandEncoder,
                                         WGPUStringView     markerLabel) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuCommandEncoderPopDebugGroup(WGPUCommandEncoder commandEncoder) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuCommandEncoderPushDebugGroup(WGPUCommandEncoder commandEncoder,
                                      WGPUStringView     groupLabel) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuCommandEncoderResolveQuerySet(WGPUCommandEncoder commandEncoder,
                                       WGPUQuerySet       querySet,
                                       uint32_t           firstQuery,
                                       uint32_t           queryCount,
                                       WGPUBuffer         destination,
                                       uint64_t destinationOffset) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuCommandEncoderSetLabel(WGPUCommandEncoder commandEncoder,
                                WGPUStringView     label) WGPU_FUNCTION_ATTRIBUTE {
    commandEncoder->set_label(label);
}
void wgpuCommandEncoderWriteTimestamp(WGPUCommandEncoder commandEncoder,
                                      WGPUQuerySet       querySet,
                                      uint32_t           queryIndex) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuCommandEncoderAddRef(WGPUCommandEncoder commandEncoder) WGPU_FUNCTION_ATTRIBUTE {
    commandEncoder->add_ref();
}
void wgpuCommandEncoderRelease(WGPUCommandEncoder commandEncoder) WGPU_FUNCTION_ATTRIBUTE {
    if (commandEncoder->release()) { commandEncoder->device->free(commandEncoder); }
}
/** @} */



/**
 * \defgroup WGPUComputePassEncoderMethods WGPUComputePassEncoder methods
 * \brief Functions whose first argument has type WGPUComputePassEncoder.
 *
 * @{
 */
void wgpuComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder computePassEncoder,
                                              uint32_t               workgroupCountX,
                                              uint32_t               workgroupCountY,
                                              uint32_t workgroupCountZ) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePassEncoderDispatchWorkgroupsIndirect(WGPUComputePassEncoder computePassEncoder,
                                                      WGPUBuffer             indirectBuffer,
                                                      uint64_t               indirectOffset)
    WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePassEncoderEnd(WGPUComputePassEncoder computePassEncoder) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePassEncoderInsertDebugMarker(WGPUComputePassEncoder computePassEncoder,
                                             WGPUStringView markerLabel) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePassEncoderPopDebugGroup(WGPUComputePassEncoder computePassEncoder)
    WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePassEncoderPushDebugGroup(WGPUComputePassEncoder computePassEncoder,
                                          WGPUStringView groupLabel) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder      computePassEncoder,
                                        uint32_t                    groupIndex,
                                        WGPU_NULLABLE WGPUBindGroup group,
                                        size_t                      dynamicOffsetCount,
                                        uint32_t const* dynamicOffsets) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePassEncoderSetLabel(WGPUComputePassEncoder computePassEncoder,
                                    WGPUStringView         label) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePassEncoderSetPipeline(WGPUComputePassEncoder computePassEncoder,
                                       WGPUComputePipeline    pipeline) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePassEncoderAddRef(WGPUComputePassEncoder computePassEncoder)
    WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePassEncoderRelease(WGPUComputePassEncoder computePassEncoder)
    WGPU_FUNCTION_ATTRIBUTE {}
/** @} */



/**
 * \defgroup WGPUComputePipelineMethods WGPUComputePipeline methods
 * \brief Functions whose first argument has type WGPUComputePipeline.
 *
 * @{
 */
/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUBindGroupLayout wgpuComputePipelineGetBindGroupLayout(WGPUComputePipeline computePipeline,
                                                          uint32_t            groupIndex)
    WGPU_FUNCTION_ATTRIBUTE {
    return nullptr;
}
void wgpuComputePipelineSetLabel(WGPUComputePipeline computePipeline,
                                 WGPUStringView      label) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePipelineAddRef(WGPUComputePipeline computePipeline) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuComputePipelineRelease(WGPUComputePipeline computePipeline) WGPU_FUNCTION_ATTRIBUTE {}
/** @} */



/**
 * \defgroup WGPUPipelineLayoutMethods WGPUPipelineLayout methods
 * \brief Functions whose first argument has type WGPUPipelineLayout.
 *
 * @{
 */
void wgpuPipelineLayoutSetLabel(WGPUPipelineLayout pipelineLayout,
                                WGPUStringView     label) WGPU_FUNCTION_ATTRIBUTE {
    pipelineLayout->set_label(label);
}
void wgpuPipelineLayoutAddRef(WGPUPipelineLayout pipelineLayout) WGPU_FUNCTION_ATTRIBUTE {
    pipelineLayout->add_ref();
}
void wgpuPipelineLayoutRelease(WGPUPipelineLayout pipelineLayout) WGPU_FUNCTION_ATTRIBUTE {
    webgpu::release(pipelineLayout);
}
/** @} */



/**
 * \defgroup WGPUQuerySetMethods WGPUQuerySet methods
 * \brief Functions whose first argument has type WGPUQuerySet.
 *
 * @{
 */
void     wgpuQuerySetDestroy(WGPUQuerySet querySet) WGPU_FUNCTION_ATTRIBUTE {}
uint32_t wgpuQuerySetGetCount(WGPUQuerySet querySet) WGPU_FUNCTION_ATTRIBUTE {
    return 0;
}
WGPUQueryType wgpuQuerySetGetType(WGPUQuerySet querySet) WGPU_FUNCTION_ATTRIBUTE {
    return WGPUQueryType_Force32;
}
void wgpuQuerySetSetLabel(WGPUQuerySet querySet, WGPUStringView label) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuQuerySetAddRef(WGPUQuerySet querySet) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuQuerySetRelease(WGPUQuerySet querySet) WGPU_FUNCTION_ATTRIBUTE {}
/** @} */


// MARK: Queue methods

/**
 * \defgroup WGPUQueueMethods WGPUQueue methods
 * \brief Functions whose first argument has type WGPUQueue.
 *
 * @{
 */
WGPUFuture wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo callbackInfo)
    WGPU_FUNCTION_ATTRIBUTE {
    return queue->on_submitted_work_done(callbackInfo);
}
void wgpuQueueSetLabel(WGPUQueue queue, WGPUStringView label) WGPU_FUNCTION_ATTRIBUTE {
    queue->set_label(label);
}

void wgpuQueueSubmit(WGPUQueue                queue,
                     size_t                   command_count,
                     WGPUCommandBuffer const* commands) WGPU_FUNCTION_ATTRIBUTE {
    queue->submit(command_count, commands);
}
/**
 * Produces a @ref DeviceError both content-timeline (`size` alignment) and device-timeline
 * errors defined by the WebGPU specification.
 */
void wgpuQueueWriteBuffer(WGPUQueue   queue,
                          WGPUBuffer  buffer,
                          uint64_t    bufferOffset,
                          void const* data,
                          size_t      size) WGPU_FUNCTION_ATTRIBUTE {
    queue->write_buffer(buffer, bufferOffset, data, size);
}
void wgpuQueueWriteTexture(WGPUQueue                        queue,
                           WGPUTexelCopyTextureInfo const*  destination,
                           void const*                      data,
                           size_t                           dataSize,
                           WGPUTexelCopyBufferLayout const* dataLayout,
                           WGPUExtent3D const*              writeSize) WGPU_FUNCTION_ATTRIBUTE {
    queue->write_texture(destination, data, dataSize, dataLayout, writeSize);
}
void wgpuQueueAddRef(WGPUQueue queue) WGPU_FUNCTION_ATTRIBUTE {
    queue->add_ref();
}
void wgpuQueueRelease(WGPUQueue queue) WGPU_FUNCTION_ATTRIBUTE {
    webgpu::release(queue);
}
/** @} */



/**
 * \defgroup WGPURenderBundleMethods WGPURenderBundle methods
 * \brief Functions whose first argument has type WGPURenderBundle.
 *
 * @{
 */
void wgpuRenderBundleSetLabel(WGPURenderBundle renderBundle,
                              WGPUStringView   label) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleAddRef(WGPURenderBundle renderBundle) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleRelease(WGPURenderBundle renderBundle) WGPU_FUNCTION_ATTRIBUTE {}
/** @} */



/**
 * \defgroup WGPURenderBundleEncoderMethods WGPURenderBundleEncoder methods
 * \brief Functions whose first argument has type WGPURenderBundleEncoder.
 *
 * @{
 */
void wgpuRenderBundleEncoderDraw(WGPURenderBundleEncoder renderBundleEncoder,
                                 uint32_t                vertexCount,
                                 uint32_t                instanceCount,
                                 uint32_t                firstVertex,
                                 uint32_t                firstInstance) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderDrawIndexed(WGPURenderBundleEncoder renderBundleEncoder,
                                        uint32_t                indexCount,
                                        uint32_t                instanceCount,
                                        uint32_t                firstIndex,
                                        int32_t                 baseVertex,
                                        uint32_t firstInstance) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderDrawIndexedIndirect(WGPURenderBundleEncoder renderBundleEncoder,
                                                WGPUBuffer              indirectBuffer,
                                                uint64_t indirectOffset) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderDrawIndirect(WGPURenderBundleEncoder renderBundleEncoder,
                                         WGPUBuffer              indirectBuffer,
                                         uint64_t indirectOffset) WGPU_FUNCTION_ATTRIBUTE {}
/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPURenderBundle wgpuRenderBundleEncoderFinish(
    WGPURenderBundleEncoder                         renderBundleEncoder,
    WGPU_NULLABLE WGPURenderBundleDescriptor const* descriptor) WGPU_FUNCTION_ATTRIBUTE {
    return nullptr;
}
void wgpuRenderBundleEncoderInsertDebugMarker(WGPURenderBundleEncoder renderBundleEncoder,
                                              WGPUStringView markerLabel) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderPopDebugGroup(WGPURenderBundleEncoder renderBundleEncoder)
    WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderPushDebugGroup(WGPURenderBundleEncoder renderBundleEncoder,
                                           WGPUStringView groupLabel) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderSetBindGroup(WGPURenderBundleEncoder     renderBundleEncoder,
                                         uint32_t                    groupIndex,
                                         WGPU_NULLABLE WGPUBindGroup group,
                                         size_t                      dynamicOffsetCount,
                                         uint32_t const* dynamicOffsets) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderSetIndexBuffer(WGPURenderBundleEncoder renderBundleEncoder,
                                           WGPUBuffer              buffer,
                                           WGPUIndexFormat         format,
                                           uint64_t                offset,
                                           uint64_t                size) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderSetLabel(WGPURenderBundleEncoder renderBundleEncoder,
                                     WGPUStringView          label) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderSetPipeline(WGPURenderBundleEncoder renderBundleEncoder,
                                        WGPURenderPipeline      pipeline) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderSetVertexBuffer(WGPURenderBundleEncoder  renderBundleEncoder,
                                            uint32_t                 slot,
                                            WGPU_NULLABLE WGPUBuffer buffer,
                                            uint64_t                 offset,
                                            uint64_t size) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderAddRef(WGPURenderBundleEncoder renderBundleEncoder)
    WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderBundleEncoderRelease(WGPURenderBundleEncoder renderBundleEncoder)
    WGPU_FUNCTION_ATTRIBUTE {}
/** @} */



/**
 * \defgroup WGPURenderPassEncoderMethods WGPURenderPassEncoder methods
 * \brief Functions whose first argument has type WGPURenderPassEncoder.
 *
 * @{
 */
void wgpuRenderPassEncoderBeginOcclusionQuery(WGPURenderPassEncoder renderPassEncoder,
                                              uint32_t queryIndex) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderPassEncoderDraw(WGPURenderPassEncoder renderPassEncoder,
                               uint32_t              vertexCount,
                               uint32_t              instanceCount,
                               uint32_t              firstVertex,
                               uint32_t              firstInstance) WGPU_FUNCTION_ATTRIBUTE {
    if (!webgpu::validate_draw(renderPassEncoder,
                               vertexCount,
                               instanceCount,
                               firstVertex,
                               firstInstance)) {
        return;
    }

    auto commands = renderPassEncoder->commands_mixin;
    commands->add(webgpu::Command{
        .draw{
            .vertex_count   = vertexCount,
            .instance_count = instanceCount,
            .first_vertex   = firstVertex,
            .first_instance = firstInstance,
        },
        .type = webgpu::CommandType::Draw,
    });
}
void wgpuRenderPassEncoderDrawIndexed(WGPURenderPassEncoder renderPassEncoder,
                                      uint32_t              indexCount,
                                      uint32_t              instanceCount,
                                      uint32_t              firstIndex,
                                      int32_t               baseVertex,
                                      uint32_t              firstInstance) WGPU_FUNCTION_ATTRIBUTE {
    auto commands = renderPassEncoder->commands_mixin;
    commands->add(webgpu::Command{
        .draw_indexed{
            .index_count    = indexCount,
            .instance_count = instanceCount,
            .first_index    = firstIndex,
            .base_vertex    = static_cast<uint32_t>(baseVertex),
            .first_instance = firstInstance,
        },
        .type = webgpu::CommandType::DrawIndexed,
    });
}

void wgpuRenderPassEncoderDrawIndexedIndirect(WGPURenderPassEncoder renderPassEncoder,
                                              WGPUBuffer            indirectBuffer,
                                              uint64_t indirectOffset) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderPassEncoderDrawIndirect(WGPURenderPassEncoder renderPassEncoder,
                                       WGPUBuffer            indirectBuffer,
                                       uint64_t indirectOffset) WGPU_FUNCTION_ATTRIBUTE {}

void wgpuRenderPassEncoderEnd(WGPURenderPassEncoder renderPassEncoder) WGPU_FUNCTION_ATTRIBUTE {
    if (!webgpu::validate(renderPassEncoder)) { return; }

    auto commands = renderPassEncoder->commands_mixin;
    commands->add(webgpu::Command{
        .end_render_pass{},
        .type = webgpu::CommandType::EndRenderPass,
    });
    commands->state = webgpu::CommandEncodingState::Open;
}

void wgpuRenderPassEncoderEndOcclusionQuery(WGPURenderPassEncoder renderPassEncoder)
    WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderPassEncoderExecuteBundles(WGPURenderPassEncoder   renderPassEncoder,
                                         size_t                  bundleCount,
                                         WGPURenderBundle const* bundles) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderPassEncoderInsertDebugMarker(WGPURenderPassEncoder renderPassEncoder,
                                            WGPUStringView markerLabel) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderPassEncoderPopDebugGroup(WGPURenderPassEncoder renderPassEncoder)
    WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderPassEncoderPushDebugGroup(WGPURenderPassEncoder renderPassEncoder,
                                         WGPUStringView groupLabel) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderPassEncoderSetBindGroup(WGPURenderPassEncoder       renderPassEncoder,
                                       uint32_t                    groupIndex,
                                       WGPU_NULLABLE WGPUBindGroup group,
                                       size_t                      dynamicOffsetCount,
                                       uint32_t const* dynamicOffsets) WGPU_FUNCTION_ATTRIBUTE {}
/**
 * @param color
 * The RGBA blend constant. Represents an `f32` color using @ref DoubleAsSupertype.
 */
void wgpuRenderPassEncoderSetBlendConstant(WGPURenderPassEncoder renderPassEncoder,
                                           WGPUColor const*      color) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderPassEncoderSetIndexBuffer(WGPURenderPassEncoder renderPassEncoder,
                                         WGPUBuffer            buffer,
                                         WGPUIndexFormat       format,
                                         uint64_t              offset,
                                         uint64_t              size) WGPU_FUNCTION_ATTRIBUTE {
    auto commands = renderPassEncoder->commands_mixin;
    if (buffer) {
        renderPassEncoder->render_commands.usage_scope.add(buffer,
                                                           webgpu::ResourceUsage::kUsageInput,
                                                           WGPUShaderStage_Vertex);
        buffer->add_ref_internal();
    }
    commands->add(webgpu::Command {
        .set_index_buffer = {
            .buffer = buffer,
            .format = format,
            .offset = offset,
            .size   = size,
        },
        .type = webgpu::CommandType::SetIndexBuffer,
    });
}
void wgpuRenderPassEncoderSetLabel(WGPURenderPassEncoder renderPassEncoder,
                                   WGPUStringView        label) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderPassEncoderSetPipeline(WGPURenderPassEncoder renderPassEncoder,
                                      WGPURenderPipeline    pipeline) WGPU_FUNCTION_ATTRIBUTE {
    if (!webgpu::validate(renderPassEncoder, pipeline)) { return; }

    auto commands = renderPassEncoder->commands_mixin;
    pipeline->add_ref_internal();
    commands->add(webgpu::Command{
        .set_render_pipeline{
            .pipeline = pipeline,
        },
        .type = webgpu::CommandType::SetRenderPipeline,
    });
}
void wgpuRenderPassEncoderSetScissorRect(WGPURenderPassEncoder renderPassEncoder,
                                         uint32_t              x,
                                         uint32_t              y,
                                         uint32_t              width,
                                         uint32_t              height) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderPassEncoderSetStencilReference(WGPURenderPassEncoder renderPassEncoder,
                                              uint32_t reference) WGPU_FUNCTION_ATTRIBUTE {}
void wgpuRenderPassEncoderSetVertexBuffer(WGPURenderPassEncoder    renderPassEncoder,
                                          uint32_t                 slot,
                                          WGPU_NULLABLE WGPUBuffer buffer,
                                          uint64_t                 offset,
                                          uint64_t                 size) WGPU_FUNCTION_ATTRIBUTE {
    auto commands = renderPassEncoder->commands_mixin;

    if (buffer) {
        renderPassEncoder->render_commands.usage_scope.add(buffer,
                                                           webgpu::ResourceUsage::kUsageInput,
                                                           WGPUShaderStage_Vertex);
        buffer->add_ref_internal();
    }
    commands->add(webgpu::Command{
        .set_vertex_buffer = {
            .slot = slot,
            .buffer = buffer,
            .offset = offset,
            .size = size,
        },
        .type = webgpu::CommandType::SetVertexBuffer,
    });
}
/**
 * TODO
 *
 * If any argument is non-finite, produces a @ref NonFiniteFloatValueError.
 */
void wgpuRenderPassEncoderSetViewport(WGPURenderPassEncoder renderPassEncoder,
                                      float                 x,
                                      float                 y,
                                      float                 width,
                                      float                 height,
                                      float                 min_depth,
                                      float                 max_depth) WGPU_FUNCTION_ATTRIBUTE {
    // TODO: Viewport validation
    // if (!webgpu::validate(renderPassEncoder, pipeline)) { return; }

    auto commands = renderPassEncoder->commands_mixin;
    commands->add(webgpu::Command{
        .set_viewport{
            .x         = x,
            .y         = y,
            .width     = width,
            .height    = height,
            .min_depth = min_depth,
            .max_depth = max_depth,
        },
        .type = webgpu::CommandType::SetViewport,
    });
}
void wgpuRenderPassEncoderAddRef(WGPURenderPassEncoder renderPassEncoder) WGPU_FUNCTION_ATTRIBUTE {
    renderPassEncoder->add_ref();
}
void wgpuRenderPassEncoderRelease(WGPURenderPassEncoder renderPassEncoder) WGPU_FUNCTION_ATTRIBUTE {
    webgpu::release(renderPassEncoder);
}
/** @} */

/**
 * \defgroup WGPURenderPipelineMethods WGPURenderPipeline methods
 * \brief Functions whose first argument has type WGPURenderPipeline.
 *
 * @{
 */

/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUBindGroupLayout wgpuRenderPipelineGetBindGroupLayout(WGPURenderPipeline renderPipeline,
                                                         uint32_t           groupIndex)
    WGPU_FUNCTION_ATTRIBUTE {
    return nullptr;
}
void wgpuRenderPipelineSetLabel(WGPURenderPipeline renderPipeline,
                                WGPUStringView     label) WGPU_FUNCTION_ATTRIBUTE {
    renderPipeline->set_label(label);
}

void wgpuRenderPipelineAddRef(WGPURenderPipeline renderPipeline) WGPU_FUNCTION_ATTRIBUTE {
    renderPipeline->add_ref();
}

void wgpuRenderPipelineRelease(WGPURenderPipeline renderPipeline) WGPU_FUNCTION_ATTRIBUTE {
    webgpu::release(renderPipeline);
}

/** @} */



/**
 * \defgroup WGPUSamplerMethods WGPUSampler methods
 * \brief Functions whose first argument has type WGPUSampler.
 *
 * @{
 */
void wgpuSamplerSetLabel(WGPUSampler sampler, WGPUStringView label) WGPU_FUNCTION_ATTRIBUTE {}

void wgpuSamplerAddRef(WGPUSampler sampler) WGPU_FUNCTION_ATTRIBUTE {}

void wgpuSamplerRelease(WGPUSampler sampler) WGPU_FUNCTION_ATTRIBUTE {}
/** @} */



/**
 * \defgroup WGPUShaderModuleMethods WGPUShaderModule methods
 * \brief Functions whose first argument has type WGPUShaderModule.
 *
 * @{
 */

WGPUFuture wgpuShaderModuleGetCompilationInfo(WGPUShaderModule                shaderModule,
                                              WGPUCompilationInfoCallbackInfo callbackInfo)
    WGPU_FUNCTION_ATTRIBUTE {
    // We don't really have compilation for shader modules since we're taking just SPIRV.
    webgpu::CallbackData* cb;
    WGPUFuture            result = shaderModule->device->get_instance()->create_future(&cb);

    *cb = webgpu::CallbackData{
        .callback  = (WGPUProc)callbackInfo.callback,
        .mode      = callbackInfo.mode,
        .userdata1 = callbackInfo.userdata1,
        .userdata2 = callbackInfo.userdata2,
        .message   = WGPU_STRING_VIEW_INIT,
        .type      = webgpu::CallbackType::CompilationInfo,
        .compilation_info
        = {.status = WGPUCompilationInfoRequestStatus_Success,
           .info   = {.nextInChain = nullptr, .messageCount = 0, .messages = nullptr}},
    };
    shaderModule->device->get_instance()->set_future_ready(result);
    return result;
}

void wgpuShaderModuleSetLabel(WGPUShaderModule shaderModule,
                              WGPUStringView   label) WGPU_FUNCTION_ATTRIBUTE {
    shaderModule->set_label(label);
}

void wgpuShaderModuleAddRef(WGPUShaderModule shaderModule) WGPU_FUNCTION_ATTRIBUTE {
    shaderModule->add_ref();
}

void wgpuShaderModuleRelease(WGPUShaderModule shaderModule) WGPU_FUNCTION_ATTRIBUTE {
    if (shaderModule->release()) { shaderModule->device->free(shaderModule); }
}
/** @} */



/**
 * \defgroup WGPUSupportedFeaturesMethods WGPUSupportedFeatures methods
 * \brief Functions whose first argument has type WGPUSupportedFeatures.
 *
 * @{
 */
/**
 * Frees array members of WGPUSupportedFeatures which were allocated by the API.
 */
void wgpuSupportedFeaturesFreeMembers(WGPUSupportedFeatures supportedFeatures)
    WGPU_FUNCTION_ATTRIBUTE {}
/** @} */



/**
 * \defgroup WGPUSupportedWGSLLanguageFeaturesMethods WGPUSupportedWGSLLanguageFeatures methods
 * \brief Functions whose first argument has type WGPUSupportedWGSLLanguageFeatures.
 *
 * @{
 */
/**
 * Frees array members of WGPUSupportedWGSLLanguageFeatures which were allocated by the API.
 */
void wgpuSupportedWGSLLanguageFeaturesFreeMembers(
    WGPUSupportedWGSLLanguageFeatures supportedWGSLLanguageFeatures) WGPU_FUNCTION_ATTRIBUTE {}
/** @} */



/**
 * \defgroup WGPUTextureMethods WGPUTexture methods
 * \brief Functions whose first argument has type WGPUTexture.
 *
 * @{
 */

/**
 * @returns
 * This value is @ref ReturnedWithOwnership.
 */
WGPUTextureView wgpuTextureCreateView(WGPUTexture                                    texture,
                                      WGPU_NULLABLE WGPUTextureViewDescriptor const* descriptor)
    WGPU_FUNCTION_ATTRIBUTE {
    return texture->create_view(descriptor);
}

void wgpuTextureDestroy(WGPUTexture texture) WGPU_FUNCTION_ATTRIBUTE {}

uint32_t wgpuTextureGetDepthOrArrayLayers(WGPUTexture texture) WGPU_FUNCTION_ATTRIBUTE {
    return texture->depth_or_array_layers;
}
WGPUTextureDimension wgpuTextureGetDimension(WGPUTexture texture) WGPU_FUNCTION_ATTRIBUTE {
    return texture->dimension;
}
WGPUTextureFormat wgpuTextureGetFormat(WGPUTexture texture) WGPU_FUNCTION_ATTRIBUTE {
    return texture->format;
}
uint32_t wgpuTextureGetHeight(WGPUTexture texture) WGPU_FUNCTION_ATTRIBUTE {
    return texture->height;
}
uint32_t wgpuTextureGetMipLevelCount(WGPUTexture texture) WGPU_FUNCTION_ATTRIBUTE {
    return texture->mip_level_count;
}
uint32_t wgpuTextureGetSampleCount(WGPUTexture texture) WGPU_FUNCTION_ATTRIBUTE {
    return texture->sample_count;
}
WGPUTextureUsage wgpuTextureGetUsage(WGPUTexture texture) WGPU_FUNCTION_ATTRIBUTE {
    return texture->usages;
}
uint32_t wgpuTextureGetWidth(WGPUTexture texture) WGPU_FUNCTION_ATTRIBUTE {
    return texture->width;
}

void wgpuTextureSetLabel(WGPUTexture texture, WGPUStringView label) WGPU_FUNCTION_ATTRIBUTE {
    texture->set_label(label);
}
void wgpuTextureAddRef(WGPUTexture texture) WGPU_FUNCTION_ATTRIBUTE {
    texture->add_ref();
}
void wgpuTextureRelease(WGPUTexture texture) WGPU_FUNCTION_ATTRIBUTE {
    webgpu::release(texture);
}
/** @} */



/**
 * \defgroup WGPUTextureViewMethods WGPUTextureView methods
 * \brief Functions whose first argument has type WGPUTextureView.
 *
 * @{
 */
void wgpuTextureViewSetLabel(WGPUTextureView textureView,
                             WGPUStringView  label) WGPU_FUNCTION_ATTRIBUTE {
    textureView->set_label(label);
}

void wgpuTextureViewAddRef(WGPUTextureView textureView) WGPU_FUNCTION_ATTRIBUTE {
    textureView->add_ref();
}

void wgpuTextureViewRelease(WGPUTextureView textureView) WGPU_FUNCTION_ATTRIBUTE {
    webgpu::release(textureView);
}

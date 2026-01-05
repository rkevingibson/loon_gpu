#pragma once

#include "volk.h"
#include "webgpu/webgpu.h"

/***
 * Argument Validation Functions for WebGPU arguments
 * These are all moved to separate functions to keep the main code path clean, while also enabling
 * the possibility of globally disabling validation if needed for performance. Most validation
 * functions take the device pointer, to report device errors on failure. All functions return true
 * iff the arguments are valid.
 */
namespace webgpu {
// MARK: Limits

static constexpr uint32_t kMaxVertexBuffers                 = 16;
static constexpr uint32_t kMaxVertexInputAttributes         = 16;
static constexpr uint32_t kMaxColorAttachments              = 8;
static constexpr uint32_t kMaxColorAttachmentBytesPerSample = 32;
static constexpr uint32_t kMaxBindGroups                    = 4;

static constexpr const char* kRequiredDeviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
};
static constexpr size_t kRequiredDeviceExtensionsCount
    = sizeof(kRequiredDeviceExtensions) / sizeof(kRequiredDeviceExtensions[0]);

bool validate(WGPUSurface surface, WGPUSurfaceConfiguration const* config);
bool validate(WGPUSurface surface, WGPUAdapter adapter, WGPUSurfaceCapabilities* capabilities);
bool validate(WGPUDevice device, WGPURenderPipelineDescriptor const* descriptor);
bool validate(WGPUDevice device, WGPUPipelineLayoutDescriptor const* descriptor);
bool validate(WGPUDevice device, WGPUBindGroupLayoutDescriptor const* descriptor);
bool validate(WGPUDevice device, WGPUBufferDescriptor const* descriptor);

bool validate(WGPUCommandEncoder encoder, WGPURenderPassDescriptor const* descriptor);

bool validate(WGPURenderPassEncoder renderPassEncoder);
bool validate(WGPURenderPassEncoder renderPassEncoder, WGPURenderPipeline pipeline);
bool validate_draw(WGPURenderPassEncoder renderPassEncoder,
                   uint32_t              vertexCount,
                   uint32_t              instanceCount,
                   uint32_t              firstVertex,
                   uint32_t              firstInstance);

bool validate_buffer_get_mapped_range(WGPUBuffer buffer, size_t offset, size_t size);


}  // namespace webgpu
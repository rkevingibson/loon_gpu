#pragma once
#include "gpu/loon_gpu.h"
#include "utilities.h"
#include "volk.h"


namespace loon::gpu {

VkFormat bridge(FORMAT format);
// WGPUTextureFormat bridge(VkFormat fmt);

VkPrimitiveTopology bridge(TOPOLOGY topo);
// VkCullModeFlags     bridge(WGPUCullMode cull);
// VkFrontFace         bridge(WGPUFrontFace ff);
// VkCompareOp         bridge(WGPUCompareFunction fn);
// WGPUPresentMode     bridge(VkPresentModeKHR mode);
// VkPresentModeKHR    bridge(WGPUPresentMode mode);
// VkImageViewType     bridge(WGPUTextureViewDimension dim);
// VkImageAspectFlags  bridge(WGPUTextureAspect aspect);
// VkIndexType         bridge(WGPUIndexFormat fmt);

// VkShaderStageFlags bridge_shader_stage(WGPUShaderStage shader_stage);

// VkSampleCountFlagBits bridge_sample_count(uint32_t sample_count);

// WGPUTextureUsage   bridge_usage_flags(VkImageUsageFlags flags);
// VkImageUsageFlags  bridge_usage_flags(WGPUTextureUsage usage);
// VkBufferUsageFlags bridge_buffer_usage(WGPUBufferUsage usage);

// Stack<WGPUCompositeAlphaMode, 5> bridge_composite_alpha_mode(VkCompositeAlphaFlagsKHR flags);
// VkCompositeAlphaFlagBitsKHR      bridge_composite_alpha_mode(WGPUCompositeAlphaMode mode);

// VkPipelineColorBlendAttachmentState   bridge(const WGPUColorTargetState& state);
// VkPipelineDepthStencilStateCreateInfo bridge(const WGPUDepthStencilState& state);

// VkAttachmentLoadOp  bridge(WGPULoadOp op);
// VkAttachmentStoreOp bridge(WGPUStoreOp op);
// VkClearColorValue   bridge_clear_color_value(WGPUColor color, VkFormat format);

// uint32_t texel_block_width(WGPUTextureFormat format);
// uint32_t texel_block_height(WGPUTextureFormat format);

// VkImageLayout image_layout_from_usage(ResourceUsage u);
};  // namespace loon::gpu
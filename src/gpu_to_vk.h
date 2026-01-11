#pragma once
#include "gpu/loon_gpu.h"
#include "volk.h"


namespace loon::gpu {

VkFormat bridge(FORMAT format);
FORMAT   bridge(VkFormat fmt);


VkPrimitiveTopology bridge(TOPOLOGY topo);
// VkCullModeFlags     bridge(WGPUCullMode cull);
// VkFrontFace         bridge(WGPUFrontFace ff);
// VkCompareOp         bridge(WGPUCompareFunction fn);
PRESENT_MODE     bridge(VkPresentModeKHR mode);
VkPresentModeKHR bridge(PRESENT_MODE mode);

// VkImageViewType     bridge(WGPUTextureViewDimension dim);
// VkImageAspectFlags  bridge(WGPUTextureAspect aspect);
// VkIndexType         bridge(WGPUIndexFormat fmt);

VkPipelineStageFlags2 bridge_pipeline_stage(STAGE stage);
// VkShaderStageFlags bridge_shader_stage(WGPUShaderStage shader_stage);

// VkSampleCountFlagBits bridge_sample_count(uint32_t sample_count);

USAGE_FLAGS       bridge_usage_flags(VkImageUsageFlags flags);
VkImageUsageFlags bridge_usage_flags(USAGE_FLAGS usage);
// VkBufferUsageFlags bridge_buffer_usage(WGPUBufferUsage usage);

// Stack<WGPUCompositeAlphaMode, 5> bridge_composite_alpha_mode(VkCompositeAlphaFlagsKHR flags);
// VkCompositeAlphaFlagBitsKHR      bridge_composite_alpha_mode(WGPUCompositeAlphaMode mode);

VkPipelineColorBlendAttachmentState bridge(const BlendDesc& state);
// VkPipelineDepthStencilStateCreateInfo bridge(const WGPUDepthStencilState& state);

VkAttachmentLoadOp  bridge(LOAD_OP op);
VkAttachmentStoreOp bridge(STORE_OP op);
// VkClearColorValue   bridge_clear_color_value(WGPUColor color, VkFormat format);

// uint32_t texel_block_width(WGPUTextureFormat format);
// uint32_t texel_block_height(WGPUTextureFormat format);

// VkImageLayout image_layout_from_usage(ResourceUsage u);
};  // namespace loon::gpu
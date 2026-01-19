#pragma once
#include "gpu/loon_gpu.h"
#include "volk.h"
#include "vulkan/vulkan_core.h"


namespace loon::gpu {

VkFormat                            bridge(FORMAT format);
VkImageAspectFlags                  aspects_for_format(FORMAT format);
FORMAT                              bridge(VkFormat fmt);
VkPrimitiveTopology                 bridge(TOPOLOGY topo);
PRESENT_MODE                        bridge(VkPresentModeKHR mode);
VkPresentModeKHR                    bridge(PRESENT_MODE mode);
VkPipelineStageFlags2               bridge_pipeline_stage(STAGE_FLAGS stage);
USAGE_FLAGS                         bridge_usage_flags(VkImageUsageFlags flags);
VkImageUsageFlags                   bridge_usage_flags(USAGE_FLAGS usage);
VkImageType                         bridge(TEXTURE tex);
VkImageViewType                     bridge_view_type(TEXTURE tex);
VkPipelineColorBlendAttachmentState bridge(const BlendDesc& state);
VkAttachmentLoadOp                  bridge(LOAD_OP op);
VkAttachmentStoreOp                 bridge(STORE_OP op);
VkCompareOp                         bridge(OP op);
VkImageLayout                       bridge(LAYOUT layout);

};  // namespace loon::gpu
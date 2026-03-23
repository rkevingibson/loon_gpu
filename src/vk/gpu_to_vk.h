#pragma once
#include "gpu/loon_gpu.h"
#include "volk.h"
#include "vulkan/vulkan_core.h"


namespace loon::gpu {

VkFormat                            bridge(Format format);
VkImageAspectFlags                  aspects_for_format(Format format);
Format                              bridge(VkFormat fmt);
VkPrimitiveTopology                 bridge(Topology topo);
PresentMode                         bridge(VkPresentModeKHR mode);
VkPresentModeKHR                    bridge(PresentMode mode);
VkPipelineStageFlags2               bridge_pipeline_stage(StageFlags stage);
UsageFlags                          bridge_usage_flags(VkImageUsageFlags flags);
VkImageUsageFlags                   bridge_usage_flags(UsageFlags usage);
VkImageType                         bridge(TextureType tex);
VkImageViewType                     bridge_view_type(TextureType tex);
VkPipelineColorBlendAttachmentState bridge(const BlendDesc& state);
VkAttachmentLoadOp                  bridge(LoadOp op);
VkAttachmentStoreOp                 bridge(StoreOp op);
VkCompareOp                         bridge(Op op);
VkImageLayout                       bridge(Layout layout);
VkIndexType                         bridge(IndexType t);

};  // namespace loon::gpu
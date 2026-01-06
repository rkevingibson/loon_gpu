#include "commands.h"

#include "device.h"
#include "objects.h"
#include "wgpu_to_vk.h"

namespace webgpu {

void free_command_resources(Command& cmd) {
    // Any command data that has a refcounted pointer needs to be destructed here.
    switch (cmd.type) {
        case CommandType::BeginRenderPass: {
            release_internal(cmd.begin_render_pass.render_pass);
        } break;
        case CommandType::ClearBuffer: {
            release_internal(cmd.clear_buffer.buffer);
        } break;
        case CommandType::CopyBufferToBuffer: {
            release_internal(cmd.copy_buffer_to_buffer.src);
            release_internal(cmd.copy_buffer_to_buffer.dst);
        } break;
        case CommandType::ResolveQuerySet: {
            release_internal(cmd.resolve_query_set.querySet);
        } break;
        case CommandType::DrawIndexedIndirect: {
            release_internal(cmd.draw_indexed_indirect.indirect_buffer);
        } break;
        case CommandType::DrawIndirect: {
            release_internal(cmd.draw_indirect.indirect_buffer);
        } break;
        case CommandType::SetBindGroup: {
            release_internal(cmd.set_bind_group.group);
        } break;
        case CommandType::SetIndexBuffer: {
            release_internal(cmd.set_index_buffer.buffer);
        } break;
        case CommandType::SetRenderPipeline: {
            release_internal(cmd.set_render_pipeline.pipeline);
        } break;
        case CommandType::SetVertexBuffer: {
            release_internal(cmd.set_vertex_buffer.buffer);
        } break;
        case CommandType::DispatchWorkgroupsIndirect: {
            release_internal(cmd.dispatch_workgroups_indirect.indirectBuffer);
        } break;
        case CommandType::SetComputePipeline: {
            release_internal(cmd.set_compute_pipeline.pipeline);
        } break;
        default: break;
    }
}

CommandList::CommandList(Allocator allocator) : m_allocator(allocator) {}

bool CommandList::add(Command&& cmd) {
    if (m_size == m_capacity) {
        // If reserve fails, need to return so API can throw an OOM error.
        if (!reserve(grow_capacity(m_size + 1))) { return false; }
    }

    ::new (m_data + m_size) Command(std::forward<Command>(cmd));
    ++m_size;
    return true;
}

bool CommandList::reserve(size_t new_capacity) {
    if (new_capacity <= m_capacity) return true;
    const auto new_blk = m_allocator.alloc(new_capacity * sizeof(Command));
    if (new_blk.ptr == nullptr) { return false; }

    std::uninitialized_move(m_data, m_data + m_size, reinterpret_cast<Command*>(new_blk.ptr));
    m_allocator.free({.ptr = m_data, .len = static_cast<uint32_t>(m_capacity * sizeof(Command))});

    m_data     = reinterpret_cast<Command*>(new_blk.ptr);
    m_capacity = new_capacity;
    return true;
}

CommandList::~CommandList() {
    reset();
    m_allocator.free({.ptr = m_data, .len = static_cast<uint32_t>(m_capacity * sizeof(Command))});
}

CommandList::CommandList(CommandList&& other) : CommandList() {
    swap(*this, other);
}

CommandList& CommandList::operator=(CommandList&& other) {
    swap(*this, other);
    return *this;
}

void swap(CommandList& a, CommandList& b) {
    using std::swap;
    swap(a.m_capacity, b.m_capacity);
    swap(a.m_size, b.m_size);
    swap(a.m_data, b.m_data);
    swap(a.m_allocator, b.m_allocator);
}

void CommandList::reset() {
    for (size_t i = 0; i < m_size; ++i) { free_command_resources(m_data[i]); }
    m_size = 0;
}

CommandEncodingError CommandsMixin::add(Command&& cmd) {
    // TODO: Check state validity
    if (!cmd_list.add(std::forward<Command>(cmd))) { return CommandEncodingError::OutOfMemory; }

    return CommandEncodingError::Success;
}

// MARK: RenderCommandsMixin

RenderCommandsMixin::RenderCommandsMixin(Allocator alloc) : usage_scope(alloc) {}

static bool compatible_usage_set(loon::gpu::ResourceUsage u) {
    // A usage set is "compatible" if any of the following are true:
    // - Each usage is storage
    // - Each usage is attachment
    // - Each usage is input, constant, storage-read, or attachment-read
    // See https://www.w3.org/TR/webgpu/#programming-model-resource-usages
    static constexpr uint8_t kCompatibleUsageList
        = kUsageInput | kUsageConstant | kUsageStorageRead | kUsageAttachmentRead;
    return (u == kUsageStorage) || (u == kUsageAttachment)
           || ((u & kCompatibleUsageList) != 0 && (u & ~kCompatibleUsageList) == 0);
}

static bool needs_barrier(const loon::gpu::ResourceUsage prev_usage,
                          const loon::gpu::ResourceUsage next_usage) {
    // Need a barrier if the previous usage was a read and next usage is a write, or vice-versa.
    static constexpr uint16_t kWriteOpsMask = kUsageStorage | kUsageAttachment | kUsageTransferDst;
    static constexpr uint16_t kReadOnlyOpsMask = kUsageInput | kUsageConstant | kUsageStorageRead
                                                 | kUsageAttachmentRead | kUsageTransferSrc;
    const auto read_only
        = [](loon::gpu::ResourceUsage u) -> bool { return (u & kWriteOpsMask) == 0; };
    const auto read_write
        = [](loon::gpu::ResourceUsage u) -> bool { return (u & kReadOnlyOpsMask) == 0; };

    return (read_only(prev_usage) && read_write(next_usage))
           || (read_write(prev_usage) && read_only(next_usage));
}

static VkAccessFlags2 usage_to_access_mask(ResourceUsage u) {
    VkAccessFlags2 result = VK_ACCESS_2_NONE;
    result |= (u & kUsageInput)
                  ? (VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT)
                  : 0;
    result |= (u & kUsageConstant) ? (VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT
                                      | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT)
                                   : 0;
    result |= (u & kUsageStorage)
                  ? (VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)
                  : 0;
    result |= (u & kUsageStorageRead) ? (VK_ACCESS_2_SHADER_STORAGE_READ_BIT) : 0;
    result |= (u & kUsageAttachment)
                  ? (VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT
                     | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)
                  : 0;
    result |= (u & kUsageAttachmentRead)
                  ? (VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT)
                  : 0;
    result |= (u & kUsageTransferDst) ? (VK_ACCESS_2_TRANSFER_WRITE_BIT) : 0;
    result |= (u & kUsageTransferSrc) ? (VK_ACCESS_2_TRANSFER_READ_BIT) : 0;
    return result;
};

static VkPipelineStageFlags2 usage_to_pipeline_stage(ResourceUsage u) {
    // TODO: Modify this to take the shader stage too, for better barriers
    switch (u) {
        case kUsageUndefined: return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        case kUsageInput: return VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        case kUsageConstant:
        case kUsageStorage:
        case kUsageStorageRead:
            return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                   | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            // TODO: These aren't specific enough - depends on image aspect, depth testing, etc.
        case kUsageAttachment:
            return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
                   | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case kUsageAttachmentRead:
            return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
                   | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case kUsagePresent: return VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT; break;
        case kUsageTransferSrc:
        case kUsageTransferDst: return VK_PIPELINE_STAGE_2_TRANSFER_BIT; break;
    }
}

static VkImageMemoryBarrier2 image_memory_barrier(WGPUDevice               device,
                                                  VkImage                  vk_image,
                                                  loon::gpu::ResourceUsage prev,
                                                  const loon::gpu::UsageScope::TextureUsage& next) {
    return VkImageMemoryBarrier2 {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr,
        .srcStageMask        = usage_to_pipeline_stage(prev),
        .srcAccessMask       = usage_to_access_mask(prev),
        .dstStageMask        = usage_to_pipeline_stage(next.usage),
        .dstAccessMask       = usage_to_access_mask(next.usage),
        .oldLayout           = loon::gpu::image_layout_from_usage(prev),
        .newLayout           = loon::gpu::image_layout_from_usage(next.usage),
        .srcQueueFamilyIndex = device->get_queue_family(),
        .dstQueueFamilyIndex = device->get_queue_family(), .image = vk_image,
        .subresourceRange = {
            .aspectMask     = loon::gpu::bridge(next.aspect),
            .baseMipLevel   = next.min_mip_level,
            .levelCount     = next.max_mip_level - next.min_mip_level + 1,
            .baseArrayLayer = next.min_array_layer,
            .layerCount     = next.max_array_layer - next.min_array_layer + 1,
        },
    };
}

static VkBufferMemoryBarrier2 buffer_memory_barrier(
    WGPUDevice                                device,
    WGPUBuffer                                buffer,
    loon::gpu::ResourceUsage                  prev,
    const loon::gpu::UsageScope::BufferUsage& next) {
    return {
        .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .pNext               = nullptr,
        .srcStageMask        = usage_to_pipeline_stage(prev),
        .srcAccessMask       = usage_to_access_mask(prev),
        .dstStageMask        = usage_to_pipeline_stage(next.usage),
        .dstAccessMask       = usage_to_access_mask(next.usage),
        .srcQueueFamilyIndex = device->get_queue_family(),
        .dstQueueFamilyIndex = device->get_queue_family(),
        .buffer              = buffer->vk_buffer,
        .offset              = 0,
        .size                = buffer->size,
    };
}

void record_pipeline_barriers(VkCommandBuffer              cmd_buffer,
                              WGPUDevice                   device,
                              const loon::gpu::UsageScope& previous_scope,
                              const loon::gpu::UsageScope& new_scope) {
    loon::gpu::ArenaVector<VkImageMemoryBarrier2> image_barriers(device->get_thread_local_arena());
    for (auto& tex_usage : new_scope.tex_usages) {
        auto previous_usage = previous_scope.tex_usages.find(tex_usage.key);
        if (previous_usage != nullptr
            && needs_barrier(previous_usage->value.usage, tex_usage.value.usage)) {
            // TODO: WGPUShaderStage - This is likely insufficient, in webgpu we only have 3 shader
            // stages, but e.g. attachment output is a separate stage in vulkan.
            image_barriers.push(image_memory_barrier(device,
                                                     tex_usage.key->vk_image,
                                                     previous_usage->value.usage,
                                                     tex_usage.value));
        }
    }

    loon::gpu::ArenaVector<VkBufferMemoryBarrier2> buffer_barriers(
        device->get_thread_local_arena());
    for (auto& buf_usage : new_scope.buffer_usages) {
        auto prev_usage = previous_scope.buffer_usages.find(buf_usage.key);
        if (prev_usage && needs_barrier(prev_usage->value.usage, buf_usage.value.usage)) {
            buffer_barriers.push(buffer_memory_barrier(device,
                                                       buf_usage.key,
                                                       prev_usage->value.usage,
                                                       buf_usage.value));
        }
    }

    if (buffer_barriers.size() == 0 && image_barriers.size() == 0) { return; }

    VkDependencyInfo dependency_info{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                    = nullptr,
        .dependencyFlags          = 0,
        .memoryBarrierCount       = 0,
        .pMemoryBarriers          = nullptr,
        .bufferMemoryBarrierCount = buffer_barriers.size(),
        .pBufferMemoryBarriers    = buffer_barriers.data(),
        .imageMemoryBarrierCount  = image_barriers.size(),
        .pImageMemoryBarriers     = image_barriers.data(),
    };
    device->vk_api.vkCmdPipelineBarrier2(cmd_buffer, &dependency_info);
}

bool record_pre_submit_synchronization_cmd(VkCommandBuffer              cmd_buffer,
                                           WGPUDevice                   device,
                                           const loon::gpu::UsageScope& first_usages) {
    loon::gpu::ArenaVector<VkImageMemoryBarrier2> image_barriers(device->get_thread_local_arena());
    for (auto& tex_usage : first_usages.tex_usages) {
        auto previous_usage = tex_usage.key->last_submitted_usage;
        if (needs_barrier(previous_usage, tex_usage.value.usage)
            || previous_usage == loon::gpu::kUsageUndefined) {
            // TODO: WGPUShaderStage - This is likely insufficient, in webgpu we only have 3 shader
            // stages, but e.g. attachment output is a separate stage in vulkan.
            image_barriers.push(image_memory_barrier(device,
                                                     tex_usage.key->vk_image,
                                                     previous_usage,
                                                     tex_usage.value));
        }
    }

    loon::gpu::ArenaVector<VkBufferMemoryBarrier2> buffer_barriers(
        device->get_thread_local_arena());

    for (auto& buf_usage : first_usages.buffer_usages) {
        auto prev_usage = buf_usage.key->last_submitted_usage;
        if (needs_barrier(prev_usage, buf_usage.value.usage)) {
            buffer_barriers.push(
                buffer_memory_barrier(device, buf_usage.key, prev_usage, buf_usage.value));
        }
    }

    if (buffer_barriers.size() == 0 && image_barriers.size() == 0) { return false; }

    VkCommandBufferBeginInfo begin_info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    auto& vk_api = device->vk_api;
    WGPU_VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));
    VkDependencyInfo dependency_info{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                    = nullptr,
        .dependencyFlags          = 0,
        .memoryBarrierCount       = 0,
        .pMemoryBarriers          = nullptr,
        .bufferMemoryBarrierCount = buffer_barriers.size(),
        .pBufferMemoryBarriers    = buffer_barriers.data(),
        .imageMemoryBarrierCount  = image_barriers.size(),
        .pImageMemoryBarriers     = image_barriers.data(),
    };
    vk_api.vkCmdPipelineBarrier2(cmd_buffer, &dependency_info);
    WGPU_VK_CHECK(vkEndCommandBuffer(cmd_buffer));

    return true;
}

void record_pipeline_barrier(VkCommandBuffer              cmd_buffer,
                             WGPUDevice                   device,
                             const loon::gpu::UsageScope& scope,
                             WGPUBuffer                   buffer,
                             ResourceUsage                usage) {
    auto prev_usage = scope.buffer_usages.find(buffer);
    if (prev_usage && needs_barrier(prev_usage->value.usage, usage)) {
        VkBufferMemoryBarrier2 memory_barrier{
            .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext               = nullptr,
            .srcStageMask        = usage_to_pipeline_stage(prev_usage->value.usage),
            .srcAccessMask       = usage_to_access_mask(prev_usage->value.usage),
            .dstStageMask        = usage_to_pipeline_stage(usage),
            .dstAccessMask       = usage_to_access_mask(usage),
            .srcQueueFamilyIndex = device->get_queue_family(),
            .dstQueueFamilyIndex = device->get_queue_family(),
            .buffer              = buffer->vk_buffer,
            .offset              = 0,
            .size                = buffer->size,
        };

        VkDependencyInfo dependency_info{
            .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext                    = nullptr,
            .dependencyFlags          = 0,
            .memoryBarrierCount       = 0,
            .pMemoryBarriers          = nullptr,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers    = &memory_barrier,
            .imageMemoryBarrierCount  = 0,
            .pImageMemoryBarriers     = nullptr,
        };
        device->vk_api.vkCmdPipelineBarrier2(cmd_buffer, &dependency_info);
    }
}
}  // namespace webgpu

// MARK: CommandEncoder

WGPUCommandEncoderImpl::~WGPUCommandEncoderImpl() = default;

void swap(WGPUCommandEncoderImpl& a, WGPUCommandEncoderImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(a.commands_mixin, b.commands_mixin);
}

WGPURenderPassEncoder WGPUCommandEncoderImpl::begin_render_pass(
    WGPURenderPassDescriptor const* descriptor) {
    if (!loon::gpu::validate(this, descriptor)) { return nullptr; }

    auto* commands  = &commands_mixin;
    commands->state = loon::gpu::CommandEncodingState::RecordingRenderPass;

    auto render_pass                      = device->m_render_passes.make(device, descriptor->label);
    render_pass->render_commands          = loon::gpu::RenderCommandsMixin(device->get_allocator());
    render_pass->encoder                  = this;
    render_pass->attachment_size          = VkExtent2D{0, 0};
    render_pass->color_attachments        = {};
    render_pass->depth_stencil_attachment = {};
    render_pass->has_depth_stencil_attachment = false;
    render_pass->commands_mixin               = commands;

    WGPUExtent3D render_extent{};

    for (uint32_t i = 0; i < descriptor->colorAttachmentCount; ++i) {
        render_pass->color_attachments.push(descriptor->colorAttachments[i]);
        if (render_pass->color_attachments[i].view != nullptr) {
            render_pass->color_attachments[i].view->add_ref_internal();
            render_extent = render_pass->color_attachments[i].view->render_extent;
            render_pass->render_commands.usage_scope.add(render_pass->color_attachments[i].view,
                                                         loon::gpu::ResourceUsage::kUsageAttachment,
                                                         WGPUShaderStage_Fragment);
        }

        if (render_pass->color_attachments[i].resolveTarget != nullptr) {
            render_pass->color_attachments[i].resolveTarget->add_ref_internal();
        }
    }

    if (descriptor->depthStencilAttachment) {
        render_pass->has_depth_stencil_attachment = true;
        render_pass->depth_stencil_attachment     = *descriptor->depthStencilAttachment;
        render_pass->depth_stencil_attachment.view->add_ref_internal();
        // TODO: Need to separate depth and stencil subresources somehow
        render_pass->render_commands.usage_scope.add(
            render_pass->depth_stencil_attachment.view,
            render_pass->depth_stencil_attachment.depthReadOnly
                    && render_pass->depth_stencil_attachment.stencilReadOnly
                ? loon::gpu::ResourceUsage::kUsageAttachmentRead
                : loon::gpu::ResourceUsage::kUsageAttachment,
            WGPUShaderStage_Fragment);
        render_extent = render_pass->depth_stencil_attachment.view->render_extent;
    }
    render_pass->attachment_size = {.width = render_extent.width, .height = render_extent.height};

    render_pass->add_ref_internal();
    commands->add(loon::gpu::Command{
        .begin_render_pass{
            .render_pass = render_pass,
        },
        .type = loon::gpu::CommandType::BeginRenderPass,
    });

    return loon::gpu::return_with_ownership(render_pass);
}

WGPUCommandBuffer WGPUCommandEncoderImpl::finish(
    WGPU_NULLABLE WGPUCommandBufferDescriptor const* descriptor) {
    const auto& vk_api          = device->vk_api;
    bool        touches_surface = false;

    loon::gpu::UsageScope first_usages(device->get_allocator());
    loon::gpu::UsageScope last_usages(device->get_allocator());

    uint64_t                 timeline_value = device->queue.get_current_timeline_value();
    WGPUCommandBuffer        result         = device->allocate_command_buffer();
    VkCommandBuffer          cmd_buffer     = result->vk_cmd_buffers[1];
    VkCommandBufferBeginInfo begin_info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    WGPU_VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));

    for (const auto& c : commands_mixin.cmd_list) {
        using namespace webgpu;
        switch (c.type) {
                // High-level commands
            case CommandType::BeginComputePass:
            case CommandType::EndComputePass: break;
            case CommandType::BeginRenderPass: {
                auto& render_pass = c.begin_render_pass.render_pass;
                // Any synchronization needs to happen here - image transitions, etc
                const auto& render_pass_scope = render_pass->render_commands.usage_scope;
                loon::gpu::record_pipeline_barriers(cmd_buffer,
                                                    device,
                                                    last_usages,
                                                    render_pass_scope);
                render_pass_scope.update_first_last_usages(first_usages, last_usages);

                loon::gpu::Stack<VkRenderingAttachmentInfo, loon::gpu::kMaxColorAttachments>
                    color_attachments;
                for (const auto& attachment : render_pass->color_attachments) {
                    if (attachment.view->texture->is_surface_image) { touches_surface = true; }

                    color_attachments.push({
                        .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                        .pNext              = nullptr,
                        .imageView          = attachment.view->vk_image_view,
                        .imageLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        .resolveMode        = VK_RESOLVE_MODE_NONE,
                        .resolveImageView   = VK_NULL_HANDLE,
                        .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .loadOp             = loon::gpu::bridge(attachment.loadOp),
                        .storeOp            = loon::gpu::bridge(attachment.storeOp),
                        .clearValue
                        = {.color = loon::gpu::bridge_clear_color_value(
                               attachment.clearValue,
                               attachment.view->vk_format)},  // Value depends on the format of
                                                              // the image view.
                    });
                }

                VkRenderingInfo rendering_info{
                    .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .renderArea           = VkRect2D{.offset = {0,0},.extent = render_pass->attachment_size, },
                    .layerCount           = 1,
                    .viewMask             = 0,
                    .colorAttachmentCount = color_attachments.size(),
                    .pColorAttachments    = color_attachments.data(),
                    .pDepthAttachment     = nullptr,
                    .pStencilAttachment   = nullptr,
                };
                vk_api.vkCmdBeginRendering(cmd_buffer, &rendering_info);

                // TODO: May need to set default viewport/scissor rectangle to meet the spec.

            } break;
            case CommandType::EndRenderPass: vk_api.vkCmdEndRendering(cmd_buffer); break;
            case CommandType::ClearBuffer: break;
            case CommandType::CopyBufferToBuffer: {
                const auto& cmd = c.copy_buffer_to_buffer;
                // Synchronization: May need a barrier before the copy
                record_pipeline_barrier(cmd_buffer,
                                        device,
                                        last_usages,
                                        cmd.src,
                                        ResourceUsage::kUsageTransferSrc);
                record_pipeline_barrier(cmd_buffer,
                                        device,
                                        last_usages,
                                        cmd.dst,
                                        ResourceUsage::kUsageTransferDst);
                first_usages.try_add(cmd.src,
                                     ResourceUsage::kUsageTransferSrc,
                                     WGPUShaderStage_None);
                last_usages.add(cmd.src, ResourceUsage::kUsageTransferSrc, WGPUShaderStage_None);
                first_usages.try_add(cmd.dst,
                                     ResourceUsage::kUsageTransferDst,
                                     WGPUShaderStage_None);
                last_usages.add(cmd.dst, ResourceUsage::kUsageTransferDst, WGPUShaderStage_None);

                VkBufferCopy region{
                    .srcOffset = cmd.src_offset,
                    .dstOffset = cmd.dst_offset,
                    .size      = cmd.size,
                };
                vk_api.vkCmdCopyBuffer(cmd_buffer,
                                       cmd.src->vk_buffer,
                                       cmd.dst->vk_buffer,
                                       1,
                                       &region);
            } break;
            case CommandType::CopyBufferToTexture:
            case CommandType::CopyTextureToBuffer:
            case CommandType::CopyTextureToTexture:
            case CommandType::InsertDebugMarker:
            case CommandType::PushDebugGroup:
            case CommandType::PopDebugGroup:
            case CommandType::ResolveQuerySet: break;
            case CommandType::Draw: {
                const auto& draw = c.draw;
                vk_api.vkCmdDraw(cmd_buffer,
                                 draw.vertex_count,
                                 draw.instance_count,
                                 draw.first_vertex,
                                 draw.first_instance);
            } break;
            case CommandType::DrawIndexed: {
                const auto& draw_indexed = c.draw_indexed;
                vk_api.vkCmdDrawIndexed(cmd_buffer,
                                        draw_indexed.index_count,
                                        draw_indexed.instance_count,
                                        draw_indexed.first_index,
                                        static_cast<int32_t>(draw_indexed.base_vertex),
                                        draw_indexed.first_instance);
            } break;
            case CommandType::DrawIndexedIndirect: {
                const auto& draw_indexed_indirect = c.draw_indexed_indirect;
                vk_api.vkCmdDrawIndexedIndirect(cmd_buffer,
                                                draw_indexed_indirect.indirect_buffer->vk_buffer,
                                                draw_indexed_indirect.indirect_offset,
                                                1,
                                                0);
            } break;
            case CommandType::DrawIndirect: {
                const auto& draw_indirect = c.draw_indirect;
                vk_api.vkCmdDrawIndirect(cmd_buffer,
                                         draw_indirect.indirect_buffer->vk_buffer,
                                         draw_indirect.indirect_offset,
                                         1,
                                         0);
            } break;
            case CommandType::BeginOcclusionQuery:
            case CommandType::EndOcclusionQuery:
            case CommandType::ExecuteBundles:
            case CommandType::SetBindGroup:
            case CommandType::SetBlendConstant: break;
            case CommandType::SetIndexBuffer: {
                const auto& cmd = c.set_index_buffer;
                vk_api.vkCmdBindIndexBuffer(cmd_buffer,
                                            cmd.buffer->vk_buffer,
                                            cmd.offset,
                                            loon::gpu::bridge(cmd.format));
            } break;
            case CommandType::SetRenderPipeline: {
                const auto& set_render_pipeline = c.set_render_pipeline;
                vk_api.vkCmdBindPipeline(cmd_buffer,
                                         VK_PIPELINE_BIND_POINT_GRAPHICS,
                                         set_render_pipeline.pipeline->vk_pipeline);
            } break;
            case CommandType::SetScissorRect:;
            case CommandType::SetStencilReference: break;
            case CommandType::SetVertexBuffer: {
                const auto& set_vertex_buffer = c.set_vertex_buffer;
                vk_api.vkCmdBindVertexBuffers(cmd_buffer,
                                              set_vertex_buffer.slot,
                                              1,
                                              &set_vertex_buffer.buffer->vk_buffer,
                                              &set_vertex_buffer.offset);
            } break;
            case CommandType::SetViewport: {
                const auto&      set_viewport = c.set_viewport;
                const VkViewport viewport{
                    .x        = set_viewport.x,
                    .y        = set_viewport.y,
                    .width    = set_viewport.width,
                    .height   = set_viewport.height,
                    .minDepth = set_viewport.min_depth,
                    .maxDepth = set_viewport.max_depth,
                };
                vk_api.vkCmdSetViewportWithCount(cmd_buffer, 1, &viewport);
            } break;
            case CommandType::DispatchWorkgroups:
            case CommandType::DispatchWorkgroupsIndirect:
            case CommandType::SetComputePipeline: break;
            default: assert(false && "Unsupported command type"); break;
        }
    }

    WGPU_VK_CHECK(vkEndCommandBuffer(cmd_buffer));

    if (descriptor) { result->set_label(descriptor->label); }
    result->first_usages          = std::move(first_usages);
    result->last_usages           = std::move(last_usages);
    result->touches_surface_image = touches_surface;
    add_ref_internal();
    result->cmd_encoder = this;

    return loon::gpu::return_with_ownership(result);
}

// MARK: RenderPassEncoder

WGPURenderPassEncoderImpl::~WGPURenderPassEncoderImpl()
    = default;  // TODO: Is this right? Any references to cleanup?

void swap(WGPURenderPassEncoderImpl& a, WGPURenderPassEncoderImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(a.render_commands, b.render_commands);
    swap(a.encoder, b.encoder);
    swap(a.attachment_size, b.attachment_size);
    swap(a.color_attachments, b.color_attachments);
    swap(a.depth_stencil_attachment, b.depth_stencil_attachment);
    swap(a.commands_mixin, b.commands_mixin);
}

// MARK: ComputePassEncoder

WGPUComputePassEncoderImpl::~WGPUComputePassEncoderImpl() = default;

void swap(WGPUComputePassEncoderImpl& a, WGPUComputePassEncoderImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(a.commands_mixin, b.commands_mixin);
}

// MARK: CommandBuffer

WGPUCommandBufferImpl::~WGPUCommandBufferImpl() {
    reset();
}

void swap(WGPUCommandBufferImpl& a, WGPUCommandBufferImpl& b) {
    using std::swap;
    swap(static_cast<WGPUObjectBase&>(a), static_cast<WGPUObjectBase&>(b));
    swap(a.cmd_pool, b.cmd_pool);
    swap(a.vk_cmd_buffers, b.vk_cmd_buffers);
    swap(a.first_usages, b.first_usages);
    swap(a.last_usages, b.last_usages);
    swap(a.touches_surface_image, b.touches_surface_image);
    swap(a.cmd_encoder, b.cmd_encoder);
    swap(a.submitted_timeline_value, b.submitted_timeline_value);
}

void WGPUCommandBufferImpl::reset() {
    if (cmd_encoder) { loon::gpu::release_internal(cmd_encoder); }
    auto& vk_api = device->vk_api;
    WGPU_VK_CHECK(
        vkResetCommandBuffer(vk_cmd_buffers[0], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT));
    WGPU_VK_CHECK(
        vkResetCommandBuffer(vk_cmd_buffers[1], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT));
}
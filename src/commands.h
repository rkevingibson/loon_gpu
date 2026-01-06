#pragma once
#include <cstdint>

#include "objects.h"
#include "utilities.h"
#include "validation.h"
#include "webgpu/webgpu.h"

namespace webgpu {

enum class CommandType : uint8_t {
    Default = 0,
    // High-level commands
    BeginComputePass,
    EndComputePass,
    BeginRenderPass,
    EndRenderPass,
    ClearBuffer,
    CopyBufferToBuffer,
    CopyBufferToTexture,
    CopyTextureToBuffer,
    CopyTextureToTexture,

    // Debug Groups
    InsertDebugMarker,
    PushDebugGroup,
    PopDebugGroup,
    ResolveQuerySet,

    // Render Pass Commands
    Draw,
    DrawIndexed,
    DrawIndexedIndirect,
    DrawIndirect,
    BeginOcclusionQuery,
    EndOcclusionQuery,
    ExecuteBundles,

    SetBindGroup,
    SetBlendConstant,
    SetIndexBuffer,
    SetRenderPipeline,
    SetScissorRect,
    SetStencilReference,
    SetVertexBuffer,
    SetViewport,

    // Compute Pass Commands
    DispatchWorkgroups,
    DispatchWorkgroupsIndirect,
    SetComputePipeline,
};

struct CmdBeginComputePass {
    // TODO: Implement
};

struct CmdEndComputePass {
    // TODO: Implement
};

struct CmdBeginRenderPass {
    WGPURenderPassEncoder render_pass;
};

struct CmdEndRenderPass {
    // TODO: Implement
};

struct CmdClearBuffer {
    WGPUBuffer buffer;
    uint64_t   offset;
    uint64_t   size;
};

struct CmdCopyBufferToBuffer {
    WGPUBuffer src;
    WGPUBuffer dst;
    uint64_t   src_offset;
    uint64_t   dst_offset;
    uint64_t   size;
};

struct CmdCopyBufferToTexture {
    // TODO: Implement
};

struct CmdCopyTextureToBuffer {
    // TODO: Implement
};

struct CmdCopyTextureToTexture {
    // TODO: Implement
};

struct CmdInsertDebugMarker {
    WGPUStringView marker_label;
};

struct CmdPushDebugGroup {
    WGPUStringView group_label;
};

struct CmdPopDebugGroup {};

struct CmdResolveQuerySet {
    WGPUQuerySet querySet;
    uint32_t     firstQuery;
    uint32_t     queryCount;
    WGPUBuffer   destination;
    uint64_t     destinationOffset;
};

struct CmdDraw {
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
};

struct CmdDrawIndexed {
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    uint32_t base_vertex;
    uint32_t first_instance;
};

struct CmdDrawIndexedIndirect {
    WGPUBuffer indirect_buffer;
    uint64_t   indirect_offset;
};

struct CmdDrawIndirect {
    WGPUBuffer indirect_buffer;
    uint64_t   indirect_offset;
};

struct CmdBeginOcclusionQuery {
    uint32_t query_index;
};

struct CmdEndOcclusionQuery {};

struct CmdExecuteBundles {
    size_t                  bundle_count;
    WGPURenderBundle const* bundles;
};

struct CmdSetBindGroup {
    uint32_t        group_index;
    WGPUBindGroup   group;
    uint32_t        dynamic_offset_count;
    uint32_t const* dynamic_offsets;
};

struct CmdSetBlendConstants {
    WGPUColor color;
};

struct CmdSetIndexBuffer {
    WGPUBuffer      buffer;
    WGPUIndexFormat format;
    uint64_t        offset;
    uint64_t        size;
};

struct CmdSetRenderPipeline {
    WGPURenderPipeline pipeline;
};

struct CmdSetScissorRect {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
};

struct CmdSetStencilReference {
    uint32_t reference;
};

struct CmdSetVertexBuffer {
    uint32_t   slot;
    WGPUBuffer buffer;
    uint64_t   offset;
    uint64_t   size;
};

struct CmdSetViewport {
    float x;
    float y;
    float width;
    float height;
    float min_depth;
    float max_depth;
};

struct CmdDispatchWorkgroups {
    uint32_t workgroupCountX;
    uint32_t workgroupCountY;
    uint32_t workgroupCountZ;
};

struct CmdDispatchWorkgroupsIndirect {
    WGPUBuffer indirectBuffer;
    uint64_t   indirectOffset;
};

struct CmdSetComputePipeline {
    WGPUComputePipeline pipeline;
};

struct Command {
    union {
        CmdBeginComputePass           begin_compute_pass;
        CmdEndComputePass             end_compute_pass;
        CmdBeginRenderPass            begin_render_pass;
        CmdEndRenderPass              end_render_pass;
        CmdClearBuffer                clear_buffer;
        CmdCopyBufferToBuffer         copy_buffer_to_buffer;
        CmdCopyBufferToTexture        copy_buffer_to_texture;
        CmdCopyTextureToBuffer        copy_texture_to_buffer;
        CmdCopyTextureToTexture       copy_texture_to_texture;
        CmdInsertDebugMarker          insert_debug_marker;
        CmdPushDebugGroup             push_debug_group;
        CmdPopDebugGroup              pop_debug_group;
        CmdResolveQuerySet            resolve_query_set;
        CmdDraw                       draw;
        CmdDrawIndexed                draw_indexed;
        CmdDrawIndexedIndirect        draw_indexed_indirect;
        CmdDrawIndirect               draw_indirect;
        CmdBeginOcclusionQuery        begin_occlusion_query;
        CmdEndOcclusionQuery          end_occlusion_query;
        CmdExecuteBundles             execute_bundles;
        CmdSetBindGroup               set_bind_group;
        CmdSetBlendConstants          set_blend_constants;
        CmdSetIndexBuffer             set_index_buffer;
        CmdSetRenderPipeline          set_render_pipeline;
        CmdSetScissorRect             set_scissor_rect;
        CmdSetStencilReference        set_stencil_reference;
        CmdSetVertexBuffer            set_vertex_buffer;
        CmdSetViewport                set_viewport;
        CmdDispatchWorkgroups         dispatch_workgroups;
        CmdDispatchWorkgroupsIndirect dispatch_workgroups_indirect;
        CmdSetComputePipeline         set_compute_pipeline;
    };
    CommandType type = CommandType::Default;
};

static constexpr size_t kCommandSize = sizeof(Command);

class CommandList {
   public:
    CommandList() = default;
    CommandList(Allocator allocator);
    CommandList(const CommandList& other) = delete;
    CommandList(CommandList&& other);
    CommandList& operator=(const CommandList& other) = delete;
    CommandList& operator=(CommandList&& other);
    ~CommandList();

    friend void swap(CommandList& a, CommandList& b);

    bool     add(Command&& cmd);
    Command* begin() { return m_data; }
    Command* end() { return m_data + m_size; }

    bool reserve(size_t size);

    void reset();

   private:
    size_t grow_capacity(size_t sz) const {
        const size_t new_capacity = m_capacity ? (m_capacity + m_capacity / 2) : 8;
        return new_capacity > sz ? new_capacity : sz;
    }
    // The command list really only needs to support three operations - appending, iterating and
    // resetting. For simplicity, let's just do a vector-like approach for now, but in theory we
    // don't need contiguous storage and could avoid reallocating on grow by using a list-like
    // approach.

    size_t    m_capacity = 0;
    size_t    m_size     = 0;
    Command*  m_data     = nullptr;
    Allocator m_allocator;
};

enum class CommandEncodingError {
    Success = 0,
    OutOfMemory,
};

enum class CommandEncodingState {
    Open,
    RecordingRenderPass,
    RecordingComputePass,
};

struct CommandsMixin {
    CommandsMixin() = default;
    CommandsMixin(Allocator allocator) : cmd_list(allocator) {}

    CommandEncodingError add(Command&& cmd);

    CommandEncodingState state = CommandEncodingState::Open;
    CommandList          cmd_list;
};

struct RenderPassLayout {
    loon::gpu::Stack<WGPUTextureFormat, loon::gpu::kMaxColorAttachments> color_formats;
    WGPUTextureFormat                                                    depth_stencil_format;
    uint32_t                                                             sample_count = 1;
};

struct RenderCommandsMixin {
    RenderCommandsMixin() = default;
    RenderCommandsMixin(Allocator allocator);

    RenderPassLayout   layout;
    bool               depth_read_only;
    bool               stencil_read_only;
    UsageScope         usage_scope;
    WGPURenderPipeline pipeline     = nullptr;
    WGPUBuffer         index_buffer = nullptr;
    WGPUIndexFormat    index_format;
    uint64_t           index_buffer_offset;
    uint64_t           index_buffer_size;
    WGPUBuffer         vertex_buffers[loon::gpu::kMaxVertexBuffers]      = {0};
    uint64_t           vertex_buffer_sizes[loon::gpu::kMaxVertexBuffers] = {0};
    uint64_t           draw_count                                        = 0;
};

void record_pipeline_barriers(VkCommandBuffer              cmd_buffer,
                              WGPUDevice                   device,
                              const loon::gpu::UsageScope& previous_scope,
                              const loon::gpu::UsageScope& new_scope);

bool record_pre_submit_synchronization_cmd(VkCommandBuffer              cmd_buffer,
                                           WGPUDevice                   device,
                                           const loon::gpu::UsageScope& first_usages);
}  // namespace webgpu

struct WGPUCommandEncoderImpl : WGPUObjectBase {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPUCommandEncoderImpl);

    WGPURenderPassEncoder begin_render_pass(WGPURenderPassDescriptor const* descriptor);

    WGPUCommandBuffer finish(WGPU_NULLABLE WGPUCommandBufferDescriptor const* descriptor);

    loon::gpu::CommandsMixin commands_mixin;
};

struct WGPURenderPassEncoderImpl : WGPUObjectBase {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPURenderPassEncoderImpl);

    loon::gpu::RenderCommandsMixin render_commands;
    WGPUCommandEncoder             encoder;
    VkExtent2D                     attachment_size;

    loon::gpu::Stack<WGPURenderPassColorAttachment, loon::gpu::kMaxColorAttachments>
                                         color_attachments;
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    bool                                 has_depth_stencil_attachment;

    loon::gpu::CommandsMixin* commands_mixin;
};

struct WGPUComputePassEncoderImpl : WGPUObjectBase {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPUComputePassEncoderImpl);
    loon::gpu::CommandsMixin* commands_mixin;
};

struct WGPUCommandBufferImpl : WGPUObjectBase {
    WGPU_DEVICE_OBJECT_DEFAULT_OPERATORS(WGPUCommandBufferImpl);

    void reset();

    // We have 2 command buffers - one for synchronization that gets recorded during QueueSubmit,
    // and one for the actual commands.
    loon::gpu::CommandPool*
        cmd_pool;  // Since allocation is thread-local, we need to be sure to free
                   // using the same one, even if it happens on a different thread.
    VkCommandBuffer       vk_cmd_buffers[2];
    loon::gpu::UsageScope first_usages;
    loon::gpu::UsageScope last_usages;
    bool                  touches_surface_image;
    WGPUCommandEncoder    cmd_encoder = nullptr;
    int64_t               submitted_timeline_value
        = -1;  // Value of the queue's timeline semaphore when this work is completed.
};

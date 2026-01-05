// Generated WebGPU C++ wrapper classes
// This file contains C++ wrapper classes for WebGPU objects

#pragma once

#include <webgpu/webgpu.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>


namespace webgpu_loon {

// Forward declarations
class Adapter;
class BindGroup;
class BindGroupLayout;
class Buffer;
class CommandBuffer;
class CommandEncoder;
class ComputePassEncoder;
class ComputePipeline;
class Device;
class Instance;
class PipelineLayout;
class QuerySet;
class Queue;
class RenderBundle;
class RenderBundleEncoder;
class RenderPassEncoder;
class RenderPipeline;
class Sampler;
class ShaderModule;
class Surface;
class Texture;
class TextureView;


class Adapter {
   public:
    Adapter(WGPUAdapter handle) : _handle(handle) {}
    Adapter(const Adapter&)            = delete;
    Adapter& operator=(const Adapter&) = delete;
    Adapter(Adapter&& other) noexcept : _handle(other._handle) { other._handle = nullptr; }
    Adapter& operator=(Adapter&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~Adapter() {
        if (_handle) { wgpuAdapterRelease(_handle); }
    }
    friend void swap(Adapter& a, Adapter& b) { std::swap(a._handle, b._handle); }

    operator WGPUAdapter() const { return _handle; }

    WGPUStatus get_limits(
        WGPULimits* limits);  // Indicates if there was an @ref OutStructChainError.

    bool has_feature(WGPUFeatureName feature);

    /*
    Get the list of @ref WGPUFeatureName values supported by the adapter.
    */
    void get_features(WGPUSupportedFeatures* features);

    WGPUStatus get_info(
        WGPUAdapterInfo* info);  // Indicates if there was an @ref OutStructChainError.

    void request_device(WGPUDeviceDescriptor const*   descriptor,
                        WGPURequestDeviceCallbackInfo callback);

   private:
    WGPUAdapter _handle = nullptr;
};  // class Adapter


class BindGroup {
   public:
    BindGroup(WGPUBindGroup handle) : _handle(handle) {}
    BindGroup(const BindGroup&)            = delete;
    BindGroup& operator=(const BindGroup&) = delete;
    BindGroup(BindGroup&& other) noexcept : _handle(other._handle) { other._handle = nullptr; }
    BindGroup& operator=(BindGroup&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~BindGroup() {
        if (_handle) { wgpuBindGroupRelease(_handle); }
    }
    friend void swap(BindGroup& a, BindGroup& b) { std::swap(a._handle, b._handle); }

    operator WGPUBindGroup() const { return _handle; }

    void set_label(WGPUStringView label);

   private:
    WGPUBindGroup _handle = nullptr;
};  // class BindGroup


class BindGroupLayout {
   public:
    BindGroupLayout(WGPUBindGroupLayout handle) : _handle(handle) {}
    BindGroupLayout(const BindGroupLayout&)            = delete;
    BindGroupLayout& operator=(const BindGroupLayout&) = delete;
    BindGroupLayout(BindGroupLayout&& other) noexcept : _handle(other._handle) {
        other._handle = nullptr;
    }
    BindGroupLayout& operator=(BindGroupLayout&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~BindGroupLayout() {
        if (_handle) { wgpuBindGroupLayoutRelease(_handle); }
    }
    friend void swap(BindGroupLayout& a, BindGroupLayout& b) { std::swap(a._handle, b._handle); }

    operator WGPUBindGroupLayout() const { return _handle; }

    void set_label(WGPUStringView label);

   private:
    WGPUBindGroupLayout _handle = nullptr;
};  // class BindGroupLayout


class Buffer {
   public:
    Buffer(WGPUBuffer handle) : _handle(handle) {}
    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : _handle(other._handle) { other._handle = nullptr; }
    Buffer& operator=(Buffer&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~Buffer() {
        if (_handle) { wgpuBufferRelease(_handle); }
    }
    friend void swap(Buffer& a, Buffer& b) { std::swap(a._handle, b._handle); }

    operator WGPUBuffer() const { return _handle; }

    void map_async(WGPUMapMode               mode,
                   size_t                    offset,
                   size_t                    size,
                   WGPUBufferMapCallbackInfo callback);

    /*
    Returns a mutable pointer to beginning of the mapped range.
    See @ref MappedRangeBehavior for error conditions and guarantees.
    This function is safe to call inside spontaneous callbacks (see @ref CallbackReentrancy).

    In Wasm, if `memcpy`ing into this range, prefer using @ref wgpuBufferWriteMappedRange
    instead for better performance.
    */
    void* get_mapped_range(size_t offset, size_t size);

    /*
    Returns a const pointer to beginning of the mapped range.
    It must not be written; writing to this range causes undefined behavior.
    See @ref MappedRangeBehavior for error conditions and guarantees.
    This function is safe to call inside spontaneous callbacks (see @ref CallbackReentrancy).

    In Wasm, if `memcpy`ing from this range, prefer using @ref wgpuBufferReadMappedRange
    instead for better performance.
    */
    void const* get_const_mapped_range(size_t offset, size_t size);

    /*
    Copies a range of data from the buffer mapping into the provided destination pointer.
    See @ref MappedRangeBehavior for error conditions and guarantees.
    This function is safe to call inside spontaneous callbacks (see @ref CallbackReentrancy).

    In Wasm, this is more efficient than copying from a mapped range into a `malloc`'d range.
    */
    WGPUStatus read_mapped_range(size_t offset,
                                 void*  data,
                                 size_t size);  // @ref WGPUStatus_Error if the copy did not occur.


    /*
    Copies a range of data from the provided source pointer into the buffer mapping.
    See @ref MappedRangeBehavior for error conditions and guarantees.
    This function is safe to call inside spontaneous callbacks (see @ref CallbackReentrancy).

    In Wasm, this is more efficient than copying from a `malloc`'d range into a mapped range.
    */
    WGPUStatus write_mapped_range(size_t      offset,
                                  void const* data,
                                  size_t size);  // @ref WGPUStatus_Error if the copy did not occur.


    void set_label(WGPUStringView label);

    WGPUBufferUsage get_usage();

    uint64_t get_size();

    WGPUBufferMapState get_map_state();

    void unmap();

    void destroy();

   private:
    WGPUBuffer _handle = nullptr;
};  // class Buffer


class CommandBuffer {
   public:
    CommandBuffer(WGPUCommandBuffer handle) : _handle(handle) {}
    CommandBuffer(const CommandBuffer&)            = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;
    CommandBuffer(CommandBuffer&& other) noexcept : _handle(other._handle) {
        other._handle = nullptr;
    }
    CommandBuffer& operator=(CommandBuffer&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~CommandBuffer() {
        if (_handle) { wgpuCommandBufferRelease(_handle); }
    }
    friend void swap(CommandBuffer& a, CommandBuffer& b) { std::swap(a._handle, b._handle); }

    operator WGPUCommandBuffer() const { return _handle; }

    void set_label(WGPUStringView label);

   private:
    WGPUCommandBuffer _handle = nullptr;
};  // class CommandBuffer


class CommandEncoder {
   public:
    CommandEncoder(WGPUCommandEncoder handle) : _handle(handle) {}
    CommandEncoder(const CommandEncoder&)            = delete;
    CommandEncoder& operator=(const CommandEncoder&) = delete;
    CommandEncoder(CommandEncoder&& other) noexcept : _handle(other._handle) {
        other._handle = nullptr;
    }
    CommandEncoder& operator=(CommandEncoder&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~CommandEncoder() {
        if (_handle) { wgpuCommandEncoderRelease(_handle); }
    }
    friend void swap(CommandEncoder& a, CommandEncoder& b) { std::swap(a._handle, b._handle); }

    operator WGPUCommandEncoder() const { return _handle; }

    CommandBuffer finish(WGPUCommandBufferDescriptor const* descriptor);

    ComputePassEncoder begin_compute_pass(WGPUComputePassDescriptor const* descriptor);

    RenderPassEncoder begin_render_pass(WGPURenderPassDescriptor const* descriptor);

    void copy_buffer_to_buffer(Buffer   source,
                               uint64_t source_offset,
                               Buffer   destination,
                               uint64_t destination_offset,
                               uint64_t size);

    void copy_buffer_to_texture(WGPUTexelCopyBufferInfo const*  source,
                                WGPUTexelCopyTextureInfo const* destination,
                                WGPUExtent3D const*             copy_size);

    void copy_texture_to_buffer(WGPUTexelCopyTextureInfo const* source,
                                WGPUTexelCopyBufferInfo const*  destination,
                                WGPUExtent3D const*             copy_size);

    void copy_texture_to_texture(WGPUTexelCopyTextureInfo const* source,
                                 WGPUTexelCopyTextureInfo const* destination,
                                 WGPUExtent3D const*             copy_size);

    void clear_buffer(Buffer buffer, uint64_t offset, uint64_t size);

    void insert_debug_marker(WGPUStringView marker_label);

    void pop_debug_group();

    void push_debug_group(WGPUStringView group_label);

    void resolve_query_set(QuerySet query_set,
                           uint32_t first_query,
                           uint32_t query_count,
                           Buffer   destination,
                           uint64_t destination_offset);

    void write_timestamp(QuerySet query_set, uint32_t query_index);

    void set_label(WGPUStringView label);

   private:
    WGPUCommandEncoder _handle = nullptr;
};  // class CommandEncoder


class ComputePassEncoder {
   public:
    ComputePassEncoder(WGPUComputePassEncoder handle) : _handle(handle) {}
    ComputePassEncoder(const ComputePassEncoder&)            = delete;
    ComputePassEncoder& operator=(const ComputePassEncoder&) = delete;
    ComputePassEncoder(ComputePassEncoder&& other) noexcept : _handle(other._handle) {
        other._handle = nullptr;
    }
    ComputePassEncoder& operator=(ComputePassEncoder&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~ComputePassEncoder() {
        if (_handle) { wgpuComputePassEncoderRelease(_handle); }
    }
    friend void swap(ComputePassEncoder& a, ComputePassEncoder& b) {
        std::swap(a._handle, b._handle);
    }

    operator WGPUComputePassEncoder() const { return _handle; }

    void insert_debug_marker(WGPUStringView marker_label);

    void pop_debug_group();

    void push_debug_group(WGPUStringView group_label);

    void set_pipeline(ComputePipeline pipeline);

    void set_bind_group(uint32_t                  group_index,
                        BindGroup                 group,
                        std::span<const uint32_t> dynamic_offsets);

    void dispatch_workgroups(uint32_t workgroupCountX,
                             uint32_t workgroupCountY,
                             uint32_t workgroupCountZ);

    void dispatch_workgroups_indirect(Buffer indirect_buffer, uint64_t indirect_offset);

    void end();

    void set_label(WGPUStringView label);

   private:
    WGPUComputePassEncoder _handle = nullptr;
};  // class ComputePassEncoder


class ComputePipeline {
   public:
    ComputePipeline(WGPUComputePipeline handle) : _handle(handle) {}
    ComputePipeline(const ComputePipeline&)            = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(ComputePipeline&& other) noexcept : _handle(other._handle) {
        other._handle = nullptr;
    }
    ComputePipeline& operator=(ComputePipeline&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~ComputePipeline() {
        if (_handle) { wgpuComputePipelineRelease(_handle); }
    }
    friend void swap(ComputePipeline& a, ComputePipeline& b) { std::swap(a._handle, b._handle); }

    operator WGPUComputePipeline() const { return _handle; }

    BindGroupLayout get_bind_group_layout(uint32_t group_index);

    void set_label(WGPUStringView label);

   private:
    WGPUComputePipeline _handle = nullptr;
};  // class ComputePipeline


class Device {
   public:
    Device(WGPUDevice handle) : _handle(handle) {}
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&& other) noexcept : _handle(other._handle) { other._handle = nullptr; }
    Device& operator=(Device&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~Device() {
        if (_handle) { wgpuDeviceRelease(_handle); }
    }
    friend void swap(Device& a, Device& b) { std::swap(a._handle, b._handle); }

    operator WGPUDevice() const { return _handle; }

    BindGroup create_bind_group(WGPUBindGroupDescriptor const* descriptor);

    BindGroupLayout create_bind_group_layout(WGPUBindGroupLayoutDescriptor const* descriptor);

    /*
    TODO

    If @ref WGPUBufferDescriptor::mappedAtCreation is `true` and the mapping allocation fails,
    returns `NULL`.
    */
    Buffer create_buffer(WGPUBufferDescriptor const* descriptor);

    CommandEncoder create_command_encoder(WGPUCommandEncoderDescriptor const* descriptor);

    ComputePipeline create_compute_pipeline(WGPUComputePipelineDescriptor const* descriptor);

    void create_compute_pipeline_async(WGPUComputePipelineDescriptor const*       descriptor,
                                       WGPUCreateComputePipelineAsyncCallbackInfo callback);

    PipelineLayout create_pipeline_layout(WGPUPipelineLayoutDescriptor const* descriptor);

    QuerySet create_query_set(WGPUQuerySetDescriptor const* descriptor);

    void create_render_pipeline_async(WGPURenderPipelineDescriptor const*       descriptor,
                                      WGPUCreateRenderPipelineAsyncCallbackInfo callback);

    RenderBundleEncoder create_render_bundle_encoder(
        WGPURenderBundleEncoderDescriptor const* descriptor);

    RenderPipeline create_render_pipeline(WGPURenderPipelineDescriptor const* descriptor);

    Sampler create_sampler(WGPUSamplerDescriptor const* descriptor);

    ShaderModule create_shader_module(WGPUShaderModuleDescriptor const* descriptor);

    Texture create_texture(WGPUTextureDescriptor const* descriptor);

    void destroy();

    /*
     */
    WGPUFuture get_lost_future();  // The @ref WGPUFuture for the device-lost event of the device.


    WGPUStatus get_limits(
        WGPULimits* limits);  // Indicates if there was an @ref OutStructChainError.

    bool has_feature(WGPUFeatureName feature);

    /*
    Get the list of @ref WGPUFeatureName values supported by the device.
    */
    void get_features(WGPUSupportedFeatures* features);

    WGPUStatus get_adapter_info(
        WGPUAdapterInfo* adapter_info);  // Indicates if there was an @ref OutStructChainError.

    Queue get_queue();

    /*
    Pushes an error scope to the current thread's error scope stack.
    See @ref ErrorScopes.
    */
    void push_error_scope(WGPUErrorFilter filter);

    /*
    Pops an error scope to the current thread's error scope stack,
    asynchronously returning the result. See @ref ErrorScopes.
    */
    void pop_error_scope(WGPUPopErrorScopeCallbackInfo callback);

    void set_label(WGPUStringView label);

   private:
    WGPUDevice _handle = nullptr;
};  // class Device


class Instance {
   public:
    Instance(WGPUInstance handle) : _handle(handle) {}
    Instance(const Instance&)            = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&& other) noexcept : _handle(other._handle) { other._handle = nullptr; }
    Instance& operator=(Instance&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~Instance() {
        if (_handle) { wgpuInstanceRelease(_handle); }
    }
    friend void swap(Instance& a, Instance& b) { std::swap(a._handle, b._handle); }

    operator WGPUInstance() const { return _handle; }

    /*
    Creates a @ref WGPUSurface, see @ref Surface-Creation for more details.*/
    Surface create_surface(
        WGPUSurfaceDescriptor const* descriptor);  // A new @ref WGPUSurface for this descriptor (or
                                                   // an error @ref WGPUSurface).

    /*
    Get the list of @ref WGPUWGSLLanguageFeatureName values supported by the instance.
    */
    void get_WGSL_language_features(WGPUSupportedWGSLLanguageFeatures* features);

    bool has_WGSL_language_feature(WGPUWGSLLanguageFeatureName feature);

    /*
    Processes asynchronous events on this `WGPUInstance`, calling any callbacks for asynchronous
    operations created with @ref WGPUCallbackMode_AllowProcessEvents.

    See @ref Process-Events for more information.
    */
    void process_events();

    void request_adapter(WGPURequestAdapterOptions const* options,
                         WGPURequestAdapterCallbackInfo   callback);

    /*
    Wait for at least one WGPUFuture in `futures` to complete, and call callbacks of the respective
    completed asynchronous operations.

    See @ref Wait-Any for more information.
    */
    WGPUWaitStatus wait_any(size_t future_count, WGPUFutureWaitInfo* futures, uint64_t timeout_NS);

   private:
    WGPUInstance _handle = nullptr;
};  // class Instance


class PipelineLayout {
   public:
    PipelineLayout(WGPUPipelineLayout handle) : _handle(handle) {}
    PipelineLayout(const PipelineLayout&)            = delete;
    PipelineLayout& operator=(const PipelineLayout&) = delete;
    PipelineLayout(PipelineLayout&& other) noexcept : _handle(other._handle) {
        other._handle = nullptr;
    }
    PipelineLayout& operator=(PipelineLayout&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~PipelineLayout() {
        if (_handle) { wgpuPipelineLayoutRelease(_handle); }
    }
    friend void swap(PipelineLayout& a, PipelineLayout& b) { std::swap(a._handle, b._handle); }

    operator WGPUPipelineLayout() const { return _handle; }

    void set_label(WGPUStringView label);

   private:
    WGPUPipelineLayout _handle = nullptr;
};  // class PipelineLayout


class QuerySet {
   public:
    QuerySet(WGPUQuerySet handle) : _handle(handle) {}
    QuerySet(const QuerySet&)            = delete;
    QuerySet& operator=(const QuerySet&) = delete;
    QuerySet(QuerySet&& other) noexcept : _handle(other._handle) { other._handle = nullptr; }
    QuerySet& operator=(QuerySet&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~QuerySet() {
        if (_handle) { wgpuQuerySetRelease(_handle); }
    }
    friend void swap(QuerySet& a, QuerySet& b) { std::swap(a._handle, b._handle); }

    operator WGPUQuerySet() const { return _handle; }

    void set_label(WGPUStringView label);

    WGPUQueryType get_type();

    uint32_t get_count();

    void destroy();

   private:
    WGPUQuerySet _handle = nullptr;
};  // class QuerySet


class Queue {
   public:
    Queue(WGPUQueue handle) : _handle(handle) {}
    Queue(const Queue&)            = delete;
    Queue& operator=(const Queue&) = delete;
    Queue(Queue&& other) noexcept : _handle(other._handle) { other._handle = nullptr; }
    Queue& operator=(Queue&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~Queue() {
        if (_handle) { wgpuQueueRelease(_handle); }
    }
    friend void swap(Queue& a, Queue& b) { std::swap(a._handle, b._handle); }

    operator WGPUQueue() const { return _handle; }

    void submit(std::span<const CommandBuffer> commands);

    void on_submitted_work_done(WGPUQueueWorkDoneCallbackInfo callback);

    /*
    Produces a @ref DeviceError both content-timeline (`size` alignment) and device-timeline
    errors defined by the WebGPU specification.
    */
    void write_buffer(Buffer buffer, uint64_t buffer_offset, void const* data, size_t size);

    void write_texture(WGPUTexelCopyTextureInfo const*  destination,
                       void const*                      data,
                       size_t                           data_size,
                       WGPUTexelCopyBufferLayout const* data_layout,
                       WGPUExtent3D const*              write_size);

    void set_label(WGPUStringView label);

   private:
    WGPUQueue _handle = nullptr;
};  // class Queue


class RenderBundle {
   public:
    RenderBundle(WGPURenderBundle handle) : _handle(handle) {}
    RenderBundle(const RenderBundle&)            = delete;
    RenderBundle& operator=(const RenderBundle&) = delete;
    RenderBundle(RenderBundle&& other) noexcept : _handle(other._handle) {
        other._handle = nullptr;
    }
    RenderBundle& operator=(RenderBundle&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~RenderBundle() {
        if (_handle) { wgpuRenderBundleRelease(_handle); }
    }
    friend void swap(RenderBundle& a, RenderBundle& b) { std::swap(a._handle, b._handle); }

    operator WGPURenderBundle() const { return _handle; }

    void set_label(WGPUStringView label);

   private:
    WGPURenderBundle _handle = nullptr;
};  // class RenderBundle


class RenderBundleEncoder {
   public:
    RenderBundleEncoder(WGPURenderBundleEncoder handle) : _handle(handle) {}
    RenderBundleEncoder(const RenderBundleEncoder&)            = delete;
    RenderBundleEncoder& operator=(const RenderBundleEncoder&) = delete;
    RenderBundleEncoder(RenderBundleEncoder&& other) noexcept : _handle(other._handle) {
        other._handle = nullptr;
    }
    RenderBundleEncoder& operator=(RenderBundleEncoder&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~RenderBundleEncoder() {
        if (_handle) { wgpuRenderBundleEncoderRelease(_handle); }
    }
    friend void swap(RenderBundleEncoder& a, RenderBundleEncoder& b) {
        std::swap(a._handle, b._handle);
    }

    operator WGPURenderBundleEncoder() const { return _handle; }

    void set_pipeline(RenderPipeline pipeline);

    void set_bind_group(uint32_t                  group_index,
                        BindGroup                 group,
                        std::span<const uint32_t> dynamic_offsets);

    void draw(uint32_t vertex_count,
              uint32_t instance_count,
              uint32_t first_vertex,
              uint32_t first_instance);

    void draw_indexed(uint32_t index_count,
                      uint32_t instance_count,
                      uint32_t first_index,
                      int32_t  base_vertex,
                      uint32_t first_instance);

    void draw_indirect(Buffer indirect_buffer, uint64_t indirect_offset);

    void draw_indexed_indirect(Buffer indirect_buffer, uint64_t indirect_offset);

    void insert_debug_marker(WGPUStringView marker_label);

    void pop_debug_group();

    void push_debug_group(WGPUStringView group_label);

    void set_vertex_buffer(uint32_t slot, Buffer buffer, uint64_t offset, uint64_t size);

    void set_index_buffer(Buffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size);

    RenderBundle finish(WGPURenderBundleDescriptor const* descriptor);

    void set_label(WGPUStringView label);

   private:
    WGPURenderBundleEncoder _handle = nullptr;
};  // class RenderBundleEncoder


class RenderPassEncoder {
   public:
    RenderPassEncoder(WGPURenderPassEncoder handle) : _handle(handle) {}
    RenderPassEncoder(const RenderPassEncoder&)            = delete;
    RenderPassEncoder& operator=(const RenderPassEncoder&) = delete;
    RenderPassEncoder(RenderPassEncoder&& other) noexcept : _handle(other._handle) {
        other._handle = nullptr;
    }
    RenderPassEncoder& operator=(RenderPassEncoder&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~RenderPassEncoder() {
        if (_handle) { wgpuRenderPassEncoderRelease(_handle); }
    }
    friend void swap(RenderPassEncoder& a, RenderPassEncoder& b) {
        std::swap(a._handle, b._handle);
    }

    operator WGPURenderPassEncoder() const { return _handle; }

    void set_pipeline(RenderPipeline pipeline);

    void set_bind_group(uint32_t                  group_index,
                        BindGroup                 group,
                        std::span<const uint32_t> dynamic_offsets);

    void draw(uint32_t vertex_count,
              uint32_t instance_count,
              uint32_t first_vertex,
              uint32_t first_instance);

    void draw_indexed(uint32_t index_count,
                      uint32_t instance_count,
                      uint32_t first_index,
                      int32_t  base_vertex,
                      uint32_t first_instance);

    void draw_indirect(Buffer indirect_buffer, uint64_t indirect_offset);

    void draw_indexed_indirect(Buffer indirect_buffer, uint64_t indirect_offset);

    void execute_bundles(std::span<const RenderBundle> bundles);

    void insert_debug_marker(WGPUStringView marker_label);

    void pop_debug_group();

    void push_debug_group(WGPUStringView group_label);

    void set_stencil_reference(uint32_t reference);

    void set_blend_constant(WGPUColor const* color);

    /*
    TODO

    If any argument is non-finite, produces a @ref NonFiniteFloatValueError.
    */
    void set_viewport(float x,
                      float y,
                      float width,
                      float height,
                      float min_depth,
                      float max_depth);

    void set_scissor_rect(uint32_t x, uint32_t y, uint32_t width, uint32_t height);

    void set_vertex_buffer(uint32_t slot, Buffer buffer, uint64_t offset, uint64_t size);

    void set_index_buffer(Buffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size);

    void begin_occlusion_query(uint32_t query_index);

    void end_occlusion_query();

    void end();

    void set_label(WGPUStringView label);

   private:
    WGPURenderPassEncoder _handle = nullptr;
};  // class RenderPassEncoder


class RenderPipeline {
   public:
    RenderPipeline(WGPURenderPipeline handle) : _handle(handle) {}
    RenderPipeline(const RenderPipeline&)            = delete;
    RenderPipeline& operator=(const RenderPipeline&) = delete;
    RenderPipeline(RenderPipeline&& other) noexcept : _handle(other._handle) {
        other._handle = nullptr;
    }
    RenderPipeline& operator=(RenderPipeline&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~RenderPipeline() {
        if (_handle) { wgpuRenderPipelineRelease(_handle); }
    }
    friend void swap(RenderPipeline& a, RenderPipeline& b) { std::swap(a._handle, b._handle); }

    operator WGPURenderPipeline() const { return _handle; }

    BindGroupLayout get_bind_group_layout(uint32_t group_index);

    void set_label(WGPUStringView label);

   private:
    WGPURenderPipeline _handle = nullptr;
};  // class RenderPipeline


class Sampler {
   public:
    Sampler(WGPUSampler handle) : _handle(handle) {}
    Sampler(const Sampler&)            = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&& other) noexcept : _handle(other._handle) { other._handle = nullptr; }
    Sampler& operator=(Sampler&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~Sampler() {
        if (_handle) { wgpuSamplerRelease(_handle); }
    }
    friend void swap(Sampler& a, Sampler& b) { std::swap(a._handle, b._handle); }

    operator WGPUSampler() const { return _handle; }

    void set_label(WGPUStringView label);

   private:
    WGPUSampler _handle = nullptr;
};  // class Sampler


class ShaderModule {
   public:
    ShaderModule(WGPUShaderModule handle) : _handle(handle) {}
    ShaderModule(const ShaderModule&)            = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule(ShaderModule&& other) noexcept : _handle(other._handle) {
        other._handle = nullptr;
    }
    ShaderModule& operator=(ShaderModule&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~ShaderModule() {
        if (_handle) { wgpuShaderModuleRelease(_handle); }
    }
    friend void swap(ShaderModule& a, ShaderModule& b) { std::swap(a._handle, b._handle); }

    operator WGPUShaderModule() const { return _handle; }

    void get_compilation_info(WGPUCompilationInfoCallbackInfo callback);

    void set_label(WGPUStringView label);

   private:
    WGPUShaderModule _handle = nullptr;
};  // class ShaderModule


class Surface {
   public:
    Surface(WGPUSurface handle) : _handle(handle) {}
    Surface(const Surface&)            = delete;
    Surface& operator=(const Surface&) = delete;
    Surface(Surface&& other) noexcept : _handle(other._handle) { other._handle = nullptr; }
    Surface& operator=(Surface&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~Surface() {
        if (_handle) { wgpuSurfaceRelease(_handle); }
    }
    friend void swap(Surface& a, Surface& b) { std::swap(a._handle, b._handle); }

    operator WGPUSurface() const { return _handle; }

    /*
    Configures parameters for rendering to `surface`.
    Produces a @ref DeviceError for all content-timeline errors defined by the WebGPU specification.

    See @ref Surface-Configuration for more details.
    */
    void configure(WGPUSurfaceConfiguration const* config);

    /*
    Provides information on how `adapter` is able to use `surface`.
    See @ref Surface-Capabilities for more details.
    */
    WGPUStatus get_capabilities(
        Adapter adapter,
        WGPUSurfaceCapabilities*
            capabilities);  // Indicates if there was an @ref OutStructChainError.

    /*
    Returns the @ref WGPUTexture to render to `surface` this frame along with metadata on the frame.
    Returns `NULL` and @ref WGPUSurfaceGetCurrentTextureStatus_Error if the surface is not
    configured.

    See @ref Surface-Presenting for more details.
    */
    void get_current_texture(WGPUSurfaceTexture* surface_texture);

    /*
    Shows `surface`'s current texture to the user.
    See @ref Surface-Presenting for more details.
    */
    WGPUStatus
    present();  // Returns @ref WGPUStatus_Error if the surface doesn't have a current texture.


    /*
    Removes the configuration for `surface`.
    See @ref Surface-Configuration for more details.
    */
    void unconfigure();

    /*
    Modifies the label used to refer to `surface`.*/
    void set_label(WGPUStringView label);

   private:
    WGPUSurface _handle = nullptr;
};  // class Surface


class Texture {
   public:
    Texture(WGPUTexture handle) : _handle(handle) {}
    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&& other) noexcept : _handle(other._handle) { other._handle = nullptr; }
    Texture& operator=(Texture&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~Texture() {
        if (_handle) { wgpuTextureRelease(_handle); }
    }
    friend void swap(Texture& a, Texture& b) { std::swap(a._handle, b._handle); }

    operator WGPUTexture() const { return _handle; }

    TextureView create_view(WGPUTextureViewDescriptor const* descriptor);

    void set_label(WGPUStringView label);

    uint32_t get_width();

    uint32_t get_height();

    uint32_t get_depth_or_array_layers();

    uint32_t get_mip_level_count();

    uint32_t get_sample_count();

    WGPUTextureDimension get_dimension();

    WGPUTextureFormat get_format();

    WGPUTextureUsage get_usage();

    void destroy();

   private:
    WGPUTexture _handle = nullptr;
};  // class Texture


class TextureView {
   public:
    TextureView(WGPUTextureView handle) : _handle(handle) {}
    TextureView(const TextureView&)            = delete;
    TextureView& operator=(const TextureView&) = delete;
    TextureView(TextureView&& other) noexcept : _handle(other._handle) { other._handle = nullptr; }
    TextureView& operator=(TextureView&& other) noexcept {
        swap(*this, other);
        return *this;
    }
    ~TextureView() {
        if (_handle) { wgpuTextureViewRelease(_handle); }
    }
    friend void swap(TextureView& a, TextureView& b) { std::swap(a._handle, b._handle); }

    operator WGPUTextureView() const { return _handle; }

    void set_label(WGPUStringView label);

   private:
    WGPUTextureView _handle = nullptr;
};  // class TextureView


}  // namespace webgpu_loon

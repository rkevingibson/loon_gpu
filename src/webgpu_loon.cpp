
#include "webgpu_loon.hpp"


namespace webgpu_loon {
WGPUStatus Adapter::get_limits(WGPULimits* limits) {
    return wgpuAdapterGetLimits(_handle, limits);
}
bool Adapter::has_feature(WGPUFeatureName feature) {
    return wgpuAdapterHasFeature(_handle, feature);
}
void Adapter::get_features(WGPUSupportedFeatures* features) {
    wgpuAdapterGetFeatures(_handle, features);
}
WGPUStatus Adapter::get_info(WGPUAdapterInfo* info) {
    return wgpuAdapterGetInfo(_handle, info);
}
void Adapter::request_device(WGPUDeviceDescriptor const*   descriptor,
                             WGPURequestDeviceCallbackInfo callback) {
    wgpuAdapterRequestDevice(_handle, descriptor, callback);
}

void BindGroup::set_label(WGPUStringView label) {
    wgpuBindGroupSetLabel(_handle, label);
}

void BindGroupLayout::set_label(WGPUStringView label) {
    wgpuBindGroupLayoutSetLabel(_handle, label);
}

void Buffer::map_async(WGPUMapMode               mode,
                       size_t                    offset,
                       size_t                    size,
                       WGPUBufferMapCallbackInfo callback) {
    wgpuBufferMapAsync(_handle, mode, offset, size, callback);
}
void* Buffer::get_mapped_range(size_t offset, size_t size) {
    return wgpuBufferGetMappedRange(_handle, offset, size);
}
void const* Buffer::get_const_mapped_range(size_t offset, size_t size) {
    return wgpuBufferGetConstMappedRange(_handle, offset, size);
}
WGPUStatus Buffer::read_mapped_range(size_t offset, void* data, size_t size) {
    return wgpuBufferReadMappedRange(_handle, offset, data, size);
}
WGPUStatus Buffer::write_mapped_range(size_t offset, void const* data, size_t size) {
    return wgpuBufferWriteMappedRange(_handle, offset, data, size);
}
void Buffer::set_label(WGPUStringView label) {
    wgpuBufferSetLabel(_handle, label);
}
WGPUBufferUsage Buffer::get_usage() {
    return wgpuBufferGetUsage(_handle);
}
uint64_t Buffer::get_size() {
    return wgpuBufferGetSize(_handle);
}
WGPUBufferMapState Buffer::get_map_state() {
    return wgpuBufferGetMapState(_handle);
}
void Buffer::unmap() {
    wgpuBufferUnmap(_handle);
}
void Buffer::destroy() {
    wgpuBufferDestroy(_handle);
}

void CommandBuffer::set_label(WGPUStringView label) {
    wgpuCommandBufferSetLabel(_handle, label);
}

CommandBuffer CommandEncoder::finish(WGPUCommandBufferDescriptor const* descriptor) {
    return wgpuCommandEncoderFinish(_handle, descriptor);
}
ComputePassEncoder CommandEncoder::begin_compute_pass(WGPUComputePassDescriptor const* descriptor) {
    return wgpuCommandEncoderBeginComputePass(_handle, descriptor);
}
RenderPassEncoder CommandEncoder::begin_render_pass(WGPURenderPassDescriptor const* descriptor) {
    return wgpuCommandEncoderBeginRenderPass(_handle, descriptor);
}
void CommandEncoder::copy_buffer_to_buffer(Buffer   source,
                                           uint64_t source_offset,
                                           Buffer   destination,
                                           uint64_t destination_offset,
                                           uint64_t size) {
    wgpuCommandEncoderCopyBufferToBuffer(_handle,
                                         WGPUBuffer(source),
                                         source_offset,
                                         WGPUBuffer(destination),
                                         destination_offset,
                                         size);
}
void CommandEncoder::copy_buffer_to_texture(WGPUTexelCopyBufferInfo const*  source,
                                            WGPUTexelCopyTextureInfo const* destination,
                                            WGPUExtent3D const*             copy_size) {
    wgpuCommandEncoderCopyBufferToTexture(_handle, source, destination, copy_size);
}
void CommandEncoder::copy_texture_to_buffer(WGPUTexelCopyTextureInfo const* source,
                                            WGPUTexelCopyBufferInfo const*  destination,
                                            WGPUExtent3D const*             copy_size) {
    wgpuCommandEncoderCopyTextureToBuffer(_handle, source, destination, copy_size);
}
void CommandEncoder::copy_texture_to_texture(WGPUTexelCopyTextureInfo const* source,
                                             WGPUTexelCopyTextureInfo const* destination,
                                             WGPUExtent3D const*             copy_size) {
    wgpuCommandEncoderCopyTextureToTexture(_handle, source, destination, copy_size);
}
void CommandEncoder::clear_buffer(Buffer buffer, uint64_t offset, uint64_t size) {
    wgpuCommandEncoderClearBuffer(_handle, WGPUBuffer(buffer), offset, size);
}
void CommandEncoder::insert_debug_marker(WGPUStringView marker_label) {
    wgpuCommandEncoderInsertDebugMarker(_handle, marker_label);
}
void CommandEncoder::pop_debug_group() {
    wgpuCommandEncoderPopDebugGroup(_handle);
}
void CommandEncoder::push_debug_group(WGPUStringView group_label) {
    wgpuCommandEncoderPushDebugGroup(_handle, group_label);
}
void CommandEncoder::resolve_query_set(QuerySet query_set,
                                       uint32_t first_query,
                                       uint32_t query_count,
                                       Buffer   destination,
                                       uint64_t destination_offset) {
    wgpuCommandEncoderResolveQuerySet(_handle,
                                      WGPUQuerySet(query_set),
                                      first_query,
                                      query_count,
                                      WGPUBuffer(destination),
                                      destination_offset);
}
void CommandEncoder::write_timestamp(QuerySet query_set, uint32_t query_index) {
    wgpuCommandEncoderWriteTimestamp(_handle, WGPUQuerySet(query_set), query_index);
}
void CommandEncoder::set_label(WGPUStringView label) {
    wgpuCommandEncoderSetLabel(_handle, label);
}

void ComputePassEncoder::insert_debug_marker(WGPUStringView marker_label) {
    wgpuComputePassEncoderInsertDebugMarker(_handle, marker_label);
}
void ComputePassEncoder::pop_debug_group() {
    wgpuComputePassEncoderPopDebugGroup(_handle);
}
void ComputePassEncoder::push_debug_group(WGPUStringView group_label) {
    wgpuComputePassEncoderPushDebugGroup(_handle, group_label);
}
void ComputePassEncoder::set_pipeline(ComputePipeline pipeline) {
    wgpuComputePassEncoderSetPipeline(_handle, WGPUComputePipeline(pipeline));
}
void ComputePassEncoder::set_bind_group(uint32_t                  group_index,
                                        BindGroup                 group,
                                        std::span<const uint32_t> dynamic_offsets) {
    wgpuComputePassEncoderSetBindGroup(_handle,
                                       group_index,
                                       WGPUBindGroup(group),
                                       dynamic_offsets.size(),
                                       dynamic_offsets.data());
}
void ComputePassEncoder::dispatch_workgroups(uint32_t workgroupCountX,
                                             uint32_t workgroupCountY,
                                             uint32_t workgroupCountZ) {
    wgpuComputePassEncoderDispatchWorkgroups(_handle,
                                             workgroupCountX,
                                             workgroupCountY,
                                             workgroupCountZ);
}
void ComputePassEncoder::dispatch_workgroups_indirect(Buffer   indirect_buffer,
                                                      uint64_t indirect_offset) {
    wgpuComputePassEncoderDispatchWorkgroupsIndirect(_handle,
                                                     WGPUBuffer(indirect_buffer),
                                                     indirect_offset);
}
void ComputePassEncoder::end() {
    wgpuComputePassEncoderEnd(_handle);
}
void ComputePassEncoder::set_label(WGPUStringView label) {
    wgpuComputePassEncoderSetLabel(_handle, label);
}

BindGroupLayout ComputePipeline::get_bind_group_layout(uint32_t group_index) {
    return wgpuComputePipelineGetBindGroupLayout(_handle, group_index);
}
void ComputePipeline::set_label(WGPUStringView label) {
    wgpuComputePipelineSetLabel(_handle, label);
}

BindGroup Device::create_bind_group(WGPUBindGroupDescriptor const* descriptor) {
    return wgpuDeviceCreateBindGroup(_handle, descriptor);
}
BindGroupLayout Device::create_bind_group_layout(WGPUBindGroupLayoutDescriptor const* descriptor) {
    return wgpuDeviceCreateBindGroupLayout(_handle, descriptor);
}
Buffer Device::create_buffer(WGPUBufferDescriptor const* descriptor) {
    return wgpuDeviceCreateBuffer(_handle, descriptor);
}
CommandEncoder Device::create_command_encoder(WGPUCommandEncoderDescriptor const* descriptor) {
    return wgpuDeviceCreateCommandEncoder(_handle, descriptor);
}
ComputePipeline Device::create_compute_pipeline(WGPUComputePipelineDescriptor const* descriptor) {
    return wgpuDeviceCreateComputePipeline(_handle, descriptor);
}
void Device::create_compute_pipeline_async(WGPUComputePipelineDescriptor const*       descriptor,
                                           WGPUCreateComputePipelineAsyncCallbackInfo callback) {
    wgpuDeviceCreateComputePipelineAsync(_handle, descriptor, callback);
}
PipelineLayout Device::create_pipeline_layout(WGPUPipelineLayoutDescriptor const* descriptor) {
    return wgpuDeviceCreatePipelineLayout(_handle, descriptor);
}
QuerySet Device::create_query_set(WGPUQuerySetDescriptor const* descriptor) {
    return wgpuDeviceCreateQuerySet(_handle, descriptor);
}
void Device::create_render_pipeline_async(WGPURenderPipelineDescriptor const*       descriptor,
                                          WGPUCreateRenderPipelineAsyncCallbackInfo callback) {
    wgpuDeviceCreateRenderPipelineAsync(_handle, descriptor, callback);
}
RenderBundleEncoder Device::create_render_bundle_encoder(
    WGPURenderBundleEncoderDescriptor const* descriptor) {
    return wgpuDeviceCreateRenderBundleEncoder(_handle, descriptor);
}
RenderPipeline Device::create_render_pipeline(WGPURenderPipelineDescriptor const* descriptor) {
    return wgpuDeviceCreateRenderPipeline(_handle, descriptor);
}
Sampler Device::create_sampler(WGPUSamplerDescriptor const* descriptor) {
    return wgpuDeviceCreateSampler(_handle, descriptor);
}
ShaderModule Device::create_shader_module(WGPUShaderModuleDescriptor const* descriptor) {
    return wgpuDeviceCreateShaderModule(_handle, descriptor);
}
Texture Device::create_texture(WGPUTextureDescriptor const* descriptor) {
    return wgpuDeviceCreateTexture(_handle, descriptor);
}
void Device::destroy() {
    wgpuDeviceDestroy(_handle);
}
WGPUFuture Device::get_lost_future() {
    return wgpuDeviceGetLostFuture(_handle);
}
WGPUStatus Device::get_limits(WGPULimits* limits) {
    return wgpuDeviceGetLimits(_handle, limits);
}
bool Device::has_feature(WGPUFeatureName feature) {
    return wgpuDeviceHasFeature(_handle, feature);
}
void Device::get_features(WGPUSupportedFeatures* features) {
    wgpuDeviceGetFeatures(_handle, features);
}
WGPUStatus Device::get_adapter_info(WGPUAdapterInfo* adapter_info) {
    return wgpuDeviceGetAdapterInfo(_handle, adapter_info);
}
Queue Device::get_queue() {
    return wgpuDeviceGetQueue(_handle);
}
void Device::push_error_scope(WGPUErrorFilter filter) {
    wgpuDevicePushErrorScope(_handle, filter);
}
void Device::pop_error_scope(WGPUPopErrorScopeCallbackInfo callback) {
    wgpuDevicePopErrorScope(_handle, callback);
}
void Device::set_label(WGPUStringView label) {
    wgpuDeviceSetLabel(_handle, label);
}

Surface Instance::create_surface(WGPUSurfaceDescriptor const* descriptor) {
    return wgpuInstanceCreateSurface(_handle, descriptor);
}
void Instance::get_WGSL_language_features(WGPUSupportedWGSLLanguageFeatures* features) {
    wgpuInstanceGetWGSLLanguageFeatures(_handle, features);
}
bool Instance::has_WGSL_language_feature(WGPUWGSLLanguageFeatureName feature) {
    return wgpuInstanceHasWGSLLanguageFeature(_handle, feature);
}
void Instance::process_events() {
    wgpuInstanceProcessEvents(_handle);
}
void Instance::request_adapter(WGPURequestAdapterOptions const* options,
                               WGPURequestAdapterCallbackInfo   callback) {
    wgpuInstanceRequestAdapter(_handle, options, callback);
}
WGPUWaitStatus Instance::wait_any(size_t              future_count,
                                  WGPUFutureWaitInfo* futures,
                                  uint64_t            timeout_NS) {
    return wgpuInstanceWaitAny(_handle, future_count, futures, timeout_NS);
}

void PipelineLayout::set_label(WGPUStringView label) {
    wgpuPipelineLayoutSetLabel(_handle, label);
}

void QuerySet::set_label(WGPUStringView label) {
    wgpuQuerySetSetLabel(_handle, label);
}
WGPUQueryType QuerySet::get_type() {
    return wgpuQuerySetGetType(_handle);
}
uint32_t QuerySet::get_count() {
    return wgpuQuerySetGetCount(_handle);
}
void QuerySet::destroy() {
    wgpuQuerySetDestroy(_handle);
}

void Queue::submit(std::span<const CommandBuffer> commands) {
    wgpuQueueSubmit(_handle,
                    commands.size(),
                    reinterpret_cast<const WGPUCommandBuffer*>(commands.data()));
}
void Queue::on_submitted_work_done(WGPUQueueWorkDoneCallbackInfo callback) {
    wgpuQueueOnSubmittedWorkDone(_handle, callback);
}
void Queue::write_buffer(Buffer buffer, uint64_t buffer_offset, void const* data, size_t size) {
    wgpuQueueWriteBuffer(_handle, WGPUBuffer(buffer), buffer_offset, data, size);
}
void Queue::write_texture(WGPUTexelCopyTextureInfo const*  destination,
                          void const*                      data,
                          size_t                           data_size,
                          WGPUTexelCopyBufferLayout const* data_layout,
                          WGPUExtent3D const*              write_size) {
    wgpuQueueWriteTexture(_handle, destination, data, data_size, data_layout, write_size);
}
void Queue::set_label(WGPUStringView label) {
    wgpuQueueSetLabel(_handle, label);
}

void RenderBundle::set_label(WGPUStringView label) {
    wgpuRenderBundleSetLabel(_handle, label);
}

void RenderBundleEncoder::set_pipeline(RenderPipeline pipeline) {
    wgpuRenderBundleEncoderSetPipeline(_handle, WGPURenderPipeline(pipeline));
}
void RenderBundleEncoder::set_bind_group(uint32_t                  group_index,
                                         BindGroup                 group,
                                         std::span<const uint32_t> dynamic_offsets) {
    wgpuRenderBundleEncoderSetBindGroup(_handle,
                                        group_index,
                                        WGPUBindGroup(group),
                                        dynamic_offsets.size(),
                                        dynamic_offsets.data());
}
void RenderBundleEncoder::draw(uint32_t vertex_count,
                               uint32_t instance_count,
                               uint32_t first_vertex,
                               uint32_t first_instance) {
    wgpuRenderBundleEncoderDraw(_handle,
                                vertex_count,
                                instance_count,
                                first_vertex,
                                first_instance);
}
void RenderBundleEncoder::draw_indexed(uint32_t index_count,
                                       uint32_t instance_count,
                                       uint32_t first_index,
                                       int32_t  base_vertex,
                                       uint32_t first_instance) {
    wgpuRenderBundleEncoderDrawIndexed(_handle,
                                       index_count,
                                       instance_count,
                                       first_index,
                                       base_vertex,
                                       first_instance);
}
void RenderBundleEncoder::draw_indirect(Buffer indirect_buffer, uint64_t indirect_offset) {
    wgpuRenderBundleEncoderDrawIndirect(_handle, WGPUBuffer(indirect_buffer), indirect_offset);
}
void RenderBundleEncoder::draw_indexed_indirect(Buffer indirect_buffer, uint64_t indirect_offset) {
    wgpuRenderBundleEncoderDrawIndexedIndirect(_handle,
                                               WGPUBuffer(indirect_buffer),
                                               indirect_offset);
}
void RenderBundleEncoder::insert_debug_marker(WGPUStringView marker_label) {
    wgpuRenderBundleEncoderInsertDebugMarker(_handle, marker_label);
}
void RenderBundleEncoder::pop_debug_group() {
    wgpuRenderBundleEncoderPopDebugGroup(_handle);
}
void RenderBundleEncoder::push_debug_group(WGPUStringView group_label) {
    wgpuRenderBundleEncoderPushDebugGroup(_handle, group_label);
}
void RenderBundleEncoder::set_vertex_buffer(uint32_t slot,
                                            Buffer   buffer,
                                            uint64_t offset,
                                            uint64_t size) {
    wgpuRenderBundleEncoderSetVertexBuffer(_handle, slot, WGPUBuffer(buffer), offset, size);
}
void RenderBundleEncoder::set_index_buffer(Buffer          buffer,
                                           WGPUIndexFormat format,
                                           uint64_t        offset,
                                           uint64_t        size) {
    wgpuRenderBundleEncoderSetIndexBuffer(_handle, WGPUBuffer(buffer), format, offset, size);
}
RenderBundle RenderBundleEncoder::finish(WGPURenderBundleDescriptor const* descriptor) {
    return wgpuRenderBundleEncoderFinish(_handle, descriptor);
}
void RenderBundleEncoder::set_label(WGPUStringView label) {
    wgpuRenderBundleEncoderSetLabel(_handle, label);
}

void RenderPassEncoder::set_pipeline(RenderPipeline pipeline) {
    wgpuRenderPassEncoderSetPipeline(_handle, WGPURenderPipeline(pipeline));
}
void RenderPassEncoder::set_bind_group(uint32_t                  group_index,
                                       BindGroup                 group,
                                       std::span<const uint32_t> dynamic_offsets) {
    wgpuRenderPassEncoderSetBindGroup(_handle,
                                      group_index,
                                      WGPUBindGroup(group),
                                      dynamic_offsets.size(),
                                      dynamic_offsets.data());
}
void RenderPassEncoder::draw(uint32_t vertex_count,
                             uint32_t instance_count,
                             uint32_t first_vertex,
                             uint32_t first_instance) {
    wgpuRenderPassEncoderDraw(_handle, vertex_count, instance_count, first_vertex, first_instance);
}
void RenderPassEncoder::draw_indexed(uint32_t index_count,
                                     uint32_t instance_count,
                                     uint32_t first_index,
                                     int32_t  base_vertex,
                                     uint32_t first_instance) {
    wgpuRenderPassEncoderDrawIndexed(_handle,
                                     index_count,
                                     instance_count,
                                     first_index,
                                     base_vertex,
                                     first_instance);
}
void RenderPassEncoder::draw_indirect(Buffer indirect_buffer, uint64_t indirect_offset) {
    wgpuRenderPassEncoderDrawIndirect(_handle, WGPUBuffer(indirect_buffer), indirect_offset);
}
void RenderPassEncoder::draw_indexed_indirect(Buffer indirect_buffer, uint64_t indirect_offset) {
    wgpuRenderPassEncoderDrawIndexedIndirect(_handle, WGPUBuffer(indirect_buffer), indirect_offset);
}
void RenderPassEncoder::execute_bundles(std::span<const RenderBundle> bundles) {
    wgpuRenderPassEncoderExecuteBundles(_handle,
                                        bundles.size(),
                                        reinterpret_cast<const WGPURenderBundle*>(bundles.data()));
}
void RenderPassEncoder::insert_debug_marker(WGPUStringView marker_label) {
    wgpuRenderPassEncoderInsertDebugMarker(_handle, marker_label);
}
void RenderPassEncoder::pop_debug_group() {
    wgpuRenderPassEncoderPopDebugGroup(_handle);
}
void RenderPassEncoder::push_debug_group(WGPUStringView group_label) {
    wgpuRenderPassEncoderPushDebugGroup(_handle, group_label);
}
void RenderPassEncoder::set_stencil_reference(uint32_t reference) {
    wgpuRenderPassEncoderSetStencilReference(_handle, reference);
}
void RenderPassEncoder::set_blend_constant(WGPUColor const* color) {
    wgpuRenderPassEncoderSetBlendConstant(_handle, color);
}
void RenderPassEncoder::set_viewport(float x,
                                     float y,
                                     float width,
                                     float height,
                                     float min_depth,
                                     float max_depth) {
    wgpuRenderPassEncoderSetViewport(_handle, x, y, width, height, min_depth, max_depth);
}
void RenderPassEncoder::set_scissor_rect(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    wgpuRenderPassEncoderSetScissorRect(_handle, x, y, width, height);
}
void RenderPassEncoder::set_vertex_buffer(uint32_t slot,
                                          Buffer   buffer,
                                          uint64_t offset,
                                          uint64_t size) {
    wgpuRenderPassEncoderSetVertexBuffer(_handle, slot, WGPUBuffer(buffer), offset, size);
}
void RenderPassEncoder::set_index_buffer(Buffer          buffer,
                                         WGPUIndexFormat format,
                                         uint64_t        offset,
                                         uint64_t        size) {
    wgpuRenderPassEncoderSetIndexBuffer(_handle, WGPUBuffer(buffer), format, offset, size);
}
void RenderPassEncoder::begin_occlusion_query(uint32_t query_index) {
    wgpuRenderPassEncoderBeginOcclusionQuery(_handle, query_index);
}
void RenderPassEncoder::end_occlusion_query() {
    wgpuRenderPassEncoderEndOcclusionQuery(_handle);
}
void RenderPassEncoder::end() {
    wgpuRenderPassEncoderEnd(_handle);
}
void RenderPassEncoder::set_label(WGPUStringView label) {
    wgpuRenderPassEncoderSetLabel(_handle, label);
}

BindGroupLayout RenderPipeline::get_bind_group_layout(uint32_t group_index) {
    return wgpuRenderPipelineGetBindGroupLayout(_handle, group_index);
}
void RenderPipeline::set_label(WGPUStringView label) {
    wgpuRenderPipelineSetLabel(_handle, label);
}

void Sampler::set_label(WGPUStringView label) {
    wgpuSamplerSetLabel(_handle, label);
}

void ShaderModule::get_compilation_info(WGPUCompilationInfoCallbackInfo callback) {
    wgpuShaderModuleGetCompilationInfo(_handle, callback);
}
void ShaderModule::set_label(WGPUStringView label) {
    wgpuShaderModuleSetLabel(_handle, label);
}

void Surface::configure(WGPUSurfaceConfiguration const* config) {
    wgpuSurfaceConfigure(_handle, config);
}
WGPUStatus Surface::get_capabilities(Adapter adapter, WGPUSurfaceCapabilities* capabilities) {
    return wgpuSurfaceGetCapabilities(_handle, WGPUAdapter(adapter), capabilities);
}
void Surface::get_current_texture(WGPUSurfaceTexture* surface_texture) {
    wgpuSurfaceGetCurrentTexture(_handle, surface_texture);
}
WGPUStatus Surface::present() {
    return wgpuSurfacePresent(_handle);
}
void Surface::unconfigure() {
    wgpuSurfaceUnconfigure(_handle);
}
void Surface::set_label(WGPUStringView label) {
    wgpuSurfaceSetLabel(_handle, label);
}

TextureView Texture::create_view(WGPUTextureViewDescriptor const* descriptor) {
    return wgpuTextureCreateView(_handle, descriptor);
}
void Texture::set_label(WGPUStringView label) {
    wgpuTextureSetLabel(_handle, label);
}
uint32_t Texture::get_width() {
    return wgpuTextureGetWidth(_handle);
}
uint32_t Texture::get_height() {
    return wgpuTextureGetHeight(_handle);
}
uint32_t Texture::get_depth_or_array_layers() {
    return wgpuTextureGetDepthOrArrayLayers(_handle);
}
uint32_t Texture::get_mip_level_count() {
    return wgpuTextureGetMipLevelCount(_handle);
}
uint32_t Texture::get_sample_count() {
    return wgpuTextureGetSampleCount(_handle);
}
WGPUTextureDimension Texture::get_dimension() {
    return wgpuTextureGetDimension(_handle);
}
WGPUTextureFormat Texture::get_format() {
    return wgpuTextureGetFormat(_handle);
}
WGPUTextureUsage Texture::get_usage() {
    return wgpuTextureGetUsage(_handle);
}
void Texture::destroy() {
    wgpuTextureDestroy(_handle);
}

void TextureView::set_label(WGPUStringView label) {
    wgpuTextureViewSetLabel(_handle, label);
}


}  // namespace webgpu_loon

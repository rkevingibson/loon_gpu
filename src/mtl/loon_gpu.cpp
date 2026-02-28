#include <gpu/loon_gpu.h>

#include <cassert>
#include <cstddef>
#include <cstring>

// NB: This include has to be before others, as it defines the implementation of the metal lib.
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal.hpp>

#include "containers.h"
#include "gpu_to_mtl.h"

namespace loon::gpu {
template class Span<const char>;
template class Span<uint8_t>;
template class Span<const gpu::SamplerDesc>;
template class Span<const gpu::ColorTarget>;
template class Span<const gpu::RenderAttachment>;
template class Span<const gpu::Format>;
template class Span<const gpu::PresentMode>;
template class Span<const gpu::CommandBuffer>;
template class Span<const gpu::SemaphoreInfo>;
template class Span<const Handle<gpu::CommandBuffer>>;
template class Span<const gpu::TextureTransition>;
template class Function<void>;

struct Buffer {
    MTL::Buffer* buffer;
    MTL::Heap*   heap;
    void*        host_ptr;
    GpuPtr       device_ptr;
};

struct Texture {
    MTL::Texture* texture;
};

struct TextureHeap {
    MTL::TextureViewPool* pool = nullptr;
    TwoLevelBitset        bitset;
};

struct Pipeline {
    MTL::RenderPipelineState*  render_pipeline  = nullptr;
    MTL::ComputePipelineState* compute_pipeline = nullptr;
    Cull                       cull_mode;
};

struct DepthStencilState {
    MTL::DepthStencilState* state = nullptr;
};

struct Semaphore {
    MTL::SharedEvent* event = nullptr;
};

struct QueueImpl {
    struct Event {
        uint64_t       completed_time;
        Function<void> callback;
    };

    MTL4::CommandQueue* command_queue  = nullptr;
    MTL::SharedEvent*   callback_event = nullptr;
    Vector<Event>       pending_events;
    uint64_t            timeline_value = 0;

    Device device = nullptr;
};

struct Surface {
    CA::MetalLayer*    metal_layer;
    CA::MetalDrawable* current_drawable = nullptr;

    // Formats from
    // https://developer.apple.com/documentation/quartzcore/cametallayer/pixelformat?language=objc
    // We don't support all of the listed options but we've got the basics
    static constexpr Format kSwapchainFormats[] = {
        Format::BGRA8Unorm,
        Format::BGRA8UnormSrgb,
        Format::RGBA16Float,
        Format::RGB10A2Unorm,
    };
    static constexpr PresentMode kPresentModes[] = {
        PresentMode::Fifo,
    };
};
struct ThreadLocalState {
    constexpr static size_t kArenaSize = 256ll * 1024;
    loon::gpu::Allocator    allocator;
    MemoryBlock             arena_memory;
    loon::gpu::Arena        arena;
    ThreadLocalState(const loon::gpu::Allocator& alloc) :
        allocator{alloc},
        arena_memory{allocator.alloc(kArenaSize)},
        arena(arena_memory.ptr, arena_memory.len) {}
    ~ThreadLocalState() { allocator.free(arena_memory); }
};

struct BufferAndOffset {
    MTL::Buffer* buffer;
    MTL::Heap*   heap;
    uint32_t     offset;
};

struct CommandBufferImpl {
    MTL4::CommandBuffer* command_buffer;
    Device               device;
};

struct GpuPtrMap {
    GpuPtr         ptr;
    Handle<Buffer> buffer;
};
static constexpr auto kPtrMapCompare
    = [](const GpuPtrMap& a, const GpuPtrMap& b) -> bool { return a.ptr > b.ptr; };

static constexpr auto lower_bound = [](GpuPtrMap* first, GpuPtrMap* last, const GpuPtrMap& value) {
    GpuPtrMap* it;
    size_t     count = last - first;
    while (count > 0) {
        const size_t step = count / 2;
        it                = first + step;
        if (kPtrMapCompare(*it, value)) {
            first = ++it;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return first;
};

struct DeviceImpl {
    Allocator          m_allocator;
    ProcLogCallback    m_log_callback = nullptr;
    void*              m_log_userdata = nullptr;
    LogLevel           m_log_level    = LogLevel::Off;
    loon::gpu::tls_key m_tls_key;

    MTL::Device* m_device = nullptr;

    MTL4::Compiler*            m_compiler;
    MTL4::CompilerTaskOptions* m_options;

    MTL::ResidencySet* m_residency_set = nullptr;
    Surface            m_surface;

    SlotMap<Buffer>            m_buffer_pool;
    SlotMap<Texture>           m_texture_pool;
    SlotMap<TextureHeap>       m_texture_heap_pool;
    SlotMap<Pipeline>          m_pipeline_pool;
    SlotMap<DepthStencilState> m_depth_stencil_state_pool;
    SlotMap<Semaphore>         m_semaphore_pool;
    Vector<GpuPtrMap>          m_ptr_map;
};

void log(Device d, LogLevel lvl, Span<const char> msg) {
    d->m_log_callback(lvl, msg, d->m_log_userdata);
};

Arena* get_thread_local_arena(Device d) {
    auto state = reinterpret_cast<ThreadLocalState*>(loon::gpu::tls_get_data(d->m_tls_key));
    if (state == nullptr) {
        auto tls_block = d->m_allocator.alloc(sizeof(ThreadLocalState));
        if (tls_block.ptr == nullptr) {
            log(d, LogLevel::Error, "Allocator out of memory"_sv);
            return nullptr;
        }
        state = ::new (tls_block.ptr) ThreadLocalState(d->m_allocator);
        loon::gpu::tls_set_data(d->m_tls_key, state);
    }
    return &state->arena;
}

Device create_device(const DeviceDesc& desc) {
    Allocator alloc
        = desc.alloc_callback ? Allocator(desc.alloc_callback, desc.alloc_userdata) : Allocator();

    auto blk = alloc.alloc(sizeof(DeviceImpl));
    if (blk.ptr == 0) { return nullptr; }

    auto device = MTL::CreateSystemDefaultDevice();
    if (!device) { return nullptr; }

    NS::Error* error = nullptr;

    auto compiler_desc = MTL4::CompilerDescriptor::alloc()->init();
    auto compiler      = device->newCompiler(compiler_desc, &error);
    compiler_desc->release();
    auto options = MTL4::CompilerTaskOptions::alloc()->init();

    // Is this a valid cast? We're being passed a CAMetalLayer, not the C++ wrapper. Not sure!
    // At the very least I want to retain it.
    auto surface_metal_layer = reinterpret_cast<CA::MetalLayer*>(desc.native_window_handle);
    surface_metal_layer->retain();

    auto residency_set_descriptor = MTL::ResidencySetDescriptor::alloc()->init();
    // NOTE: Not sure how much this matters, should dig in more
    residency_set_descriptor->setInitialCapacity(64);
    auto residency_set = device->newResidencySet(residency_set_descriptor, nullptr);
    residency_set_descriptor->release();

    // NOTE: Not 100% sure about if this is UB - I need a pointer to the device in order to capture
    // it in the lambdas, that are created during construction of the object. It probably is
    // technically undefined behviour. I'm also pretty certain it's totally fine in practice, if a
    // bit wonky.
    auto d = reinterpret_cast<DeviceImpl*>(blk.ptr);
    return new (blk.ptr) DeviceImpl{
        .m_allocator                = alloc,
        .m_log_callback             = desc.log_callback,
        .m_log_userdata             = desc.log_userdata,
        .m_log_level                = desc.log_level,
        .m_tls_key                  = loon::gpu::tls_alloc([](void* data) {
            auto state = reinterpret_cast<ThreadLocalState*>(data);
            state->~ThreadLocalState();
        }),
        .m_device                   = device,
        .m_compiler                 = compiler,
        .m_options                  = options,
        .m_residency_set            = residency_set,
        .m_surface                  = Surface{
            .metal_layer = surface_metal_layer,
            .current_drawable = nullptr,
        },
        .m_buffer_pool              = SlotMap<Buffer>(alloc,
                                         [d](Buffer* b) {
                                             if (b->buffer) {
                                                 d->m_residency_set->removeAllocation(b->buffer);
                                                 b->buffer->release();
                                             }
                                             b->~Buffer();
                                         }),
        .m_texture_pool             = SlotMap<Texture>(alloc,
                                           [d](Texture* t) {
                                               if (t->texture) {
                                                   d->m_residency_set->removeAllocation(t->texture);
                                                   t->texture->release();
                                               }
                                               t->~Texture();
                                           }),
        .m_texture_heap_pool        = SlotMap<TextureHeap>(alloc,
                                                    [](TextureHeap* t) {
                                                        if (t->pool) { t->pool->release(); }
                                                        t->~TextureHeap();
                                                    }),
        .m_pipeline_pool            = SlotMap<Pipeline>(alloc,
                                             [](Pipeline* p) {
                                                 if (p->compute_pipeline) {
                                                     p->compute_pipeline->release();
                                                 }
                                                 if (p->render_pipeline) {
                                                     p->render_pipeline->release();
                                                 }
                                             }),
        .m_depth_stencil_state_pool = SlotMap<DepthStencilState>(alloc,
                                                                 [](DepthStencilState* s) {
                                                                     if (s->state) {
                                                                         s->state->release();
                                                                     }
                                                                 }),
        .m_semaphore_pool           = SlotMap<Semaphore>(alloc,
                                               [](Semaphore* s) {
                                                   if (s->event) { s->event->release(); }
                                               }),
        .m_ptr_map                  = Vector<GpuPtrMap>(alloc),
    };
}

void device_wait_for_idle(Device d) {
    // TODO: Not sure how to implement this one.
}

// MARK: Surface functions
SurfaceCapabilities get_surface_capabilities(Device d) {
    return SurfaceCapabilities{
        .formats       = Surface::kSwapchainFormats,
        .present_modes = Surface::kPresentModes,
        .usages = UsageFlags::ColorAttachment | UsageFlags::TransferDst | UsageFlags::Storage,
    };
}

bool configure_surface(Device d, const SurfaceConfiguration& config) {
    d->m_surface.metal_layer->setPixelFormat(bridge(config.format));
    d->m_surface.metal_layer->setDrawableSize(CGSize{
        .width  = static_cast<float>(config.width),
        .height = static_cast<float>(config.height),
    });
    return true;
}

void unconfigure_surface(Device d) {
    // Not much to do here.
    if (d->m_surface.current_drawable) {
        d->m_surface.current_drawable->release();
        d->m_surface.current_drawable = nullptr;
    }
}

SurfaceTextureInfo get_current_texture(Device d) {
    CA::MetalDrawable* drawable = d->m_surface.metal_layer->nextDrawable();

    if (!drawable) {
        return SurfaceTextureInfo{
            .texture = 0,
            .status  = SurfaceStatus::Error,
        };
    }

    d->m_surface.current_drawable = drawable;
    MTL::Texture*   tex           = drawable->texture();
    Handle<Texture> handle        = d->m_texture_pool.emplace({
               .texture = tex,
    });

    return SurfaceTextureInfo{
        .texture = handle,
        .status  = SurfaceStatus::Success,
    };
}

SurfaceStatus present(Device d, Queue queue) {
    queue->command_queue->signalDrawable(d->m_surface.current_drawable);

    d->m_surface.current_drawable->present();
    d->m_surface.current_drawable->release();
    d->m_surface.current_drawable = nullptr;
    return SurfaceStatus::Success;
}

// MARK: Buffers:

Handle<Buffer> malloc(Device d, size_t bytes, Memory memory) {
    return malloc(d, bytes, 64, memory);
}

Handle<Buffer> malloc(Device d, size_t bytes, size_t align, Memory memory) {
    MTL::HeapDescriptor* heap_info = MTL::HeapDescriptor::alloc()->init();
    heap_info->setType(MTL::HeapTypePlacement);
    heap_info->setStorageMode(memory == Memory::Gpu ? MTL::StorageModePrivate
                                                    : MTL::StorageModeShared);
    heap_info->setCpuCacheMode(memory == Memory::Default ? MTL::CPUCacheModeWriteCombined
                                                         : MTL::CPUCacheModeDefaultCache);
    heap_info->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    heap_info->setSize(bytes);

    MTL::Heap* heap = d->m_device->newHeap(heap_info);
    heap_info->release();

    MTL::Buffer* buffer = heap->newBuffer(bytes, 0, 0);
    if (!buffer) { return {}; }

    auto handle = d->m_buffer_pool.emplace({
        .buffer = buffer,
    });
    d->m_residency_set->addAllocation(heap);
    d->m_residency_set->commit();
    // TODO: Can i reduce the frenquency of commits at all?

    return handle;
}

void free(Device d, Handle<Buffer> buffer) {
    d->m_buffer_pool.erase(buffer);
}

GpuPtr get_device_pointer(Device d, Handle<Buffer> buffer) {
    return d->m_buffer_pool[buffer].buffer->gpuAddress();
}

void* get_host_pointer(Device d, Handle<Buffer> buffer) {
    return d->m_buffer_pool[buffer].buffer->contents();
}

BufferAndOffset buffer_and_offset_from_ptr(Device d, GpuPtr ptr) {
    const auto  it = lower_bound(d->m_ptr_map.begin(), d->m_ptr_map.end(), GpuPtrMap{.ptr = ptr});
    const auto& b  = d->m_buffer_pool[it->buffer];
    return {
        .buffer = b.buffer,
        .offset = static_cast<uint32_t>(ptr - b.device_ptr),
        .heap   = b.heap,
    };
}

// MARK: Textures
TextureSizeAlign get_texture_size_align(Device d, const TextureDesc& desc) {
    MTL::TextureDescriptor* info = MTL::TextureDescriptor::alloc()->init();
    info->setTextureType(bridge(desc.type));
    info->setPixelFormat(bridge(desc.format));
    info->setWidth(desc.dimensions.x);
    info->setHeight(desc.dimensions.y);
    info->setDepth(desc.dimensions.z);
    info->setMipmapLevelCount(desc.mip_count);
    info->setSampleCount(desc.sample_count);
    info->setArrayLength(desc.layer_count);
    info->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    info->setUsage(bridge_texture_usage(desc.usage));

    const auto size_align = d->m_device->heapTextureSizeAndAlign(info);
    info->release();

    return {.size = size_align.size, .align = size_align.align};
}

Handle<Texture> create_texture(Device d, const TextureDesc& desc, GpuPtr location) {
    MTL::TextureDescriptor* info = MTL::TextureDescriptor::alloc()->init();
    info->setTextureType(bridge(desc.type));
    info->setPixelFormat(bridge(desc.format));
    info->setWidth(desc.dimensions.x);
    info->setHeight(desc.dimensions.y);
    info->setDepth(desc.dimensions.z);
    info->setMipmapLevelCount(desc.mip_count);
    info->setSampleCount(desc.sample_count);
    info->setArrayLength(desc.layer_count);
    info->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    info->setUsage(bridge_texture_usage(desc.usage));

    MTL::Texture* texture = nullptr;
    if (location != 0) {
        // Specify location, use the mtlheap from the buffer
        auto memory = buffer_and_offset_from_ptr(d, location);
        texture     = memory.heap->newTexture(info, memory.offset);
    } else {
        texture = d->m_device->newTexture(info);
        d->m_residency_set->addAllocation(texture);
        d->m_residency_set->commit();
    }

    info->release();


    auto handle = d->m_texture_pool.emplace({.texture = texture});
    return handle;
}

Handle<TextureHeap> create_texture_heap(Device d, size_t size) {
    auto view_pool_descriptor = MTL::ResourceViewPoolDescriptor::alloc();
    view_pool_descriptor->setResourceViewCount(size);

    auto texture_view_pool = d->m_device->newTextureViewPool(view_pool_descriptor, nullptr);
    view_pool_descriptor->release();

    const auto handle = d->m_texture_heap_pool.emplace(TextureHeap{
        .pool   = texture_view_pool,
        .bitset = TwoLevelBitset(d->m_allocator, size),
    });
    return handle;
}

void free(Device d, Handle<Texture> h) {
    d->m_texture_pool.erase(h);
}

void free(Device d, Handle<TextureHeap> h) {
    d->m_texture_heap_pool.erase(h);
}

TextureView add_texture_view_to_heap(Device d, Handle<TextureHeap> h, const TextureViewDesc& desc) {
    auto&                       heap      = d->m_texture_heap_pool[h];
    auto&                       tex       = d->m_texture_pool[desc.texture];
    MTL::TextureViewDescriptor* view_info = MTL::TextureViewDescriptor::alloc()->init();
    view_info->setLevelRange(NS::Range(desc.base_mip, desc.mip_count));
    view_info->setPixelFormat(bridge(desc.format));
    view_info->setSliceRange(NS::Range(desc.base_layer, desc.layer_count));
    view_info->setTextureType(tex.texture->textureType());
    const auto      index        = heap.bitset.set_leading_zero();
    MTL::ResourceID texture_view = heap.pool->setTextureView(tex.texture, view_info, index);
    view_info->release();

    return texture_view._impl;
}

void remove_texture_view_from_heap(Device d, Handle<TextureHeap> h, TextureView view) {
    auto&      heap = d->m_texture_heap_pool[h];
    const auto base = heap.pool->baseResourceID()._impl;
    assert(view > base && view < base + heap.pool->resourceViewCount());
    const auto index = view - heap.pool->baseResourceID()._impl;
    heap.bitset.clear_bit(index);
}


// MARK: Pipelines
Handle<Pipeline> create_compute_pipeline(Device d, ShaderSource compute) {
    NS::String* shader_source = NS::String::alloc()->init(compute.spirv.data(),
                                                          compute.spirv.size(),
                                                          NS::UTF8StringEncoding,
                                                          false);
    NS::String* entry_point   = NS::String::alloc()->init((void*)compute.entry_point.data(),
                                                        compute.entry_point.size(),
                                                        NS::UTF8StringEncoding,
                                                        false);
    NS::Error*  error         = nullptr;

    MTL4::LibraryDescriptor* lib_desc = MTL4::LibraryDescriptor::alloc()->init();
    lib_desc->setSource(shader_source);
    MTL::Library* lib = d->m_compiler->newLibrary(lib_desc, &error);

    auto desc = MTL4::ComputePipelineDescriptor::alloc()->init();

    auto func_desc = MTL4::LibraryFunctionDescriptor::alloc()->init();
    func_desc->setLibrary(lib);
    func_desc->setName(entry_point);
    desc->setComputeFunctionDescriptor(func_desc);


    MTL::ComputePipelineState* compute_pipeline
        = d->m_compiler->newComputePipelineState(desc, d->m_options, &error);

    func_desc->release();
    desc->release();
    lib_desc->release();
    lib->release();
    entry_point->release();
    shader_source->release();

    return d->m_pipeline_pool.emplace({
        .compute_pipeline = compute_pipeline,
        .render_pipeline  = nullptr,
    });
}

Handle<Pipeline> create_graphics_pipeline(Device            d,
                                          ShaderSource      vertex,
                                          ShaderSource      fragment,
                                          const RasterDesc& desc) {
    // TODO: Error handling/propagation
    NS::String* vert_source      = NS::String::alloc()->init(vertex.spirv.data(),
                                                        vertex.spirv.size(),
                                                        NS::UTF8StringEncoding,
                                                        false);
    NS::String* vert_entry_point = NS::String::alloc()->init((void*)vertex.entry_point.data(),
                                                             vertex.entry_point.size(),
                                                             NS::UTF8StringEncoding,
                                                             false);

    NS::String* frag_source      = NS::String::alloc()->init(fragment.spirv.data(),
                                                        fragment.spirv.size(),
                                                        NS::UTF8StringEncoding,
                                                        false);
    NS::String* frag_entry_point = NS::String::alloc()->init((void*)fragment.entry_point.data(),
                                                             fragment.entry_point.size(),
                                                             NS::UTF8StringEncoding,
                                                             false);
    NS::Error*  error            = nullptr;

    MTL4::LibraryDescriptor* vert_lib_desc = MTL4::LibraryDescriptor::alloc()->init();
    vert_lib_desc->setSource(vert_source);
    MTL::Library* vert_lib = d->m_compiler->newLibrary(vert_lib_desc, &error);
    vert_lib_desc->release();

    MTL4::LibraryDescriptor* frag_lib_desc = MTL4::LibraryDescriptor::alloc()->init();
    frag_lib_desc->setSource(frag_source);
    MTL::Library* frag_lib = d->m_compiler->newLibrary(frag_lib_desc, &error);
    frag_lib_desc->release();

    auto vert_func_desc = MTL4::LibraryFunctionDescriptor::alloc()->init();
    vert_func_desc->setLibrary(vert_lib);
    vert_func_desc->setName(vert_entry_point);

    auto frag_func_desc = MTL4::LibraryFunctionDescriptor::alloc()->init();
    frag_func_desc->setLibrary(frag_lib);
    frag_func_desc->setName(frag_entry_point);

    auto pipeline_desc = MTL4::RenderPipelineDescriptor::alloc()->init();
    pipeline_desc->setVertexFunctionDescriptor(vert_func_desc);
    pipeline_desc->setFragmentFunctionDescriptor(frag_func_desc);
    pipeline_desc->setInputPrimitiveTopology(bridge(desc.topology));
    pipeline_desc->setAlphaToCoverageState(desc.alpha_to_coverage
                                               ? MTL4::AlphaToCoverageStateEnabled
                                               : MTL4::AlphaToCoverageStateDisabled);
    pipeline_desc->setRasterizationEnabled(true);
    pipeline_desc->setRasterSampleCount(desc.sample_count);

    auto     color_attachments    = pipeline_desc->colorAttachments();
    uint32_t color_attachment_idx = 0;
    for (auto c : desc.color_targets) {
        auto attachment = color_attachments->object(color_attachment_idx++);
        attachment->setPixelFormat(bridge(c.format));
        const auto& state = c.blendstate;
        const bool  blend_disabled
            = state.color_op == Blend::Add && state.src_color_factor == Factor::One
              && state.dst_color_factor == Factor::Zero && state.alpha_op == Blend::Add
              && state.src_alpha_factor == Factor::One && state.dst_color_factor == Factor::Zero;

        attachment->setBlendingState(blend_disabled ? MTL4::BlendStateDisabled
                                                    : MTL4::BlendStateEnabled);
        attachment->setWriteMask(state.color_write_mask);

        attachment->setAlphaBlendOperation(bridge(state.alpha_op));
        attachment->setRgbBlendOperation(bridge(state.color_op));
        attachment->setSourceAlphaBlendFactor(bridge(state.src_alpha_factor));
        attachment->setDestinationAlphaBlendFactor(bridge(state.dst_alpha_factor));
    }

    MTL::RenderPipelineState* render_pipeline
        = d->m_compiler->newRenderPipelineState(pipeline_desc, d->m_options, &error);

    pipeline_desc->release();
    frag_func_desc->release();
    vert_func_desc->release();
    frag_lib->release();
    vert_lib->release();
    frag_entry_point->release();
    frag_source->release();
    vert_entry_point->release();
    vert_source->release();


    return d->m_pipeline_pool.emplace({
        .compute_pipeline = nullptr,
        .render_pipeline  = render_pipeline,
        .cull_mode        = desc.cull,
    });
}

void free(Device d, Handle<Pipeline> pipeline) {
    d->m_pipeline_pool.erase(pipeline);
}

// MARK: State Objects

Handle<DepthStencilState> create_depth_stencil_state(Device d, const DepthStencilDesc& desc) {
    auto info = MTL::DepthStencilDescriptor::alloc()->init();
    info->setDepthCompareFunction(bridge(desc.depth_test));
    info->setDepthWriteEnabled((desc.depth_mode & DepthFlags::Write) == DepthFlags::Write);

    auto backface_stencil = MTL::StencilDescriptor::alloc()->init();
    backface_stencil->setStencilFailureOperation(bridge(desc.stencil_back.fail_op));
    backface_stencil->setDepthFailureOperation(bridge(desc.stencil_back.depth_fail_op));
    backface_stencil->setDepthStencilPassOperation(bridge(desc.stencil_back.pass_op));
    backface_stencil->setStencilCompareFunction(bridge(desc.stencil_back.test));
    backface_stencil->setReadMask(desc.stencil_read_mask);
    backface_stencil->setWriteMask(desc.stencil_write_mask);
    info->setBackFaceStencil(backface_stencil);

    auto frontface_stencil = MTL::StencilDescriptor::alloc()->init();
    frontface_stencil->setStencilFailureOperation(bridge(desc.stencil_front.fail_op));
    frontface_stencil->setDepthFailureOperation(bridge(desc.stencil_front.depth_fail_op));
    frontface_stencil->setDepthStencilPassOperation(bridge(desc.stencil_front.pass_op));
    frontface_stencil->setStencilCompareFunction(bridge(desc.stencil_front.test));
    frontface_stencil->setReadMask(desc.stencil_read_mask);
    frontface_stencil->setWriteMask(desc.stencil_write_mask);
    info->setFrontFaceStencil(frontface_stencil);
    auto state = d->m_device->newDepthStencilState(info);
    return d->m_depth_stencil_state_pool.emplace({
        .state = state,
    });
}

void free(Device d, Handle<DepthStencilState> state) {
    d->m_depth_stencil_state_pool.erase(state);
}

// MARK: Semaphores

Handle<Semaphore> create_semaphore(Device d, uint64_t initValue) {
    auto event = d->m_device->newSharedEvent();
    event->setSignaledValue(initValue);
    return d->m_semaphore_pool.emplace({
        .event = event,
    });
}

void wait_semaphore(Device d, Handle<Semaphore> sema, uint64_t value) {
    d->m_semaphore_pool[sema].event->waitUntilSignaledValue(value, UINT64_MAX);
}

void free(Device d, Handle<Semaphore> sema) {
    d->m_semaphore_pool.erase(sema);
}


// MARK: Queue

Queue get_queue(Device d, QueueType type) {
    return nullptr;
}

CommandBuffer queue_start_command_recording(Queue q) {
    return CommandBuffer();
}

void queue_submit(Queue                     q,
                  Span<const CommandBuffer> command_buffers,
                  Span<const SemaphoreInfo> wait_semaphores,
                  Span<const SemaphoreInfo> signal_semaphores) {
    auto d     = q->device;
    auto arena = *get_thread_local_arena(d);

    for (auto s : wait_semaphores) {
        // NOTE: Metal doesn't support waiting for a specific stage, so this is a full pipeline
        // flush?
        MTL::SharedEvent* event = d->m_semaphore_pool[s.semaphore].event;
        q->command_queue->wait(event, s.value);
    }


    Span<MTL4::CommandBuffer*> commands;
    for (auto cmd : command_buffers) {
        MTL4::CommandBuffer* buf = reinterpret_cast<MTL4::CommandBuffer*>(cmd->command_buffer);
        buf->endCommandBuffer();
        commands = concat(&arena, commands, buf);
    }
    q->command_queue->commit(commands.data(), commands.size());

    for (auto s : signal_semaphores) {
        // NOTE: Metal doesn't support signaling after a specific stage.
        MTL::SharedEvent* event = d->m_semaphore_pool[s.semaphore].event;
        q->command_queue->signalEvent(event, s.value);
    }
}

void queue_cancel(Queue q, Span<const Handle<CommandBuffer>> command_buffers) {}

void queue_on_submitted_work_completed(Queue q, Function<void>&& fn) {
    q->pending_events.emplace_back(
        QueueImpl::Event{.completed_time = q->timeline_value, .callback = std::move(fn)});
    q->command_queue->signalEvent(q->callback_event, q->timeline_value++);
}

void queue_process_events(Queue q) {
    uint64_t current_time = q->callback_event->signaledValue();
    uint32_t i            = 0;
    while (i < q->pending_events.size() && q->pending_events[i].completed_time <= current_time) {
        q->pending_events[i].callback();
        i++;
    }
    if (i != 0) {
        q->pending_events.erase(q->pending_events.begin(), q->pending_events.begin() + i);
    }
}

// MARK: CommandBuffer

void cmd_memcpy(CommandBuffer cmd, GpuPtr destGpu, GpuPtr srcGpu, size_t size) {
    auto d = cmd->device;

    auto encoder = cmd->command_buffer->computeCommandEncoder();
    auto src     = buffer_and_offset_from_ptr(d, srcGpu);
    auto dst     = buffer_and_offset_from_ptr(d, destGpu);
    encoder->copyFromBuffer(src.buffer, src.offset, dst.buffer, dst.offset, size);
    encoder->endEncoding();
}

void cmd_copy_to_texture(CommandBuffer                  cmd,
                         GpuPtr                         srcGpu,
                         Handle<Texture>                texture,
                         const BufferToTextureCopyInfo& info) {
    auto d = cmd->device;

    auto encoder = cmd->command_buffer->computeCommandEncoder();
    auto src     = buffer_and_offset_from_ptr(d, srcGpu);

    // TODO: I need more info here than I'm currently getting, need an API change.
    // encoder->copyFromBuffer(src.buffer, src.offset,,info.buffer_image_size,)
}

void cmd_copy_from_texture(CommandBuffer   cmd,
                           GpuPtr          destGpu,
                           GpuPtr          srcGpu,
                           Handle<Texture> texture) {}

void cmd_set_texture_heap(CommandBuffer cmd, Handle<TextureHeap> heap) {
    // This is currently a NOOP since all textures are in the global residency set.
}

void cmd_barrier(CommandBuffer                 cmd,
                 StageFlags                    before,
                 StageFlags                    after,
                 Span<const TextureTransition> image_transitions,
                 HazardFlags                   hazards) {
    auto d = cmd->device;

    // cmd->
}

void cmd_set_pipeline(CommandBuffer cmd, Handle<Pipeline> pipeline) {}

void cmd_set_depth_stencil_state(CommandBuffer cmd, Handle<DepthStencilState> state) {}

void cmd_set_scissor_rect(CommandBuffer cmd, const Rect2D& rect) {}

void cmd_dispatch(CommandBuffer cmd, GpuPtr dataGpu, const Dimension3D& gridDimensions) {}

void cmd_dispatch_indirect(CommandBuffer cmd, GpuPtr dataGpu, GpuPtr gridDimensionsGpu) {}

void cmd_begin_render_pass(CommandBuffer cmd, RenderPassDesc desc) {}
void cmd_end_render_pass(CommandBuffer cmd) {}

void cmd_draw(CommandBuffer cmd,
              GpuPtr        vertexDataGpu,
              GpuPtr        fragmentDataGpu,
              uint32_t      vertexCount,
              uint32_t      instanceCount) {}
void cmd_draw_indexed_instanced(CommandBuffer cmd,
                                GpuPtr        vertexDataGpu,
                                GpuPtr        pixelDataGpu,
                                GpuPtr        indicesGpu,
                                uint32_t      indexCount,
                                uint32_t      instanceCount) {}
void cmd_draw_indexed_instanced_indirect(CommandBuffer cmd,
                                         GpuPtr        vertexDataGpu,
                                         GpuPtr        pixelDataGpu,
                                         GpuPtr        indicesGpu,
                                         GpuPtr        argsGpu) {}
void cmd_draw_indexed_instanced_indirect_multi(CommandBuffer cmd,
                                               GpuPtr        vertexDataGpu,
                                               GpuPtr        pixelDataGpu,
                                               GpuPtr        indicesGpu,
                                               GpuPtr        argsGpu,
                                               GpuPtr        drawCountGpu,
                                               uint32_t      maxDraws) {}

}  // namespace loon::gpu
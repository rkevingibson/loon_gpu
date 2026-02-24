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


struct Device::Impl {
    Impl(const DeviceDesc& desc);
    ~Impl();

    void add_ref() { atomic_fetch_add(&m_refcount, 1); }
    bool release_ref() { return atomic_fetch_add(&m_refcount, -1) - 1 == 0; }

    bool initialize(const DeviceDesc& desc);
    void wait_for_device_idle();

    // Surface:
    SurfaceCapabilities get_surface_capabilities();
    bool                configure_surface(const SurfaceConfiguration& config);
    void                unconfigure_surface();
    SurfaceTextureInfo  get_current_texture();
    SurfaceStatus       present(Queue queue);

    // Buffers:
    Handle<Buffer>  malloc(size_t bytes, Memory memory = Memory::Default);
    Handle<Buffer>  malloc(size_t bytes, size_t align, Memory memory = Memory::Default);
    void            free(Handle<Buffer> buffer);
    GpuPtr          get_device_pointer(Handle<Buffer> buffer);
    void*           get_host_pointer(Handle<Buffer> buffer);
    BufferAndOffset buffer_and_offset_from_ptr(GpuPtr ptr);

    // Textures:
    TextureSizeAlign    get_texture_size_align(const TextureDesc& desc);
    Handle<Texture>     create_texture(const TextureDesc& desc, GpuPtr location);
    Handle<TextureHeap> create_texture_heap(size_t size);

    TextureView add_texture_view_to_heap(Handle<TextureHeap>, const TextureViewDesc& desc);
    void        remove_texture_view_from_heap(Handle<TextureHeap>, TextureView);

    void free(Handle<Texture>);
    void free(Handle<TextureHeap>);

    // Pipelines
    Handle<Pipeline> create_compute_pipeline(ShaderSource computeIR);
    Handle<Pipeline> create_graphics_pipeline(ShaderSource      vertex,
                                              ShaderSource      fragment,
                                              const RasterDesc& desc);
    void             free(Handle<Pipeline> pipeline);

    // State objects
    Handle<DepthStencilState> create_depth_stencil_state(const DepthStencilDesc& desc);
    void                      free(Handle<DepthStencilState> state);

    // Queue
    Queue get_queue(QueueType type);

    // Semaphores
    Handle<Semaphore> create_semaphore(uint64_t initValue);
    void              wait_semaphore(Handle<Semaphore> sema, uint64_t value);
    void              free(Handle<Semaphore> sema);

    void   log(LogLevel lvl, Span<const char> msg);
    Arena* get_thread_local_arena();

   private:
    friend class Queue;

    int64_t            m_refcount = 1;
    Allocator          m_allocator;
    ProcLogCallback    m_log_callback = nullptr;
    void*              m_log_userdata = nullptr;
    LogLevel           m_log_level    = LogLevel::Off;
    loon::gpu::tls_key m_tls_key;

    MTL::Device* m_device = nullptr;

    MTL4::Compiler*            m_compiler;
    MTL4::CompilerTaskOptions* m_options;

    MTL::ResidencySet* m_residency_set = nullptr;

    SlotMap<Buffer>            m_buffer_pool;
    SlotMap<Texture>           m_texture_pool;
    SlotMap<TextureHeap>       m_texture_heap_pool;
    SlotMap<Pipeline>          m_pipeline_pool;
    SlotMap<DepthStencilState> m_depth_stencil_state_pool;
    SlotMap<Semaphore>         m_semaphore_pool;

    Surface m_surface;

    struct GpuPtrMap {
        GpuPtr         ptr;
        Handle<Buffer> buffer;
    };
    static constexpr auto kPtrMapCompare
        = [](const GpuPtrMap& a, const GpuPtrMap& b) -> bool { return a.ptr > b.ptr; };

    static constexpr auto lower_bound
        = [](GpuPtrMap* first, GpuPtrMap* last, const GpuPtrMap& value) {
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

    Vector<GpuPtrMap> m_ptr_map;
};

void Device::Impl::log(LogLevel lvl, Span<const char> msg) {
    m_log_callback(lvl, msg, m_log_userdata);
};

Arena* Device::Impl::get_thread_local_arena() {
    auto state = reinterpret_cast<ThreadLocalState*>(loon::gpu::tls_get_data(m_tls_key));
    if (state == nullptr) {
        auto tls_block = m_allocator.alloc(sizeof(ThreadLocalState));
        if (tls_block.ptr == nullptr) {
            log(LogLevel::Error, "Allocator out of memory"_sv);
            return nullptr;
        }
        state = ::new (tls_block.ptr) ThreadLocalState(m_allocator);
        loon::gpu::tls_set_data(m_tls_key, state);
    }
    return &state->arena;
}

Device::Impl::Impl(const DeviceDesc& desc) :
    m_allocator{desc.alloc_callback ? Allocator(desc.alloc_callback, desc.alloc_userdata)
                                    : Allocator()},
    m_log_callback(desc.log_callback),
    m_log_userdata(desc.log_userdata),
    m_log_level(desc.log_level),
    m_tls_key{loon::gpu::tls_alloc([](void* data) {
        auto state = reinterpret_cast<ThreadLocalState*>(data);
        state->~ThreadLocalState();
    })},
    m_buffer_pool(m_allocator,
                  [this](Buffer* b) {
                      if (b->buffer) {
                          m_residency_set->removeAllocation(b->buffer);
                          b->buffer->release();
                      }
                      b->~Buffer();
                  }),
    m_texture_pool(m_allocator,
                   [this](Texture* t) {
                       if (t->texture) {
                           m_residency_set->removeAllocation(t->texture);
                           t->texture->release();
                       }
                       t->~Texture();
                   }),
    m_texture_heap_pool(m_allocator,
                        [](TextureHeap* t) {
                            if (t->pool) { t->pool->release(); }
                            t->~TextureHeap();
                        }),
    m_pipeline_pool(m_allocator,
                    [](Pipeline* p) {
                        if (p->compute_pipeline) { p->compute_pipeline->release(); }
                        if (p->render_pipeline) { p->render_pipeline->release(); }
                    }),
    m_depth_stencil_state_pool(m_allocator,
                               [](DepthStencilState* s) {
                                   if (s->state) { s->state->release(); }
                               }),
    m_semaphore_pool(m_allocator, [](Semaphore* s) {
        if (s->event) { s->event->release(); }
    }) {}

Device::Impl::~Impl() {
    m_device->release();
}

bool Device::Impl::initialize(const DeviceDesc& desc) {
    m_device = MTL::CreateSystemDefaultDevice();
    if (!m_device) { return false; }

    NS::Error* error = nullptr;

    auto compiler_desc = MTL4::CompilerDescriptor::alloc()->init();
    m_compiler         = m_device->newCompiler(compiler_desc, &error);
    compiler_desc->release();
    m_options = MTL4::CompilerTaskOptions::alloc()->init();


    // Is this a valid cast? We're being passed a CAMetalLayer, not the C++ wrapper. Not sure!
    // At the very least I want to retain it.
    m_surface.metal_layer = reinterpret_cast<CA::MetalLayer*>(desc.native_window_handle);
    m_surface.metal_layer->retain();

    auto residency_set_descriptor = MTL::ResidencySetDescriptor::alloc()->init();
    // NOTE: Not sure how much this matters, should dig in more
    residency_set_descriptor->setInitialCapacity(64);
    m_residency_set = m_device->newResidencySet(residency_set_descriptor, nullptr);
    residency_set_descriptor->release();



    return true;
}

void Device::Impl::wait_for_device_idle() {
    // TODO: Not sure how to implement this one.
}

// MARK: Surface functions
SurfaceCapabilities Device::Impl::get_surface_capabilities() {
    return SurfaceCapabilities{
        .formats       = Surface::kSwapchainFormats,
        .present_modes = Surface::kPresentModes,
        .usages = UsageFlags::ColorAttachment | UsageFlags::TransferDst | UsageFlags::Storage,
    };
}

bool Device::Impl::configure_surface(const SurfaceConfiguration& config) {
    m_surface.metal_layer->setPixelFormat(bridge(config.format));
    m_surface.metal_layer->setDrawableSize(CGSize{
        .width  = static_cast<float>(config.width),
        .height = static_cast<float>(config.height),
    });
    return true;
}

void Device::Impl::unconfigure_surface() {
    // Not much to do here.
    if (m_surface.current_drawable) {
        m_surface.current_drawable->release();
        m_surface.current_drawable = nullptr;
    }
}

SurfaceTextureInfo Device::Impl::get_current_texture() {
    CA::MetalDrawable* drawable = m_surface.metal_layer->nextDrawable();

    if (!drawable) {
        return SurfaceTextureInfo{
            .texture = 0,
            .status  = SurfaceStatus::Error,
        };
    }

    m_surface.current_drawable = drawable;
    MTL::Texture*   tex        = drawable->texture();
    Handle<Texture> handle     = m_texture_pool.emplace({
            .texture = tex,
    });

    return SurfaceTextureInfo{
        .texture = handle,
        .status  = SurfaceStatus::Success,
    };
}

SurfaceStatus Device::Impl::present(Queue queue) {
    auto queue_impl = reinterpret_cast<QueueImpl*>(queue.queue);
    queue_impl->command_queue->signalDrawable(m_surface.current_drawable);

    m_surface.current_drawable->present();
    m_surface.current_drawable->release();
    m_surface.current_drawable = nullptr;
    return SurfaceStatus::Success;
}

// MARK: Buffers:

Handle<Buffer> Device::Impl::malloc(size_t bytes, Memory memory) {
    return malloc(bytes, 64, memory);
}

Handle<Buffer> Device::Impl::malloc(size_t bytes, size_t align, Memory memory) {
    MTL::HeapDescriptor* heap_info = MTL::HeapDescriptor::alloc()->init();
    heap_info->setType(MTL::HeapTypePlacement);
    heap_info->setStorageMode(memory == Memory::Gpu ? MTL::StorageModePrivate
                                                    : MTL::StorageModeShared);
    heap_info->setCpuCacheMode(memory == Memory::Default ? MTL::CPUCacheModeWriteCombined
                                                         : MTL::CPUCacheModeDefaultCache);
    heap_info->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    heap_info->setSize(bytes);

    MTL::Heap* heap = m_device->newHeap(heap_info);
    heap_info->release();

    MTL::Buffer* buffer = heap->newBuffer(bytes, 0, 0);
    if (!buffer) { return {}; }

    auto handle = m_buffer_pool.emplace({
        .buffer = buffer,
    });
    m_residency_set->addAllocation(heap);
    m_residency_set->commit();
    // TODO: Can i reduce the frenquency of commits at all?

    return handle;
}

void Device::Impl::free(Handle<Buffer> buffer) {
    m_buffer_pool.erase(buffer);
}

GpuPtr Device::Impl::get_device_pointer(Handle<Buffer> buffer) {
    return m_buffer_pool[buffer].buffer->gpuAddress();
}

void* Device::Impl::get_host_pointer(Handle<Buffer> buffer) {
    return m_buffer_pool[buffer].buffer->contents();
}

BufferAndOffset Device::Impl::buffer_and_offset_from_ptr(GpuPtr ptr) {
    const auto  it = lower_bound(m_ptr_map.begin(), m_ptr_map.end(), GpuPtrMap{.ptr = ptr});
    const auto& b  = m_buffer_pool[it->buffer];
    return {
        .buffer = b.buffer,
        .offset = static_cast<uint32_t>(ptr - b.device_ptr),
        .heap   = b.heap,
    };
}

// MARK: Textures
TextureSizeAlign Device::Impl::get_texture_size_align(const TextureDesc& desc) {
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

    const auto size_align = m_device->heapTextureSizeAndAlign(info);
    info->release();

    return {.size = size_align.size, .align = size_align.align};
}

Handle<Texture> Device::Impl::create_texture(const TextureDesc& desc, GpuPtr location) {
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
        auto memory = buffer_and_offset_from_ptr(location);
        texture     = memory.heap->newTexture(info, memory.offset);
    } else {
        texture = m_device->newTexture(info);
        m_residency_set->addAllocation(texture);
        m_residency_set->commit();
    }

    info->release();


    auto handle = m_texture_pool.emplace({.texture = texture});
    return handle;
}

Handle<TextureHeap> Device::Impl::create_texture_heap(size_t size) {
    auto view_pool_descriptor = MTL::ResourceViewPoolDescriptor::alloc();
    view_pool_descriptor->setResourceViewCount(size);

    auto texture_view_pool = m_device->newTextureViewPool(view_pool_descriptor, nullptr);
    view_pool_descriptor->release();

    const auto handle = m_texture_heap_pool.emplace(TextureHeap{
        .pool   = texture_view_pool,
        .bitset = TwoLevelBitset(m_allocator, size),
    });
    return handle;
}

void Device::Impl::free(Handle<Texture> h) {
    m_texture_pool.erase(h);
}

void Device::Impl::free(Handle<TextureHeap> h) {
    m_texture_heap_pool.erase(h);
}

TextureView Device::Impl::add_texture_view_to_heap(Handle<TextureHeap>    h,
                                                   const TextureViewDesc& desc) {
    auto&                       heap      = m_texture_heap_pool[h];
    auto&                       tex       = m_texture_pool[desc.texture];
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

void Device::Impl::remove_texture_view_from_heap(Handle<TextureHeap> h, TextureView view) {
    auto&      heap = m_texture_heap_pool[h];
    const auto base = heap.pool->baseResourceID()._impl;
    assert(view > base && view < base + heap.pool->resourceViewCount());
    const auto index = view - heap.pool->baseResourceID()._impl;
    heap.bitset.clear_bit(index);
}


// MARK: Pipelines
Handle<Pipeline> Device::Impl::create_compute_pipeline(ShaderSource compute) {
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
    MTL::Library* lib = m_compiler->newLibrary(lib_desc, &error);

    auto desc = MTL4::ComputePipelineDescriptor::alloc()->init();

    auto func_desc = MTL4::LibraryFunctionDescriptor::alloc()->init();
    func_desc->setLibrary(lib);
    func_desc->setName(entry_point);
    desc->setComputeFunctionDescriptor(func_desc);


    MTL::ComputePipelineState* compute_pipeline
        = m_compiler->newComputePipelineState(desc, m_options, &error);

    func_desc->release();
    desc->release();
    lib_desc->release();
    lib->release();
    entry_point->release();
    shader_source->release();

    return m_pipeline_pool.emplace({
        .compute_pipeline = compute_pipeline,
        .render_pipeline  = nullptr,
    });
}

Handle<Pipeline> Device::Impl::create_graphics_pipeline(ShaderSource      vertex,
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
    MTL::Library* vert_lib = m_compiler->newLibrary(vert_lib_desc, &error);
    vert_lib_desc->release();

    MTL4::LibraryDescriptor* frag_lib_desc = MTL4::LibraryDescriptor::alloc()->init();
    frag_lib_desc->setSource(frag_source);
    MTL::Library* frag_lib = m_compiler->newLibrary(frag_lib_desc, &error);
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
        = m_compiler->newRenderPipelineState(pipeline_desc, m_options, &error);

    pipeline_desc->release();
    frag_func_desc->release();
    vert_func_desc->release();
    frag_lib->release();
    vert_lib->release();
    frag_entry_point->release();
    frag_source->release();
    vert_entry_point->release();
    vert_source->release();


    return m_pipeline_pool.emplace({
        .compute_pipeline = nullptr,
        .render_pipeline  = render_pipeline,
        .cull_mode        = desc.cull,
    });
}

void Device::Impl::free(Handle<Pipeline> pipeline) {
    m_pipeline_pool.erase(pipeline);
}

// MARK: State Objects

Handle<DepthStencilState> Device::Impl::create_depth_stencil_state(const DepthStencilDesc& desc) {
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
    auto state = m_device->newDepthStencilState(info);
    return m_depth_stencil_state_pool.emplace({
        .state = state,
    });
}

void Device::Impl::free(Handle<DepthStencilState> state) {
    m_depth_stencil_state_pool.erase(state);
}

// MARK: Semaphores

Handle<Semaphore> Device::Impl::create_semaphore(uint64_t initValue) {
    auto event = m_device->newSharedEvent();
    event->setSignaledValue(initValue);
    return m_semaphore_pool.emplace({
        .event = event,
    });
}

void Device::Impl::wait_semaphore(Handle<Semaphore> sema, uint64_t value) {
    m_semaphore_pool[sema].event->waitUntilSignaledValue(value, UINT64_MAX);
}

void Device::Impl::free(Handle<Semaphore> sema) {
    m_semaphore_pool.erase(sema);
}


// MARK: Queue

Queue Device::Impl::get_queue(QueueType type) {
    return Queue(nullptr, nullptr);
}

CommandBuffer Queue::start_command_recording() {
    return CommandBuffer();
}

void Queue::submit(Span<const CommandBuffer> command_buffers,
                   Span<const SemaphoreInfo> wait_semaphores,
                   Span<const SemaphoreInfo> signal_semaphores) {
    auto q     = reinterpret_cast<QueueImpl*>(queue);
    auto d     = reinterpret_cast<Device::Impl*>(device);
    auto arena = *d->get_thread_local_arena();

    for (auto s : wait_semaphores) {
        // NOTE: Metal doesn't support waiting for a specific stage, so this is a full pipeline
        // flush?
        MTL::SharedEvent* event = d->m_semaphore_pool[s.semaphore].event;
        q->command_queue->wait(event, s.value);
    }


    Span<MTL4::CommandBuffer*> commands;
    for (auto cmd : command_buffers) {
        MTL4::CommandBuffer* buf = reinterpret_cast<MTL4::CommandBuffer*>(cmd.buffer);
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

void Queue::cancel(Span<const Handle<CommandBuffer>> command_buffers) {}

void Queue::on_submitted_work_completed(Function<void>&& fn) {
    auto q = reinterpret_cast<QueueImpl*>(queue);

    q->pending_events.emplace_back(
        QueueImpl::Event{.completed_time = q->timeline_value, .callback = std::move(fn)});
    q->command_queue->signalEvent(q->callback_event, q->timeline_value++);
}

void Queue::process_events() {
    auto     q            = reinterpret_cast<QueueImpl*>(queue);
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

void CommandBuffer::memcpy(GpuPtr destGpu, GpuPtr srcGpu, size_t size) {
    auto d   = reinterpret_cast<Device::Impl*>(device);
    auto cmd = reinterpret_cast<MTL4::CommandBuffer*>(buffer);

    auto encoder = cmd->computeCommandEncoder();
    auto src     = d->buffer_and_offset_from_ptr(srcGpu);
    auto dst     = d->buffer_and_offset_from_ptr(destGpu);
    encoder->copyFromBuffer(src.buffer, src.offset, dst.buffer, dst.offset, size);
    encoder->endEncoding();
}

void CommandBuffer::copy_to_texture(GpuPtr                         srcGpu,
                                    Handle<Texture>                texture,
                                    const BufferToTextureCopyInfo& info) {
    auto d   = reinterpret_cast<Device::Impl*>(device);
    auto cmd = reinterpret_cast<MTL4::CommandBuffer*>(buffer);

    auto encoder = cmd->computeCommandEncoder();
    auto src     = d->buffer_and_offset_from_ptr(srcGpu);

    // TODO: I need more info here than I'm currently getting, need an API change.
    // encoder->copyFromBuffer(src.buffer, src.offset,,info.buffer_image_size,)
}

void CommandBuffer::copy_from_texture(GpuPtr destGpu, GpuPtr srcGpu, Handle<Texture> texture) {}

void CommandBuffer::set_texture_heap(Handle<TextureHeap> heap) {
    // This is currently a NOOP since all textures are in the global residency set.
}

void CommandBuffer::barrier(StageFlags                    before,
                            StageFlags                    after,
                            Span<const TextureTransition> image_transitions,
                            HazardFlags                   hazards) {
    auto d   = reinterpret_cast<Device::Impl*>(device);
    auto cmd = reinterpret_cast<MTL4::CommandBuffer*>(buffer);

    // cmd->
}

void CommandBuffer::set_pipeline(Handle<Pipeline> pipeline) {}

void CommandBuffer::set_depth_stencil_state(Handle<DepthStencilState> state) {}

void CommandBuffer::set_scissor_rect(const Rect2D& rect) {}

void CommandBuffer::dispatch(GpuPtr dataGpu, const Dimension3D& gridDimensions) {}

void CommandBuffer::dispatch_indirect(GpuPtr dataGpu, GpuPtr gridDimensionsGpu) {}

void CommandBuffer::begin_render_pass(RenderPassDesc desc) {}
void CommandBuffer::end_render_pass() {}

void CommandBuffer::draw(GpuPtr   vertexDataGpu,
                         GpuPtr   fragmentDataGpu,
                         uint32_t vertexCount,
                         uint32_t instanceCount) {}
void CommandBuffer::draw_indexed_instanced(GpuPtr   vertexDataGpu,
                                           GpuPtr   pixelDataGpu,
                                           GpuPtr   indicesGpu,
                                           uint32_t indexCount,
                                           uint32_t instanceCount) {}
void CommandBuffer::draw_indexed_instanced_indirect(GpuPtr vertexDataGpu,
                                                    GpuPtr pixelDataGpu,
                                                    GpuPtr indicesGpu,
                                                    GpuPtr argsGpu) {}
void CommandBuffer::draw_indexed_instanced_indirect_multi(GpuPtr   vertexDataGpu,
                                                          GpuPtr   pixelDataGpu,
                                                          GpuPtr   indicesGpu,
                                                          GpuPtr   argsGpu,
                                                          GpuPtr   drawCountGpu,
                                                          uint32_t maxDraws) {}

// MARK: Device wrapper

Device Device::create(const DeviceDesc& desc) {
    Impl* impl = new Impl(desc);
    if (impl->initialize(desc)) { return Device(impl); }

    return Device(nullptr);
}

Device::~Device() {
    if (impl && impl->release_ref()) { delete impl; }
}

Device::Device(const Device& other) : impl{other.impl} {
    impl->add_ref();
}

Device& Device::operator=(const Device& other) {
    if (this == &other || other.impl == impl) return *this;
    other.impl->add_ref();
    if (impl && impl->release_ref()) { delete impl; }
    impl = other.impl;
    return *this;
}

void Device::wait_for_device_idle() {
    impl->wait_for_device_idle();
}

SurfaceCapabilities Device::get_surface_capabilities() {
    return impl->get_surface_capabilities();
}

bool Device::configure_surface(const SurfaceConfiguration& config) {
    return impl->configure_surface(config);
}

void Device::unconfigure_surface() {
    return impl->unconfigure_surface();
}

SurfaceTextureInfo Device::get_current_texture() {
    return impl->get_current_texture();
}

SurfaceStatus Device::present(Queue queue) {
    return impl->present(queue);
}

Handle<Buffer> Device::malloc(size_t bytes, Memory memory) {
    return impl->malloc(bytes, memory);
}

Handle<Buffer> Device::malloc(size_t bytes, size_t align, Memory memory) {
    return impl->malloc(bytes, align, memory);
}

void Device::free(Handle<Buffer> buffer) {
    return impl->free(buffer);
}

GpuPtr Device::get_device_pointer(Handle<Buffer> buffer) {
    return impl->get_device_pointer(buffer);
}

void* Device::get_host_pointer(Handle<Buffer> buffer) {
    return impl->get_host_pointer(buffer);
}

Handle<Texture> Device::create_texture(const TextureDesc& desc, GpuPtr location) {
    return impl->create_texture(desc, location);
}
Handle<TextureHeap> Device::create_texture_heap(size_t size) {
    return impl->create_texture_heap(size);
}

TextureView Device::add_texture_view_to_heap(Handle<TextureHeap>    heap,
                                             const TextureViewDesc& desc) {
    return impl->add_texture_view_to_heap(heap, desc);
}

void Device::remove_texture_view_from_heap(Handle<TextureHeap> heap, TextureView idx) {
    return impl->remove_texture_view_from_heap(heap, idx);
}

void Device::free(Handle<Texture> h) {
    return impl->free(h);
}

void Device::free(Handle<TextureHeap> h) {
    return impl->free(h);
}

Handle<Pipeline> Device::create_compute_pipeline(ShaderSource source) {
    return impl->create_compute_pipeline(source);
}

Handle<Pipeline> Device::create_graphics_pipeline(ShaderSource      vertex,
                                                  ShaderSource      fragment,
                                                  const RasterDesc& desc) {
    return impl->create_graphics_pipeline(vertex, fragment, desc);
}

void Device::free(Handle<Pipeline> pipeline) {
    return impl->free(pipeline);
}

Handle<DepthStencilState> Device::create_depth_stencil_state(const DepthStencilDesc& desc) {
    return impl->create_depth_stencil_state(desc);
}

Queue Device::get_queue(QueueType type) {
    return impl->get_queue(type);
}

Handle<Semaphore> Device::create_semaphore(uint64_t initValue) {
    return impl->create_semaphore(initValue);
}

void Device::wait_semaphore(Handle<Semaphore> sema, uint64_t value) {
    impl->wait_semaphore(sema, value);
}

void Device::free(Handle<Semaphore> sema) {
    impl->free(sema);
}

}  // namespace loon::gpu
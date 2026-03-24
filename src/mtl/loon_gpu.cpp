#include <gpu/loon_gpu.h>

#include <cassert>
#include <cstddef>
#include <cstring>

#include "platform_utils.h"

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

template <class T, class CompareFn>
static constexpr auto lower_bound = [](T* first, T* last, const T& value) {
    CompareFn compare;
    T*        it;
    size_t    count = last - first;
    while (count > 0) {
        const size_t step = count / 2;
        it                = first + step;
        if (compare(*it, value)) {
            first = ++it;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return first;
};

struct Buffer {
    MTL::Buffer* buffer;
    MTL::Heap*   heap;
};

struct Texture {
    MTL::Texture* texture;
};

struct SamplerMapping {
    Sampler            sampler;
    MTL::SamplerState* state;
};

struct SamplerMappingCompare {
    constexpr bool operator()(const SamplerMapping& a, const SamplerMapping& b) {
        return a.sampler > b.sampler;
    }
};

struct TextureHeap {
    MTL::TextureViewPool*  pool = nullptr;
    TwoLevelBitset         bitset;
    Vector<SamplerMapping> sampler_lookup;
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

struct Surface {
    CA::MetalLayer*    metal_layer;
    CA::MetalDrawable* current_drawable = nullptr;
    uint64_t           frame_idx        = 0;

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

struct CommandPool;
struct CommandBufferImpl {
    MTL4::CommandBuffer*         command_buffer = nullptr;
    Queue                        queue;
    CommandPool*                 pool;
    Device                       device;
    MTL4::ComputeCommandEncoder* compute_encoder = nullptr;
    MTL4::RenderCommandEncoder*  render_encoder  = nullptr;
};

struct CommandPool {
    MTL4::CommandAllocator*         allocator = nullptr;
    SegmentArray<CommandBufferImpl> command_buffers;  // In theory
    uint64_t                        buffer_free_idx = 0;
    uint64_t                        frame_idx       = 0;
};

struct CommandSuperpool {
    static constexpr uint32_t kPoolsPerGroup                                   = 3;
    static constexpr uint32_t kMaxSimultaneousCommands                         = 64;
    int64_t                   available_pools                                  = ~0;
    CommandPool               pools[kMaxSimultaneousCommands * kPoolsPerGroup] = {};
};

struct QueueImpl {
    struct Event {
        uint64_t       completed_time;
        Function<void> callback;
    };

    MTL4::CommandQueue* command_queue = nullptr;
    CommandSuperpool    command_superpool;
    MTL::SharedEvent*   callback_event = nullptr;
    Vector<Event>       pending_events;
    uint64_t            timeline_value = 0;

    Device device = nullptr;
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
    Buffer*  buffer;
    uint32_t offset;
};

struct GpuPtrMap {
    GpuPtr         ptr;
    Handle<Buffer> buffer;
};

struct PtrMapCompare {
    constexpr bool operator()(const GpuPtrMap& a, const GpuPtrMap& b) { return a.ptr > b.ptr; }
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

    QueueImpl m_queue;

    SlotMap<Buffer>            m_buffer_pool;
    SlotMap<Texture>           m_texture_pool;
    SlotMap<TextureHeap>       m_texture_heap_pool;
    SlotMap<Pipeline>          m_pipeline_pool;
    SlotMap<DepthStencilState> m_depth_stencil_state_pool;
    SlotMap<Semaphore>         m_semaphore_pool;
    rwlock                     m_ptr_map_lock = LOON_RWLOCK_INIT;
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

void destroy_device(Device d) {
    device_wait_for_idle(d);
    unconfigure_surface(d);

    auto& q = d->m_queue;
    if (q.command_queue) {
        for (auto& p : q.command_superpool.pools) {
            if (p.allocator) { p.allocator->release(); }
            for (uint32_t i = 0; i < p.command_buffers.size(); ++i) {
                p.command_buffers[i].command_buffer->release();
            }
        }

        q.callback_event->release();
    }

    d->m_semaphore_pool.clear();
    d->m_depth_stencil_state_pool.clear();
    d->m_pipeline_pool.clear();
    d->m_texture_heap_pool.clear();
    d->m_texture_pool.clear();
    d->m_buffer_pool.clear();

    d->m_residency_set->release();
    d->m_options->release();
    d->m_compiler->release();
    d->m_device->release();

    tls_free(d->m_tls_key);
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
    d->m_surface.frame_idx++;
    MTL::Texture*   tex    = drawable->texture();
    Handle<Texture> handle = d->m_texture_pool.emplace({
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

GpuPtr malloc(Device d, size_t bytes, Memory memory) {
    return malloc(d, bytes, 64, memory);
}

GpuPtr malloc(Device d, size_t bytes, size_t align, Memory memory) {
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
        .heap   = heap,
    });
    d->m_residency_set->addAllocation(heap);
    d->m_residency_set->commit();
    // TODO: Can i reduce the frenquency of commits at all?

    rwlock_lock_write(&d->m_ptr_map_lock);
    const auto insertion_pos = lower_bound<GpuPtrMap, PtrMapCompare>(d->m_ptr_map.begin(),
                                                                     d->m_ptr_map.end(),
                                                                     {.ptr = buffer->gpuAddress()});
    d->m_ptr_map.insert(insertion_pos, {.ptr = buffer->gpuAddress(), .buffer = handle});
    rwlock_unlock_write(&d->m_ptr_map_lock);

    return buffer->gpuAddress();
}

BufferAndOffset buffer_and_offset_from_ptr(Device d, GpuPtr ptr) {
    rwlock_lock_read(&d->m_ptr_map_lock);
    const auto it = lower_bound<GpuPtrMap, PtrMapCompare>(d->m_ptr_map.begin(),
                                                          d->m_ptr_map.end(),
                                                          GpuPtrMap{.ptr = ptr});
    auto       b  = &d->m_buffer_pool[it->buffer];
    rwlock_unlock_read(&d->m_ptr_map_lock);

    return {
        .buffer = b,
        .offset = static_cast<uint32_t>(ptr - b->buffer->gpuAddress()),
    };
}

void* get_host_pointer(Device d, GpuPtr ptr) {
    auto info = buffer_and_offset_from_ptr(d, ptr);
    return info.buffer->buffer->contents();
}

void free(Device d, GpuPtr ptr) {
    rwlock_lock_write(&d->m_ptr_map_lock);
    const auto it = lower_bound<GpuPtrMap, PtrMapCompare>(d->m_ptr_map.begin(),
                                                          d->m_ptr_map.end(),
                                                          GpuPtrMap{.ptr = ptr});
    assert(it->ptr == ptr);  // Shouldn't free from another pointer in the same allocation.
    d->m_buffer_pool.erase(it->buffer);
    d->m_ptr_map.erase(it, it + 1);
    rwlock_unlock_write(&d->m_ptr_map_lock);
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
        texture     = memory.buffer->heap->newTexture(info, memory.offset);
    } else {
        texture = d->m_device->newTexture(info);
        d->m_residency_set->addAllocation(texture);
        d->m_residency_set->commit();
    }

    info->release();
    auto handle = d->m_texture_pool.emplace({.texture = texture});
    return handle;
}

Handle<TextureHeap> create_texture_heap(Device d, const TextureHeapDesc& desc) {
    auto view_pool_descriptor = MTL::ResourceViewPoolDescriptor::alloc();
    view_pool_descriptor->setResourceViewCount(desc.texture_count);

    auto texture_view_pool = d->m_device->newTextureViewPool(view_pool_descriptor, nullptr);
    view_pool_descriptor->release();

    const auto handle = d->m_texture_heap_pool.emplace(TextureHeap{
        .pool           = texture_view_pool,
        .bitset         = TwoLevelBitset(d->m_allocator, desc.texture_count),
        .sampler_lookup = Vector<SamplerMapping>(d->m_allocator),
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

Sampler add_sampler_to_heap(Device d, Handle<TextureHeap> h, const SamplerDesc& sampler) {
    auto& heap = d->m_texture_heap_pool[h];

    MTL::SamplerDescriptor* desc = MTL::SamplerDescriptor::alloc()->init();
    desc->setNormalizedCoordinates(sampler.coord == SamplerCoords::Normalized);
    auto filter = bridge_minmag(sampler.filter);
    desc->setMagFilter(filter);
    desc->setMinFilter(filter);
    desc->setMipFilter(bridge_mip(sampler.filter));
    auto addressing = bridge(sampler.address);
    desc->setRAddressMode(addressing);
    desc->setSAddressMode(addressing);
    desc->setTAddressMode(addressing);
    desc->setMaxAnisotropy((uint32_t)sampler.max_anisotropy);
    auto sampler_state = d->m_device->newSamplerState(desc);
    desc->release();

    Sampler result = sampler_state->gpuResourceID()._impl;
    // Need to add to some list so we can free it easily. For now using a sorted list, could be a
    // hash map if we expect a lot of creation/freeing.
    const auto insertion_pos
        = lower_bound<SamplerMapping, SamplerMappingCompare>(heap.sampler_lookup.begin(),
                                                             heap.sampler_lookup.end(),
                                                             {.sampler = result});
    heap.sampler_lookup.insert(insertion_pos,
                               {
                                   .sampler = result,
                                   .state   = sampler_state,
                               });

    return result;
}

void remove_sampler_from_heap(Device d, Handle<TextureHeap> h, Sampler s) {
    auto& heap = d->m_texture_heap_pool[h];

    const auto it = lower_bound<SamplerMapping, SamplerMappingCompare>(heap.sampler_lookup.begin(),
                                                                       heap.sampler_lookup.end(),
                                                                       {.sampler = s});
    assert(it->sampler == s);
    it->state->release();
    heap.sampler_lookup.erase(it, it + 1);
}

// MARK: Pipelines
Handle<Pipeline> create_compute_pipeline(Device d, ShaderSource compute) {
    NS::String* shader_source = NS::String::alloc()->init((void*)compute.spirv.data(),
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
    NS::String* vert_source      = NS::String::alloc()->init((void*)vertex.spirv.data(),
                                                        vertex.spirv.size(),
                                                        NS::UTF8StringEncoding,
                                                        false);
    NS::String* vert_entry_point = NS::String::alloc()->init((void*)vertex.entry_point.data(),
                                                             vertex.entry_point.size(),
                                                             NS::UTF8StringEncoding,
                                                             false);

    NS::String* frag_source      = NS::String::alloc()->init((void*)fragment.spirv.data(),
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

void free_depth_stencil_state(Device d, Handle<DepthStencilState> state) {
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

static void reset_command_pool(CommandPool* pool) {
    pool->allocator->reset();
    pool->buffer_free_idx = 0;
}

static CommandPool* get_command_pool(Queue queue, uint64_t frame_idx) {
    CommandSuperpool& superpool       = queue->command_superpool;
    CommandPool*      pool            = nullptr;
    int64_t           available_pools = atomic_load(&superpool.available_pools);
    bool              index_good      = false;
    uint64_t          idx;
    while (!index_good && available_pools != 0) {
        idx                    = count_trailing_zeros(available_pools);
        const uint64_t mask    = ~(1ull << idx);
        const int64_t  desired = static_cast<int64_t>(available_pools & mask);
        index_good = atomic_compare_exchange(&superpool.available_pools, &available_pools, desired);
    }

    if (index_good) {
        pool = &superpool.pools[CommandSuperpool::kPoolsPerGroup * idx
                                + (frame_idx % CommandSuperpool::kPoolsPerGroup)];

        if (pool->allocator == nullptr) {
            // Initialize the command pool here.
            *pool = CommandPool{
                .allocator       = queue->device->m_device->newCommandAllocator(),
                .command_buffers = SegmentArray<CommandBufferImpl>(queue->device->m_allocator),
                .buffer_free_idx = 0,
                .frame_idx       = 0,
            };
        } else if (pool->frame_idx != frame_idx) {
            // Last time this was used was on a different frame, so reset the pool.
            reset_command_pool(pool);
        }
    } else {
        log(queue->device,
            LogLevel::Error,
            "Unable to get command pool - too many command buffers in flight at once"_sv);
    }

    return pool;
}

static void release_command_pool(Queue q, CommandPool* pool) {
    auto&         superpool = q->command_superpool;
    const int64_t idx       = (pool - superpool.pools) / CommandSuperpool::kPoolsPerGroup;

    // Need to set the bit in available pools using a compare-exchange loop
    int64_t previous = atomic_load(&superpool.available_pools);

    int64_t desired = previous | (1ll << idx);
    while (!atomic_compare_exchange(&superpool.available_pools, &previous, desired)) {
        desired = previous | (1ll << idx);
    }
}

static CommandBufferImpl* get_command_buffer(Queue q, CommandPool* pool) {
    auto device = q->device;

    if (pool->command_buffers.size() <= pool->buffer_free_idx) {
        pool->command_buffers.emplace_back(CommandBufferImpl{
            .command_buffer = device->m_device->newCommandBuffer(),
            .queue          = q,
            .pool           = pool,
            .device         = device,
        });
    }

    CommandBufferImpl* result = &pool->command_buffers[pool->buffer_free_idx];
    pool->buffer_free_idx++;
    return result;
}

Queue get_queue(Device d, QueueType type) {
    if (d->m_queue.command_queue == nullptr) {
        auto& q = d->m_queue = {
            .command_queue  = d->m_device->newMTL4CommandQueue(),
            .callback_event = d->m_device->newSharedEvent(),
            .pending_events = Vector<QueueImpl::Event>(d->m_allocator),
            .timeline_value = 0,
            .device         = d,
        };
    }

    return &d->m_queue;
}

CommandBuffer queue_start_command_recording(Queue q) {
    auto d = q->device;

    CommandPool* pool = get_command_pool(q, d->m_surface.frame_idx);
    if (pool == nullptr) { return nullptr; }

    CommandBuffer buffer = get_command_buffer(q, pool);
    if (buffer) {}
    return buffer;
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

static bool is_in_compute_pass(CommandBuffer cmd) {
    return (cmd->compute_encoder != nullptr);
}

static bool is_in_render_pass(CommandBuffer cmd) {
    return (cmd->render_encoder != nullptr);
}

static MTL4::ComputeCommandEncoder* get_compute_encoder(CommandBuffer cmd) {
    assert(!is_in_render_pass(cmd));
    if (!cmd->compute_encoder) {
        cmd->compute_encoder = cmd->command_buffer->computeCommandEncoder();
    }

    return cmd->compute_encoder;
}

static void end_compute_pass(CommandBuffer cmd) {
    assert(is_in_compute_pass(cmd));
    cmd->compute_encoder->endEncoding();
    cmd->compute_encoder->release();
    cmd->compute_encoder = nullptr;
}

void cmd_memcpy(CommandBuffer cmd, GpuPtr destGpu, GpuPtr srcGpu, size_t size) {
    auto d = cmd->device;

    auto encoder = get_compute_encoder(cmd);
    auto src     = buffer_and_offset_from_ptr(d, srcGpu);
    auto dst     = buffer_and_offset_from_ptr(d, destGpu);
    encoder->copyFromBuffer(src.buffer->buffer, src.offset, dst.buffer->buffer, dst.offset, size);
}

void cmd_copy_to_texture(CommandBuffer                  cmd,
                         GpuPtr                         srcGpu,
                         Handle<Texture>                texture,
                         const BufferToTextureCopyInfo& info) {
    auto d = cmd->device;

    auto encoder = get_compute_encoder(cmd);
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
}

void cmd_set_pipeline(CommandBuffer cmd, Handle<Pipeline> pipeline) {
    auto& p = cmd->device->m_pipeline_pool[pipeline];
    assert((is_in_render_pass(cmd) && p.render_pipeline)
           || (is_in_compute_pass(cmd) && p.compute_pipeline));
    if (is_in_render_pass(cmd)) {
        cmd->render_encoder->setRenderPipelineState(p.render_pipeline);
        // TODO: Cull mode
    } else {
        cmd->compute_encoder->setComputePipelineState(p.compute_pipeline);
    }
}

void cmd_set_depth_stencil_state(CommandBuffer cmd, Handle<DepthStencilState> state) {
    assert(is_in_render_pass(cmd));
    auto& d = cmd->device->m_depth_stencil_state_pool[state];
    cmd->render_encoder->setDepthStencilState(d.state);
}

void cmd_set_scissor_rect(CommandBuffer cmd, const Rect2D& rect) {
    assert(is_in_render_pass(cmd));

    cmd->render_encoder->setScissorRect(MTL::ScissorRect{
        .x      = rect.offset_x,
        .y      = rect.offset_y,
        .width  = rect.width,
        .height = rect.height,
    });
}

void cmd_dispatch(CommandBuffer cmd, GpuPtr dataGpu, const Dimension3D& gridDimensions) {}

void cmd_dispatch_indirect(CommandBuffer cmd, GpuPtr dataGpu, GpuPtr gridDimensionsGpu) {}

void cmd_begin_render_pass(CommandBuffer cmd, RenderPassDesc desc) {
    assert(!is_in_render_pass(cmd));
    if (is_in_compute_pass(cmd)) { end_compute_pass(cmd); }

    auto                        d    = cmd->device;
    MTL4::RenderPassDescriptor* pass = MTL4::RenderPassDescriptor::alloc()->init();

    uint32_t attachment_idx   = 0;
    auto     pass_attachments = pass->colorAttachments();
    for (const auto& c : desc.color_attachments) {
        MTL::RenderPassColorAttachmentDescriptor* attachment
            = MTL::RenderPassColorAttachmentDescriptor::alloc()->init();
        auto& tex = d->m_texture_pool[c.texture];
        attachment->setTexture(tex.texture);
        attachment->setLoadAction(bridge(c.load_op));
        attachment->setStoreAction(bridge(c.store_op));
        attachment->setClearColor(
            MTL::ClearColor(c.clear_color.r, c.clear_color.g, c.clear_color.b, c.clear_color.a));
        pass_attachments->setObject(attachment, attachment_idx);
        attachment_idx++;
    }

    if (desc.depth_attachment.texture) {
        MTL::RenderPassDepthAttachmentDescriptor* depth_desc
            = MTL::RenderPassDepthAttachmentDescriptor::alloc()->init();
        auto& depth_tex = d->m_texture_pool[desc.depth_attachment.texture];
        depth_desc->setTexture(depth_tex.texture);
        depth_desc->setLoadAction(bridge(desc.depth_attachment.load_op));
        depth_desc->setStoreAction(bridge(desc.depth_attachment.store_op));
        depth_desc->setClearDepth(desc.depth_attachment.clear_color.r);
        pass->setDepthAttachment(depth_desc);
        depth_desc->release();
    }

    cmd->command_buffer->renderCommandEncoder(pass);
    pass->release();
}

void cmd_end_render_pass(CommandBuffer cmd) {
    assert(is_in_render_pass(cmd));
    cmd->render_encoder->endEncoding();
    cmd->render_encoder->release();
    cmd->render_encoder = nullptr;
}

void cmd_draw(CommandBuffer cmd,
              GpuPtr        vertexDataGpu,
              GpuPtr        fragmentDataGpu,
              uint32_t      vertexCount,
              uint32_t      instanceCount) {
    assert(is_in_render_pass(cmd));


    cmd->render_encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, vertexCount, instanceCount);
}

void cmd_draw_indexed_instanced(CommandBuffer cmd, const DrawIndexedInstancedInfo& args) {}

void cmd_draw_indexed_instanced_indirect(CommandBuffer cmd, const DrawIndexedIndirectInfo& args) {}

void cmd_draw_indexed_instanced_indirect_multi(CommandBuffer                cmd,
                                               const MultiDrawIndirectInfo& args) {}

void cmd_finalize(CommandBuffer cmd) {
    assert(!is_in_render_pass(cmd));
    if (is_in_compute_pass(cmd)) { end_compute_pass(cmd); }
    cmd->command_buffer->endCommandBuffer();
}

}  // namespace loon::gpu
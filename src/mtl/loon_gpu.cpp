#include <gpu/loon_gpu.h>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "format_info.h"
#include "metal_compute_metadata.h"
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
template class Span<const ColorTarget>;
template class Span<const RenderAttachment>;
template class Span<const Format>;
template class Span<const PresentMode>;
template class Span<const CommandBuffer>;
template class Span<const SemaphoreInfo>;
template class Span<const TextureTransition>;
template class Span<const SpecializationConstant>;
template class Function<void>;

template <class T>
using id = NS::SharedPtr<T>;

template <class T, class... Args>
id<T> make_id(Args&&... args) {
    return NS::TransferPtr(T::alloc()->init(std::forward<Args>(args)...));
}

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
    id<MTL::Buffer> buffer;
    id<MTL::Heap>   heap;
};

struct Texture {
    id<MTL::Texture> texture;
    Format           format;
};

struct SamplerMapping {
    Sampler               sampler;
    id<MTL::SamplerState> state;
};

struct SamplerMappingCompare {
    constexpr bool operator()(const SamplerMapping& a, const SamplerMapping& b) {
        return a.sampler > b.sampler;
    }
};

struct TextureHeap {
    id<MTL::TextureViewPool> pool = nullptr;
    TwoLevelBitset           bitset;
    Vector<SamplerMapping>   sampler_lookup;
};

struct Pipeline {
    id<MTL::RenderPipelineState>  render_pipeline  = nullptr;
    id<MTL::ComputePipelineState> compute_pipeline = nullptr;
    Topology                      topology;
    ShaderMetadata                metadata;
};

struct DepthStencilState {
    id<MTL::DepthStencilState> state = nullptr;
};

struct Semaphore {
    id<MTL::SharedEvent> event = nullptr;
};

struct Surface {
    id<CA::MetalLayer>    metal_layer;
    id<CA::MetalDrawable> current_drawable = nullptr;
    Handle<Texture>       current_texture  = {};
    uint64_t              frame_idx        = 0;

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
    id<MTL4::CommandBuffer>         command_buffer = nullptr;
    id<MTL4::ArgumentTable>         argument_table = nullptr;
    Queue                           queue;
    CommandPool*                    pool;
    Device                          device;
    id<MTL4::ComputeCommandEncoder> compute_encoder = nullptr;
    id<MTL4::RenderCommandEncoder>  render_encoder  = nullptr;

    MTL::PrimitiveType current_topology;
    MTL::Size required_threadgroup_size;  // Required threadgroup size of the currently bound
                                          // compute pipeline.
};

struct CommandPool {
    id<MTL4::CommandAllocator>      allocator      = nullptr;
    id<MTL4::ArgumentTable>         argument_table = nullptr;
    SegmentArray<CommandBufferImpl> command_buffers;  // In theory
    uint64_t                        frame_idx       = 0;
    uint32_t                        buffer_free_idx = 0;
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

    id<MTL4::CommandQueue> command_queue = nullptr;
    CommandSuperpool       command_superpool;
    id<MTL::SharedEvent>   callback_event = nullptr;
    Vector<Event>          pending_events;
    uint64_t               timeline_value = 0;

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

    id<MTL::Device> m_device = nullptr;

    id<MTL4::Compiler>            m_compiler;
    id<MTL4::CompilerTaskOptions> m_options;

    id<MTL::ResidencySet> m_residency_set = nullptr;
    Surface               m_surface;

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

    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device) { return nullptr; }

    if (!device->supportsFamily(MTL::GPUFamilyMetal4)) { return nullptr; }

    NS::Error* error = nullptr;

    auto compiler_desc = make_id<MTL4::CompilerDescriptor>();
    auto compiler      = NS::TransferPtr(device->newCompiler(compiler_desc.get(), &error));
    auto options       = make_id<MTL4::CompilerTaskOptions>();

    // It seems like this is a valid cast from testing.
    auto surface_metal_layer
        = NS::RetainPtr(reinterpret_cast<CA::MetalLayer*>(desc.native_window_handle));

    auto residency_set_descriptor = make_id<MTL::ResidencySetDescriptor>();
    // NOTE: Not sure how much this matters, should dig in more
    residency_set_descriptor->setInitialCapacity(64);
    auto residency_set
        = NS::TransferPtr(device->newResidencySet(residency_set_descriptor.get(), nullptr));
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
                                             if (b->heap) {
                                                 d->m_residency_set->removeAllocation(b->heap.get());
                                             }
                                             b->~Buffer();
                                         }),
        .m_texture_pool             = SlotMap<Texture>(alloc,
                                           [d](Texture* t) {
                                               if (t->texture) {
                                                   d->m_residency_set->removeAllocation(t->texture.get());
                                               }
                                               t->~Texture();
                                           }),
        .m_texture_heap_pool        = SlotMap<TextureHeap>(alloc,
                                                    [](TextureHeap* t) {
                                                        t->~TextureHeap();
                                                    }),
        .m_pipeline_pool            = SlotMap<Pipeline>(alloc,
                                             [](Pipeline* p) {
                                                p->~Pipeline();
                                             }),
        .m_depth_stencil_state_pool = SlotMap<DepthStencilState>(alloc,
                                                                 [](DepthStencilState* s) {
                                                                     s->~DepthStencilState();
                                                                 }),
        .m_semaphore_pool           = SlotMap<Semaphore>(alloc,
                                               [](Semaphore* s) {
                                                s->~Semaphore();
                                               }),
        .m_ptr_map                  = Vector<GpuPtrMap>(alloc),
    };
}

void destroy_device(Device d) {
    device_wait_for_idle(d);
    unconfigure_surface(d);

    d->m_semaphore_pool.clear();
    d->m_depth_stencil_state_pool.clear();
    d->m_pipeline_pool.clear();
    d->m_texture_heap_pool.clear();
    d->m_texture_pool.clear();
    d->m_buffer_pool.clear();
    tls_free(d->m_tls_key);

    auto allocator = d->m_allocator;
    d->~DeviceImpl();
    allocator.free({.ptr = d, .len = sizeof(DeviceImpl)});
}

Backend device_backend() {
    return Backend::Metal;
}

void device_wait_for_idle(Device d) {
    // NOTE: If work is being submitted concurrently, this won't behave correctly - it technically
    // only waits until any currently submitted work is done.
    auto& q = d->m_queue;
    if (q.command_queue) {
        q.command_queue->signalEvent(q.callback_event.get(), ++q.timeline_value);
        q.callback_event->waitUntilSignaledValue(q.timeline_value, UINT64_MAX);
    }
}

// MARK: Surface functions
SurfaceCapabilities get_surface_capabilities(Device d) {
    return SurfaceCapabilities{
        .usages  = UsageFlags::ColorAttachment | UsageFlags::TransferDst | UsageFlags::Storage,
        .formats = Surface::kSwapchainFormats,
        .present_modes = Surface::kPresentModes,
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
    d->m_surface.current_drawable = nullptr;
}

SurfaceTextureInfo get_current_texture(Device d) {
    id<CA::MetalDrawable> drawable = NS::RetainPtr(d->m_surface.metal_layer->nextDrawable());


    if (!drawable) {
        return SurfaceTextureInfo{
            .status  = SurfaceStatus::Error,
            .texture = {0},
        };
    }
    d->m_queue.command_queue->wait(drawable.get());

    d->m_surface.current_drawable = drawable;
    d->m_surface.frame_idx++;
    id<MTL::Texture> tex = NS::RetainPtr(drawable->texture());

    Handle<Texture> handle       = d->m_texture_pool.emplace({
              .texture = tex,
    });
    d->m_surface.current_texture = handle;

    return SurfaceTextureInfo{
        .status  = SurfaceStatus::Success,
        .texture = handle,
    };
}

SurfaceStatus present(Device d, Queue queue) {
    queue->command_queue->signalDrawable(d->m_surface.current_drawable.get());
    d->m_surface.current_drawable->present();
    d->m_surface.current_drawable = nullptr;
    d->m_texture_pool.erase(d->m_surface.current_texture);
    return SurfaceStatus::Success;
}

// MARK: Buffers:

GpuPtr malloc(Device d, size_t bytes, Memory memory) {
    return malloc(d, bytes, 64, memory);
}

GpuPtr malloc(Device d, size_t bytes, size_t align, Memory memory) {
    id<MTL::HeapDescriptor> heap_info = make_id<MTL::HeapDescriptor>();
    heap_info->setType(MTL::HeapTypePlacement);
    heap_info->setStorageMode(memory == Memory::Gpu ? MTL::StorageModePrivate
                                                    : MTL::StorageModeShared);
    heap_info->setCpuCacheMode(memory == Memory::Default ? MTL::CPUCacheModeWriteCombined
                                                         : MTL::CPUCacheModeDefaultCache);
    heap_info->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    heap_info->setSize(bytes);

    id<MTL::Heap> heap = NS::TransferPtr(d->m_device->newHeap(heap_info.get()));

    const MTL::ResourceOptions resource_options
        = (memory == Memory::Gpu ? MTL::ResourceStorageModePrivate : MTL::ResourceStorageModeShared)
          | (memory == Memory::Default ? MTL::ResourceCPUCacheModeWriteCombined
                                       : MTL::ResourceCPUCacheModeDefaultCache);


    id<MTL::Buffer> buffer = NS::TransferPtr(heap->newBuffer(bytes, resource_options, 0));
    if (!buffer) { return 0; }

    auto handle = d->m_buffer_pool.emplace({
        .buffer = buffer,
        .heap   = heap,
    });
    d->m_residency_set->addAllocation(heap.get());
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
    id<MTL::TextureDescriptor> info = make_id<MTL::TextureDescriptor>();
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

    const auto size_align = d->m_device->heapTextureSizeAndAlign(info.get());

    return {.size = size_align.size, .align = size_align.align};
}

Handle<Texture> create_texture(Device d, const TextureDesc& desc, GpuPtr location) {
    id<MTL::TextureDescriptor> info = make_id<MTL::TextureDescriptor>();
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

    id<MTL::Texture> texture = nullptr;
    if (location != 0) {
        // Specify location, use the mtlheap from the buffer
        auto memory = buffer_and_offset_from_ptr(d, location);
        texture     = NS::TransferPtr(memory.buffer->heap->newTexture(info.get(), memory.offset));
    } else {
        texture = NS::TransferPtr(d->m_device->newTexture(info.get()));
        d->m_residency_set->addAllocation(texture.get());
        d->m_residency_set->commit();
    }

    auto handle = d->m_texture_pool.emplace({
        .texture = texture,
        .format  = desc.format,
    });
    return handle;
}

Handle<TextureHeap> create_texture_heap(Device d, const TextureHeapDesc& desc) {
    auto view_pool_descriptor = make_id<MTL::ResourceViewPoolDescriptor>();
    view_pool_descriptor->setResourceViewCount(desc.texture_count);

    auto texture_view_pool
        = NS::TransferPtr(d->m_device->newTextureViewPool(view_pool_descriptor.get(), nullptr));

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
    auto&                          heap      = d->m_texture_heap_pool[h];
    auto&                          tex       = d->m_texture_pool[desc.texture];
    id<MTL::TextureViewDescriptor> view_info = make_id<MTL::TextureViewDescriptor>();
    view_info->setLevelRange(NS::Range(desc.base_mip, desc.mip_count));
    view_info->setPixelFormat(bridge(desc.format));
    view_info->setSliceRange(NS::Range(desc.base_layer, desc.layer_count));
    view_info->setTextureType(tex.texture->textureType());
    const auto      index = heap.bitset.set_leading_zero();
    MTL::ResourceID texture_view
        = heap.pool->setTextureView(tex.texture.get(), view_info.get(), index);

    return texture_view._impl;
}

void remove_texture_view_from_heap(Device d, Handle<TextureHeap> h, TextureView view) {
    auto&      heap = d->m_texture_heap_pool[h];
    const auto base = heap.pool->baseResourceID()._impl;
    assert(view >= base && view < base + heap.pool->resourceViewCount());
    const auto index = view - heap.pool->baseResourceID()._impl;
    heap.bitset.clear_bit(index);
}

Sampler add_sampler_to_heap(Device d, Handle<TextureHeap> h, const SamplerDesc& sampler) {
    auto& heap = d->m_texture_heap_pool[h];

    id<MTL::SamplerDescriptor> desc = make_id<MTL::SamplerDescriptor>();
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
    desc->setSupportArgumentBuffers(true);
    auto sampler_state = NS::TransferPtr(d->m_device->newSamplerState(desc.get()));


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
    heap.sampler_lookup.erase(it, it + 1);
}

static id<NS::String> get_span_as_string(Span<const char> span) {
    id<NS::String> source_view
        = make_id<NS::String>((void*)span.data(), span.size(), NS::UTF8StringEncoding, false);
    id<NS::String> copy = make_id<NS::String>(source_view.get());
    return copy;
}

// MARK: Pipelines
static id<MTL::FunctionConstantValues> construct_constant_values(
    Span<const SpecializationConstant> constants) {
    auto result = make_id<MTL::FunctionConstantValues>();

    constexpr auto copy_bytes = []<class T>(char* dst, const T& x) {
        T val = x;
        memcpy(dst, &val, sizeof(T));
    };

    for (const auto& c : constants) {
        char          data[8] = {};
        MTL::DataType type    = MTL::DataTypeUChar;
        switch (c.type) {
            case SpecializationConstantType::UInt8:
                copy_bytes(data, static_cast<uint8_t>(c.int_val));
                type = MTL::DataTypeUChar;
                break;
            case SpecializationConstantType::UInt16:
                copy_bytes(data, static_cast<uint16_t>(c.int_val));
                type = MTL::DataTypeUShort;
                break;
            case SpecializationConstantType::UInt32:
                copy_bytes(data, static_cast<uint16_t>(c.int_val));
                type = MTL::DataTypeUInt;
                break;
            case SpecializationConstantType::Int8:
                copy_bytes(data, static_cast<uint16_t>(c.int_val));
                type = MTL::DataTypeChar;
                break;
            case SpecializationConstantType::Int16:
                copy_bytes(data, static_cast<uint16_t>(c.int_val));
                type = MTL::DataTypeShort;
                break;
            case SpecializationConstantType::Int32:
                copy_bytes(data, static_cast<uint16_t>(c.int_val));
                type = MTL::DataTypeInt;
                break;
            case SpecializationConstantType::Boolean:
                copy_bytes(data, c.bool_val);
                type = MTL::DataTypeBool;
                break;
            case SpecializationConstantType::Float32:
                copy_bytes(data, static_cast<float>(c.float_val));
                type = MTL::DataTypeFloat;
                break;
        }
        result->setConstantValue(data, type, c.constant_id);
    }

    return result;
}

Handle<Pipeline> create_compute_pipeline(Device                             d,
                                         ShaderSource                       compute,
                                         Span<const SpecializationConstant> constants) {
    id<NS::String> shader_source = get_span_as_string(compute.spirv.cast<const char>());
    id<NS::String> entry_point   = get_span_as_string(compute.entry_point);

    NS::Error*       error   = nullptr;
    auto             options = make_id<MTL::CompileOptions>();
    id<MTL::Library> lib
        = NS::TransferPtr(d->m_device->newLibrary(shader_source.get(), options.get(), &error));

    auto desc      = make_id<MTL4::ComputePipelineDescriptor>();
    auto func_desc = make_id<MTL4::LibraryFunctionDescriptor>();
    func_desc->setLibrary(lib.get());
    func_desc->setName(entry_point.get());

    if (constants.is_empty()) {
        desc->setComputeFunctionDescriptor(func_desc.get());
    } else {
        auto specialized_func_desc = make_id<MTL4::SpecializedFunctionDescriptor>();
        specialized_func_desc->setFunctionDescriptor(func_desc.get());
        auto function_constants = construct_constant_values(constants);
        specialized_func_desc->setConstantValues(function_constants.get());
        desc->setComputeFunctionDescriptor(specialized_func_desc.get());
    }

    id<MTL::ComputePipelineState> compute_pipeline = NS::TransferPtr(
        d->m_compiler->newComputePipelineState(desc.get(), d->m_options.get(), &error));

    const auto metadata = parse_metadata(*get_thread_local_arena(d),
                                         compute.spirv.cast<const char>(),
                                         compute.entry_point);

    return d->m_pipeline_pool.emplace({
        .render_pipeline  = nullptr,
        .compute_pipeline = compute_pipeline,
        .metadata         = metadata,
    });
}

Handle<Pipeline> create_graphics_pipeline(Device                             d,
                                          ShaderSource                       vertex,
                                          ShaderSource                       fragment,
                                          const RasterDesc&                  desc,
                                          Span<const SpecializationConstant> constants) {
    // TODO: Error handling/propagation
    id<NS::String> vert_source      = get_span_as_string(vertex.spirv.cast<const char>());
    id<NS::String> vert_entry_point = get_span_as_string(vertex.entry_point);
    id<NS::String> frag_source      = get_span_as_string(fragment.spirv.cast<const char>());
    id<NS::String> frag_entry_point = get_span_as_string(fragment.entry_point);
    NS::Error*     error            = nullptr;

    auto options = make_id<MTL::CompileOptions>();

    id<MTL::Library> vert_lib
        = NS::TransferPtr(d->m_device->newLibrary(vert_source.get(), options.get(), &error));

    if (error != nullptr) { printf("%s\n", error->localizedDescription()->utf8String()); }

    id<MTL::Library> frag_lib
        = NS::TransferPtr(d->m_device->newLibrary(frag_source.get(), options.get(), &error));

    if (error) { printf("%s\n", error->localizedDescription()->utf8String()); }

    auto vert_func_desc = make_id<MTL4::LibraryFunctionDescriptor>();
    vert_func_desc->setLibrary(vert_lib.get());
    vert_func_desc->setName(vert_entry_point.get());

    auto frag_func_desc = make_id<MTL4::LibraryFunctionDescriptor>();
    frag_func_desc->setLibrary(frag_lib.get());
    frag_func_desc->setName(frag_entry_point.get());

    auto pipeline_desc = make_id<MTL4::RenderPipelineDescriptor>();

    if (constants.is_empty()) {
        pipeline_desc->setVertexFunctionDescriptor(vert_func_desc.get());
        pipeline_desc->setFragmentFunctionDescriptor(frag_func_desc.get());
    } else {
        auto function_constants         = construct_constant_values(constants);
        auto specialized_vert_func_desc = make_id<MTL4::SpecializedFunctionDescriptor>();
        specialized_vert_func_desc->setFunctionDescriptor(vert_func_desc.get());
        specialized_vert_func_desc->setConstantValues(function_constants.get());
        pipeline_desc->setVertexFunctionDescriptor(specialized_vert_func_desc.get());

        auto specialized_frag_func_desc = make_id<MTL4::SpecializedFunctionDescriptor>();
        specialized_frag_func_desc->setFunctionDescriptor(frag_func_desc.get());
        specialized_frag_func_desc->setConstantValues(function_constants.get());
        pipeline_desc->setFragmentFunctionDescriptor(specialized_frag_func_desc.get());
    }
    pipeline_desc->setFragmentFunctionDescriptor(frag_func_desc.get());
    pipeline_desc->setInputPrimitiveTopology(bridge_topology_class(desc.topology));
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
        attachment->setSourceRGBBlendFactor(bridge(state.src_color_factor));
        attachment->setDestinationRGBBlendFactor(bridge(state.dst_color_factor));
        attachment->setSourceAlphaBlendFactor(bridge(state.src_alpha_factor));
        attachment->setDestinationAlphaBlendFactor(bridge(state.dst_alpha_factor));
    }

    id<MTL::RenderPipelineState> render_pipeline = NS::TransferPtr(
        d->m_compiler->newRenderPipelineState(pipeline_desc.get(), d->m_options.get(), &error));

    return d->m_pipeline_pool.emplace({
        .render_pipeline  = render_pipeline,
        .compute_pipeline = nullptr,
        .topology         = desc.topology,
    });
}

void free(Device d, Handle<Pipeline> pipeline) {
    d->m_pipeline_pool.erase(pipeline);
}

// MARK: State Objects

Handle<DepthStencilState> create_depth_stencil_state(Device d, const DepthStencilDesc& desc) {
    auto info = make_id<MTL::DepthStencilDescriptor>();
    info->setDepthCompareFunction(bridge(desc.depth_test));
    info->setDepthWriteEnabled((desc.depth_mode & DepthFlags::Write) == DepthFlags::Write);

    auto backface_stencil = make_id<MTL::StencilDescriptor>();
    backface_stencil->setStencilFailureOperation(bridge(desc.stencil_back.fail_op));
    backface_stencil->setDepthFailureOperation(bridge(desc.stencil_back.depth_fail_op));
    backface_stencil->setDepthStencilPassOperation(bridge(desc.stencil_back.pass_op));
    backface_stencil->setStencilCompareFunction(bridge(desc.stencil_back.test));
    backface_stencil->setReadMask(desc.stencil_read_mask);
    backface_stencil->setWriteMask(desc.stencil_write_mask);
    info->setBackFaceStencil(backface_stencil.get());

    auto frontface_stencil = make_id<MTL::StencilDescriptor>();
    frontface_stencil->setStencilFailureOperation(bridge(desc.stencil_front.fail_op));
    frontface_stencil->setDepthFailureOperation(bridge(desc.stencil_front.depth_fail_op));
    frontface_stencil->setDepthStencilPassOperation(bridge(desc.stencil_front.pass_op));
    frontface_stencil->setStencilCompareFunction(bridge(desc.stencil_front.test));
    frontface_stencil->setReadMask(desc.stencil_read_mask);
    frontface_stencil->setWriteMask(desc.stencil_write_mask);
    info->setFrontFaceStencil(frontface_stencil.get());
    auto state = NS::TransferPtr(d->m_device->newDepthStencilState(info.get()));
    return d->m_depth_stencil_state_pool.emplace({
        .state = state,
    });
}

void free_depth_stencil_state(Device d, Handle<DepthStencilState> state) {
    d->m_depth_stencil_state_pool.erase(state);
}

// MARK: Semaphores

Handle<Semaphore> create_semaphore(Device d, uint64_t initValue) {
    auto event = NS::TransferPtr(d->m_device->newSharedEvent());
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

        if (!pool->allocator) {
            auto argument_table_desc = make_id<MTL4::ArgumentTableDescriptor>();
            argument_table_desc->setMaxBufferBindCount(2);
            argument_table_desc->setInitializeBindings(false);
            argument_table_desc->setMaxSamplerStateBindCount(0);
            argument_table_desc->setMaxTextureBindCount(0);
            // Initialize the command pool here.
            *pool = CommandPool{
                .allocator      = NS::TransferPtr(queue->device->m_device->newCommandAllocator()),
                .argument_table = NS::TransferPtr(
                    queue->device->m_device->newArgumentTable(argument_table_desc.get(), nullptr)),
                .command_buffers = SegmentArray<CommandBufferImpl>(queue->device->m_allocator),
                .frame_idx       = 0,
                .buffer_free_idx = 0,
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
            .command_buffer = NS::TransferPtr(device->m_device->newCommandBuffer()),
            .argument_table = pool->argument_table,
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
    if (!d->m_queue.command_queue) {
        d->m_queue = {
            .command_queue  = NS::TransferPtr(d->m_device->newMTL4CommandQueue()),
            .callback_event = NS::TransferPtr(d->m_device->newSharedEvent()),
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
    if (buffer) {
        buffer->command_buffer->beginCommandBuffer(pool->allocator.get());
        buffer->compute_encoder = nullptr;
        buffer->render_encoder  = nullptr;
        buffer->command_buffer->useResidencySet(d->m_residency_set.get());
    }
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
        id<MTL::SharedEvent> event = d->m_semaphore_pool[s.semaphore].event;
        q->command_queue->wait(event.get(), s.value);
    }


    Span<MTL4::CommandBuffer*> commands;
    for (auto cmd : command_buffers) {
        MTL4::CommandBuffer* buf = cmd->command_buffer.get();
        commands                 = concat(&arena, commands, buf);
    }
    q->command_queue->commit(commands.data(), commands.size());

    for (auto s : signal_semaphores) {
        // NOTE: Metal doesn't support signaling after a specific stage.
        id<MTL::SharedEvent> event = d->m_semaphore_pool[s.semaphore].event;
        q->command_queue->signalEvent(event.get(), s.value);
    }
}

void queue_cancel(Queue q, Span<const Handle<CommandBuffer>> command_buffers) {}

void queue_on_submitted_work_completed(Queue q, Function<void>&& fn) {
    q->pending_events.emplace_back(
        QueueImpl::Event{.completed_time = q->timeline_value, .callback = std::move(fn)});
    q->command_queue->signalEvent(q->callback_event.get(), q->timeline_value++);
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
    return (bool)cmd->compute_encoder;
}

static bool is_in_render_pass(CommandBuffer cmd) {
    return (bool)cmd->render_encoder;
}

static id<MTL4::ComputeCommandEncoder> get_compute_encoder(CommandBuffer cmd) {
    assert(!is_in_render_pass(cmd));
    if (!cmd->compute_encoder) {
        cmd->compute_encoder = NS::RetainPtr(cmd->command_buffer->computeCommandEncoder());
    }

    return cmd->compute_encoder;
}

static void end_compute_pass(CommandBuffer cmd) {
    assert(is_in_compute_pass(cmd));
    cmd->compute_encoder->endEncoding();
    cmd->compute_encoder = nullptr;
}

void cmd_memcpy(CommandBuffer cmd, GpuPtr destGpu, GpuPtr srcGpu, size_t size) {
    auto d = cmd->device;

    auto encoder = get_compute_encoder(cmd);
    auto src     = buffer_and_offset_from_ptr(d, srcGpu);
    auto dst     = buffer_and_offset_from_ptr(d, destGpu);
    encoder->copyFromBuffer(src.buffer->buffer.get(),
                            src.offset,
                            dst.buffer->buffer.get(),
                            dst.offset,
                            size);
}

void cmd_copy_to_texture(CommandBuffer                  cmd,
                         GpuPtr                         srcGpu,
                         Handle<Texture>                texture,
                         const BufferToTextureCopyInfo& info) {
    auto d = cmd->device;

    auto             encoder     = get_compute_encoder(cmd);
    auto             src         = buffer_and_offset_from_ptr(d, srcGpu);
    const auto&      t           = d->m_texture_pool[texture];
    const FormatInfo format_info = get_format_info(t.format);
    const uint64_t   pixels_per_row
        = info.source_row_pixels_stride == 0 ? info.image_extent.x : info.source_row_pixels_stride;
    const uint64_t rows_per_image
        = info.source_plane_rows_stride == 0 ? info.image_extent.y : info.source_plane_rows_stride;
    encoder->copyFromBuffer(
        src.buffer->buffer.get(),
        src.offset,
        pixels_per_row * format_info.block_size_bytes,
        rows_per_image * format_info.block_size_bytes * pixels_per_row,
        MTL::Size::Make(info.image_extent.x, info.image_extent.y, info.image_extent.z),
        t.texture.get(),
        info.base_layer,
        info.base_mip,
        MTL::Origin::Make(info.destination_image_offset.x,
                          info.destination_image_offset.y,
                          info.destination_image_offset.z));
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
    (void)image_transitions;  // Only thing we could maybe use image transitions for is waiting
                              // for/signalling the drawable, but we're currently doing that when
                              // getting the drawable and presenting.

    assert(!is_in_render_pass(
        cmd));  // To match vulkan behaviour, don't allow barriers in a render pass.
    // If we're in a compute pass, can encode this

    // TODO: Avoid creating a compute encoder if we're not in one, for more efficient barriers.
    // TODO: Ensure visibility options are correct here.

    if (before == StageFlags::None || after == StageFlags::None) {
        // In vulkan this is a valid image transition, but it's meaningless here. Skip it.
        return;
    } else if (before == StageFlags::RasterColorOut || after == StageFlags::RasterColorOut) {
        // TODO: Check correctness of this in render-to-texture cases.
        // These barriers should be covered by waiting for drawables? uncertain here, maybe needed
        // for deferred rendering but I think there's implicit render pass synchronization.
        return;
    }

    // This isn't the optimal option, but for now this should work:
    auto encoder = get_compute_encoder(cmd);

    encoder->barrierAfterStages(bridge(before), bridge(after), MTL4::VisibilityOptionDevice);

    end_compute_pass(cmd);
}

void cmd_set_pipeline(CommandBuffer cmd, Handle<Pipeline> pipeline) {
    auto& p = cmd->device->m_pipeline_pool[pipeline];
    assert((is_in_render_pass(cmd) && p.render_pipeline) || p.compute_pipeline);
    if (is_in_render_pass(cmd)) {
        cmd->render_encoder->setRenderPipelineState(p.render_pipeline.get());
        cmd->current_topology = bridge(p.topology);
    } else {
        auto compute_encoder = get_compute_encoder(cmd);
        compute_encoder->setComputePipelineState(p.compute_pipeline.get());
        cmd->required_threadgroup_size = MTL::Size::Make(p.metadata.required_threadgroup_size.x,
                                                         p.metadata.required_threadgroup_size.y,
                                                         p.metadata.required_threadgroup_size.z);
    }
}

void cmd_set_depth_stencil_state(CommandBuffer cmd, Handle<DepthStencilState> state) {
    assert(is_in_render_pass(cmd));
    auto& d = cmd->device->m_depth_stencil_state_pool[state];
    cmd->render_encoder->setDepthStencilState(d.state.get());
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

static void set_compute_ptrs(CommandBuffer cmd, GpuPtr computeDataGpu) {
    cmd->argument_table->setAddress(computeDataGpu, 0);
    cmd->compute_encoder->setArgumentTable(cmd->argument_table.get());
}

void cmd_dispatch(CommandBuffer cmd, GpuPtr dataGpu, const Dimension3D& gridDimensions) {
    assert(!is_in_render_pass(cmd));

    auto encoder = get_compute_encoder(cmd);
    set_compute_ptrs(cmd, dataGpu);
    encoder->dispatchThreadgroups(
        MTL::Size::Make(gridDimensions.x, gridDimensions.y, gridDimensions.z),
        cmd->required_threadgroup_size);
}

void cmd_dispatch_indirect(CommandBuffer cmd, GpuPtr dataGpu, GpuPtr gridDimensionsGpu) {
    assert(!is_in_render_pass(cmd));

    auto encoder = get_compute_encoder(cmd);
    set_compute_ptrs(cmd, dataGpu);
    encoder->dispatchThreadgroups(gridDimensionsGpu, cmd->required_threadgroup_size);
}

void cmd_begin_render_pass(CommandBuffer cmd, RenderPassDesc desc) {
    assert(!is_in_render_pass(cmd));
    if (is_in_compute_pass(cmd)) { end_compute_pass(cmd); }

    auto                           d    = cmd->device;
    id<MTL4::RenderPassDescriptor> pass = make_id<MTL4::RenderPassDescriptor>();

    uint32_t attachment_idx   = 0;
    auto     pass_attachments = pass->colorAttachments();
    for (const auto& c : desc.color_attachments) {
        id<MTL::RenderPassColorAttachmentDescriptor> attachment
            = make_id<MTL::RenderPassColorAttachmentDescriptor>();
        auto& tex = d->m_texture_pool[c.texture];
        attachment->setTexture(tex.texture.get());
        attachment->setLoadAction(bridge(c.load_op));
        attachment->setStoreAction(bridge(c.store_op));
        attachment->setClearColor(
            MTL::ClearColor(c.clear_color.r, c.clear_color.g, c.clear_color.b, c.clear_color.a));
        pass_attachments->setObject(attachment.get(), attachment_idx);
        attachment_idx++;
    }

    if (desc.depth_attachment.texture) {
        id<MTL::RenderPassDepthAttachmentDescriptor> depth_desc
            = make_id<MTL::RenderPassDepthAttachmentDescriptor>();
        auto& depth_tex = d->m_texture_pool[desc.depth_attachment.texture];
        depth_desc->setTexture(depth_tex.texture.get());
        depth_desc->setLoadAction(bridge(desc.depth_attachment.load_op));
        depth_desc->setStoreAction(bridge(desc.depth_attachment.store_op));
        depth_desc->setClearDepth(desc.depth_attachment.clear_color.r);
        pass->setDepthAttachment(depth_desc.get());
    }

    cmd->render_encoder = NS::RetainPtr(cmd->command_buffer->renderCommandEncoder(pass.get()));

    cmd->render_encoder->setViewport(MTL::Viewport{
        .originX = 0,
        .originY = 0,
        .width   = (float)desc.render_area.width,
        .height  = (float)desc.render_area.height,
        .znear   = 0,
        .zfar    = 1,
    });
}

void cmd_end_render_pass(CommandBuffer cmd) {
    assert(is_in_render_pass(cmd));
    cmd->render_encoder->endEncoding();
    cmd->render_encoder = nullptr;
}

void cmd_set_front_face(CommandBuffer cmd, FrontFace front) {
    assert(is_in_render_pass(cmd));
    cmd->render_encoder->setFrontFacingWinding(
        front == FrontFace::CCW ? MTL::WindingCounterClockwise : MTL::WindingClockwise);
}

void cmd_set_cull_mode(CommandBuffer cmd, Cull cull) {
    assert(is_in_render_pass(cmd));
    cmd->render_encoder->setCullMode(bridge(cull));
}

static void set_graphics_ptrs(CommandBuffer cmd, GpuPtr vertexDataGpu, GpuPtr fragmentDataGpu) {
    cmd->argument_table->setAddress(vertexDataGpu, 0);
    cmd->argument_table->setAddress(fragmentDataGpu, 1);
    cmd->render_encoder->setArgumentTable(cmd->argument_table.get(),
                                          MTL::RenderStageVertex | MTL::RenderStageFragment);
}

void cmd_draw(CommandBuffer cmd,
              GpuPtr        vertexDataGpu,
              GpuPtr        fragmentDataGpu,
              uint32_t      vertexCount,
              uint32_t      instanceCount) {
    assert(is_in_render_pass(cmd));

    set_graphics_ptrs(cmd, vertexDataGpu, fragmentDataGpu);
    cmd->render_encoder->drawPrimitives(cmd->current_topology, 0, vertexCount, instanceCount);
}

void cmd_draw_indexed_instanced(CommandBuffer cmd, const DrawIndexedInstancedInfo& args) {
    assert(is_in_render_pass(cmd));

    set_graphics_ptrs(cmd, args.vertexDataGpu, args.fragmentDataGpu);

    const uint32_t index_buffer_size
        = args.indexCount * (args.type == IndexType::UInt16 ? sizeof(uint16_t) : sizeof(uint32_t));
    cmd->render_encoder->drawIndexedPrimitives(
        cmd->current_topology,
        args.indexCount,
        args.type == IndexType::UInt16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32,
        args.indicesGpu,
        index_buffer_size,
        args.instanceCount);
}

void cmd_draw_indexed_instanced_indirect(CommandBuffer cmd, const DrawIndexedIndirectInfo& args) {
    assert(is_in_render_pass(cmd));

    set_graphics_ptrs(cmd, args.vertexDataGpu, args.fragmentDataGpu);
    auto index_info = buffer_and_offset_from_ptr(cmd->device, args.indicesGpu);

    cmd->render_encoder->drawIndexedPrimitives(
        cmd->current_topology,
        args.type == IndexType::UInt16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32,
        args.indicesGpu,
        index_info.buffer->buffer->length() - index_info.offset,
        args.argsGpu);
}

void cmd_draw_indexed_instanced_indirect_multi(CommandBuffer                cmd,
                                               const MultiDrawIndirectInfo& args) {
    // Multidraw not supported on metal.
    assert(false);
}

void cmd_finalize(CommandBuffer cmd) {
    assert(!is_in_render_pass(cmd));
    if (is_in_compute_pass(cmd)) { end_compute_pass(cmd); }
    cmd->command_buffer->endCommandBuffer();

    release_command_pool(cmd->queue, cmd->pool);
}

}  // namespace loon::gpu

#include <gpu/loon_gpu.h>

#include <cassert>
#include <cstddef>
#include <cstring>

#include "containers.h"

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal.hpp>


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
    Handle<Buffer> malloc(size_t bytes, Memory memory = Memory::Default);
    Handle<Buffer> malloc(size_t bytes, size_t align, Memory memory = Memory::Default);
    void           free(Handle<Buffer> buffer);
    GpuPtr         get_device_pointer(Handle<Buffer> buffer);
    void*          get_host_pointer(Handle<Buffer> buffer);


    // Textures:
    Handle<Texture>     create_texture(const TextureDesc& desc);
    Handle<TextureHeap> create_texture_heap(size_t size);

    uint32_t add_texture_view_to_heap(Handle<TextureHeap>, const TextureViewDesc& desc);
    void     remove_texture_view_from_heap(Handle<TextureHeap>, uint32_t);

    void free(Handle<Texture>);
    void free(Handle<TextureHeap>);

    // Pipelines
    Handle<Pipeline> create_compute_pipeline(ShaderSource computeIR);
    Handle<Pipeline> create_graphics_pipeline(ShaderSource vertex,
                                              ShaderSource fragment,
                                              RasterDesc   desc);
    Handle<Pipeline> create_graphics_meshlet_pipeline(ShaderSource meshletIR,
                                                      ShaderSource pixelIR,
                                                      RasterDesc   desc);
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

   private:
    int64_t            m_refcount = 1;
    Allocator          m_allocator;
    ProcLogCallback    m_log_callback = nullptr;
    void*              m_log_userdata = nullptr;
    LogLevel           m_log_level    = LogLevel::Off;
    loon::gpu::tls_key m_tls_key;

    MTL::Device* m_device = nullptr;
};


Device::Impl::Impl(const DeviceDesc& desc) :
    m_allocator{desc.alloc_callback ? Allocator(desc.alloc_callback, desc.alloc_userdata)
                                    : Allocator()},
    m_log_callback(desc.log_callback),
    m_log_userdata(desc.log_userdata),
    m_log_level(desc.log_level),
    m_tls_key{loon::gpu::tls_alloc([](void* data) {
        auto state = reinterpret_cast<ThreadLocalState*>(data);
        state->~ThreadLocalState();
    })} {}

Device::Impl::~Impl() {
    m_device->release();
}

bool Device::Impl::initialize(const DeviceDesc& desc) {
    m_device = MTL::CreateSystemDefaultDevice();
    if (!m_device) { return false; }

    return true;
}

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

Handle<Texture> Device::create_texture(const TextureDesc& desc) {
    return impl->create_texture(desc);
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

Handle<DepthStencilState> Device::create_depth_stencil_state(DepthStencilDesc desc) {
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
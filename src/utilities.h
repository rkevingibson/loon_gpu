#pragma once

#include <cassert>
#include <mutex>

#include "containers.h"
#include "platform_utils.h"
#include "volk.h"
#include "webgpu/webgpu.h"

namespace webgpu {

// MARK: ReferenceCount
struct ReferenceCount {
    void               add() { webgpu::atomic_fetch_add(&reference_count, 1); }
    [[nodiscard]] bool release() {
        const auto old_count = webgpu::atomic_fetch_add(&reference_count, -1);
        assert((old_count) != 0);
        return (old_count - 1) == 0;
    }
    void    release_all() { webgpu::atomic_exchange(&reference_count, 0); }
    int64_t count() { return webgpu::atomic_load(&reference_count); };

   private:
    // TODO: Should probably start this at 1, and ensure that I can't add/release once it goes to 0
    int64_t reference_count{0};
};

template <typename T>
T* return_with_ownership(T* ptr) {
    ptr->add_ref();
    return ptr;
}

// MARK: Label
class Label {
   public:
    Label() = default;
    Label(const Allocator& backup_alloc, WGPUStringView label) : Label() {
        set(backup_alloc, label);
    }
    WGPUStringView get() const;
    void           set(const Allocator& backup_alloc, WGPUStringView label);

   private:
    static constexpr uint32_t label_buffer_size = 256;
    uint32_t                  m_capacity        = label_buffer_size;
    uint32_t                  m_len             = 0;
    union {
        char* m_label = nullptr;
        char  m_inline_buffer[label_buffer_size];
    };
};

// MARK: UsageScope

enum ResourceUsage : uint16_t {
    kUsageUndefined      = 0,
    kUsageInput          = 1 << 0,
    kUsageConstant       = 1 << 1,
    kUsageStorage        = 1 << 2,
    kUsageStorageRead    = 1 << 3,
    kUsageAttachment     = 1 << 4,
    kUsageAttachmentRead = 1 << 5,
    kUsagePresent        = 1 << 6,
    kUsageTransferSrc    = 1 << 7,
    kUsageTransferDst    = 1 << 8,
};

class UsageScope {
   public:
    UsageScope() = default;
    explicit UsageScope(Allocator allocator);
    void reset();

    void add(WGPUTextureView tex, ResourceUsage usage, WGPUShaderStage stage);

    void add(WGPUBuffer buffer, ResourceUsage usage, WGPUShaderStage stage);
    void try_add(WGPUBuffer buffer, ResourceUsage usage, WGPUShaderStage stage);

    void update_resource_last_usages();

    void update_first_last_usages(UsageScope& first_usages, UsageScope& last_usages) const;

    struct TextureUsage {
        ResourceUsage     usage           = ResourceUsage::kUsageUndefined;
        WGPUShaderStage   stage           = WGPUShaderStage_None;
        uint32_t          min_mip_level   = 0;
        uint32_t          max_mip_level   = 0;
        uint32_t          min_array_layer = 0;
        uint32_t          max_array_layer = 0;
        WGPUTextureAspect aspect          = WGPUTextureAspect_Undefined;
    };

    struct BufferUsage {
        ResourceUsage   usage = ResourceUsage::kUsageUndefined;
        WGPUShaderStage stage = WGPUShaderStage_None;
    };

    HashTable<WGPUTexture, TextureUsage> tex_usages;
    HashTable<WGPUBuffer, BufferUsage>   buffer_usages;
};


// MARK: Arena

class Arena {
   public:
    Arena() = default;
    Arena(void* ptr, size_t size) noexcept :
        m_ptr(reinterpret_cast<uintptr_t>(ptr)), m_begin(m_ptr), m_size(size) {};

    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;

    [[nodiscard]] void* alloc(size_t size) {
        const uintptr_t ptr    = m_ptr;
        const uintptr_t newptr = ptr + size;
        const uintptr_t end    = m_begin + m_size;
        if (newptr > end) { return nullptr; }
        m_ptr = newptr;
        return reinterpret_cast<void*>(ptr);
    }

    void free(const void* ptr, size_t size) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        if (p + size == m_ptr && p >= m_begin) { m_ptr = p; }
    }

    bool owns(const void* ptr) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        return p >= m_begin && p < m_begin + m_size;
    }

   private:
    uintptr_t m_ptr{0};
    uintptr_t m_begin{0};
    size_t    m_size{0};
};

// MARK: ArenaVector

// A simple adapter for creating a stack-like interface on top of an arena.
// Note: This takes advantage of the fact that Arena will have subsequent allocations located
// one after another. As such, only one ArenaVector can be used at a time for an arena - ie.
// push calls should not be interleaved.
// TODO: Get rid of this, replace with some functions on Arena, just to be more explicit.
template <class T>
class ArenaVector {
   public:
    ArenaVector(Arena* arena) :
        m_data(reinterpret_cast<T*>(arena->alloc(0))), m_size(0), m_arena(arena) {}
    ~ArenaVector() { m_arena->free(m_data, m_size * sizeof(T)); }

    bool push(const T& val) {
        // TODO: Worry about alignment?
        auto ptr = m_arena->alloc(sizeof(T));
        if (!ptr) { return false; }
        memcpy(ptr, &val, sizeof(T));
        ++m_size;
        return true;
    }

    T*       data() { return m_data; }
    const T* data() const { return m_data; }
    const T* begin() const { return m_data; }
    const T* end() const { return m_data + m_size; }
    uint32_t size() const { return m_size; }

   private:
    T*       m_data = nullptr;
    uint32_t m_size{0};
    Arena*   m_arena;
};

// MARK: ScopeGuard - a tiny RAII wrapper for deferring work to scope exit.
template <typename T>
class ScopeGuard {
   public:
    ScopeGuard(T&& fn) noexcept : m_fn(fn) {};
    ScopeGuard(ScopeGuard&& other) noexcept {}
    ScopeGuard(const ScopeGuard&)            = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard& operator=(ScopeGuard&&)      = delete;
    ~ScopeGuard() { m_fn(); }

   private:
    T m_fn;
};
template <class T>
ScopeGuard(T) -> ScopeGuard<T>;

// MARK: Callbacks

enum class CallbackType {
    BufferMap,
    CompilationInfo,
    CreateComputePipeline,
    CreateRenderPipeline,
    DeviceLost,
    PopErrorScope,
    QueueWorkDone,
    RequestAdapter,
    RequestDevice,
};

struct CallbackData {
    WGPUProc         callback;
    WGPUCallbackMode mode;
    void*            userdata1;
    void*            userdata2;
    WGPUStringView   message;
    CallbackType     type;
    union {
        struct {
            WGPUMapAsyncStatus status;
        } buffer_map;
        struct {
            WGPUCompilationInfoRequestStatus status;
            WGPUCompilationInfo              info;
        } compilation_info;
        struct {
            WGPUCreatePipelineAsyncStatus status;
            WGPUComputePipeline           pipeline;
        } create_compute_pipeline;
        struct {
            WGPUCreatePipelineAsyncStatus status;
            WGPURenderPipeline            pipeline;
        } create_render_pipeline;
        struct {
            WGPUDevice const*    device;
            WGPUDeviceLostReason reason;
        } device_lost;
        struct {
            WGPUPopErrorScopeStatus status;
            WGPUErrorType           type;
        } pop_error_scope;
        struct {
            WGPUQueueWorkDoneStatus status;
        } queue_work_done;
        struct {
            WGPURequestAdapterStatus status;
            WGPUAdapter              adapter;
        } request_adapter;
        struct {
            WGPURequestDeviceStatus status;
            WGPUDevice              device;
        } request_device;
    };
    bool ready = false;
};

// MARK: CommandPool

// A caching pool for command buffers
//
class CommandPool {
   public:
    CommandPool(WGPUDevice device, Arena* arena);
    ~CommandPool();
    WGPUCommandBuffer allocate_command_buffer(int64_t current_timeline_value);
    void              free_command_buffer(WGPUCommandBuffer, int64_t timeline_value);

   private:
    bool grow();

    std::mutex    mutex;
    WGPUDevice    device;
    Arena*        arena;
    VkCommandPool pool;

    struct RingBufferEntry {
        WGPUCommandBuffer buffer         = VK_NULL_HANDLE;
        int64_t           timeline_value = -1;
        bool              is_reset() const { return timeline_value < 0; }
    };
    RingBufferEntry* ring_buffer = nullptr;
    uint32_t         read_idx    = 0;
    uint32_t         write_idx   = 0;
    uint32_t         capacity    = 0;
};

// MARK: StagingBuffer

class StagingBuffer {
   public:
    StagingBuffer() = default;
    StagingBuffer(WGPUDevice device, size_t size);

    void* get_mapped_range(size_t offset, size_t size);
    void  unmap(WGPUBuffer target, size_t offset, size_t size);

   private:
    WGPUBuffer buffer = nullptr;
};

// MARK: DescriptorSetAllocator
class DescriptorSetAllocator {
    struct Pool;

   public:
    DescriptorSetAllocator() = default;
    DescriptorSetAllocator(WGPUDevice device);

    struct DescriptorAllocation {
        Pool*           pool = nullptr;
        VkDescriptorSet set  = VK_NULL_HANDLE;
        void            free(WGPUDevice device);
    };

    DescriptorAllocation alloc(WGPUBindGroupLayout layout);

   private:
    struct Pool {
        VkDescriptorPool vk_pool;
        uint32_t         sets_remaining;
    };
    // Had to go for the punny name
    struct PoolQueue {
        webgpu::SegmentArray<Pool> pools;
    };
    WGPUDevice                             m_device;
    webgpu::HashTable<uint32_t, PoolQueue> m_pools;
};

};  // namespace webgpu
#pragma once
#include <cstddef>
#include <cstdint>
#include <iterator>

#ifdef __cpp_concepts
#    define REQUIRES(x) requires x
#else
#    define REQUIRES(x)
#endif


namespace loon::gpu {

constexpr size_t kMaxColorAttachments = 16;
constexpr size_t kMaxNumBuffers       = 32ull * 1024;
constexpr size_t kMaxNumTextures      = 64ull * 1024;
constexpr size_t kMaxTextureHeapSize  = 32ull * 1024;

template <class T>
class Span {
    template <class U>
    static constexpr bool is_const = false;
    template <class U>
    static constexpr bool is_const<const U> = true;

    template <class U, class V>
    static constexpr bool is_const_of = false;
    template <class U>
    static constexpr bool is_const_of<U, const U> = true;

   public:
    // Construct an empty Span
    constexpr Span() noexcept = default;
    constexpr Span(T* ptr, size_t len) noexcept : m_ptr{ptr}, m_len{len} {}
    constexpr Span(T* begin, T* end) noexcept :
        m_ptr{begin}, m_len{static_cast<size_t>(end - begin)} {}

    // Construct from fixed size array
    template <size_t N>
    constexpr Span(T (&a)[N]) noexcept : Span(a, N) {}

    // Construct from an initializer list - only acceptable in the case where a Span is being
    // passed as a function argument, otherwise the initializer list will not outlive the Span, and
    // you end up with a dangling pointer.
    // e.g. for a function void Process(rkg::span<const float> x);
    // We can do Process({1, 2, 3});
    // But not:
    // Span args = {1,2,3}; // Dangling pointer after this line.
    // Process(args);
    constexpr Span(std::initializer_list<T> v) noexcept REQUIRES(is_const<T>) :
        Span(v.begin(), v.size()) {}

    // Construct from a single value - same rules as for initializer list, useful for passing one
    // element to a function which takes a span
    constexpr Span(const T& v) noexcept REQUIRES(is_const<T>) : Span(&v, 1) {}

    template <typename U>
    constexpr Span(const Span<U>& src) noexcept REQUIRES((is_const_of<U, T>)) :
        Span(src.data(), src.size()) {}

    // Accessors

    constexpr T* data() const noexcept { return m_ptr; }

    constexpr size_t size() const noexcept { return m_len; }

    constexpr bool is_empty() const noexcept { return m_len == 0; }

    // TODO: Add asserts for bounds checks in the following:
    constexpr T&       operator[](size_t i) const noexcept { return m_ptr[i]; }
    constexpr T&       front() const noexcept { return *m_ptr; }
    constexpr T&       back() const noexcept { return *(m_ptr + m_len - 1); }
    constexpr T*       begin() const noexcept { return m_ptr; }
    constexpr T*       end() const noexcept { return m_ptr + m_len; }
    constexpr const T* cbegin() const noexcept { return begin(); }
    constexpr const T* cend() const noexcept { return end(); }

    constexpr std::reverse_iterator<T*> rbegin() const noexcept {
        return std::make_reverse_iterator(end());
    }
    constexpr std::reverse_iterator<T*> rend() const noexcept {
        return std::make_reverse_iterator(begin());
    }

    Span<unsigned char> as_bytes() {
        return Span<unsigned char>((unsigned char*)m_ptr, m_len * sizeof(T));
    }

    template <typename U>
    Span<U> cast() const {
        return Span<U>((U*)m_ptr, m_len * sizeof(T) / sizeof(U));
    }

    // Modifiers:

    // Remove the first `n` elements from the Span
    constexpr Span<T> remove_prefix(size_t n) const noexcept { return Span(m_ptr + n, m_len - n); }

    // Remove the last `n` elements from the Span
    constexpr Span<T> remove_suffix(size_t n) const noexcept { return Span(m_ptr, m_len - n); }

    constexpr bool contains(const T* needle) const noexcept {
        return begin() <= needle && needle < end();
    }

    constexpr bool contains(const Span<T>& other) const noexcept {
        return begin() <= other.begin() && other.end() <= end();
    }

    constexpr bool starts_with(const Span<T>& other) const noexcept
    // requires is_byte_type<T>
    {
        if (other.size() > m_len) return false;
        for (size_t i = 0; i < other.size(); ++i) {
            if (m_ptr[i] != other[i]) { return false; }
        }
        return true;
    }

    constexpr bool ends_with(const Span<T>& other) const noexcept
    // requires is_byte_type<T>
    {
        if (other.size() > m_len) return false;

        for (size_t i = 0, offset = m_len - other.size(); i < other.size(); ++i) {
            if (m_ptr[offset + i] != other[i]) { return false; }
        }
        return true;
    }

    constexpr Span<T> trim_whitespace() const noexcept
    // requires is_byte_type<T>
    {
        auto           ptr = m_ptr;
        auto           len = m_len;
        constexpr auto is_whitespace
            = [](T c) { return c == '\0' || c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
        // Trim front
        while (len != 0 && is_whitespace(*ptr)) {
            len--;
            ptr++;
        }

        // Trim back
        T* tail = end() - 1;
        while (len != 0 && is_whitespace(*tail)) {
            tail--;
            len--;
        }
        return Span<T>(ptr, len);
    }

    friend inline bool operator==(Span<const T> lhs, Span<const T> rhs)
    // requires is_byte_type<T>
    {
        return lhs.size() == rhs.size() && memcmp(lhs.data(), rhs.data(), lhs.size()) == 0;
    }

   protected:
    T*     m_ptr{nullptr};
    size_t m_len{0};
};

using ByteSpan = Span<uint8_t>;

inline constexpr Span<const char> operator""_sv(const char* val, size_t len) {
    return Span<const char>(val, len);
}

#define WGPU_STRCAT_IMPL(a, b) a##b
#define WGPU_STRCAT(a, b)      WGPU_STRCAT_IMPL(a, b)
#define WGPU_FN_TYPEDEF(name)  WGPU_STRCAT(WGPUProc, name)

typedef struct MemoryBlock {
    void*    ptr;
    uint32_t len;
} MemoryBlock;

// Opaque handles

template <class T>
struct Handle {
    uint64_t h;
};

struct Device;
struct Pipeline;
struct Buffer;
struct Texture;
struct TextureHeap;
struct DepthStencilState;
struct BlendState;
struct Queue;
struct CommandBuffer;
struct Semaphore;

using GpuPtr = uint64_t;

// Enums
typedef enum LogLevel {
    LogLevel_Off     = 0x00000000,
    LogLevel_Error   = 0x00000001,
    LogLevel_Warning = 0x00000002,
    LogLevel_Info    = 0x00000003,
    LogLevel_Debug   = 0x00000004,
    LogLevel_Force32 = 0x7FFFFFFF,
} LogLevel;

typedef enum GpuPreference {
    GpuPreference_Discrete = 0,
    GpuPreference_Integrated,
} GpuPreference;

enum MEMORY { MEMORY_DEFAULT, MEMORY_GPU, MEMORY_READBACK };
enum CULL { CULL_CCW, CULL_CW, CULL_ALL, CULL_NONE };
enum DEPTH_FLAGS { DEPTH_NONE = 0, DEPTH_READ = 0x1, DEPTH_WRITE = 0x2 };
enum OP {
    OP_NEVER,
    OP_LESS,
    OP_EQUAL,
    OP_LESS_EQUAL,
    OP_GREATER,
    OP_NOT_EQUAL,
    OP_GREATER_EQUAL,
    OP_ALWAYS,
    OP_KEEP,
};
enum BLEND { BLEND_ADD, BLEND_SUBTRACT, BLEND_REV_SUBTRACT, BLEND_MIN, BLEND_MAX };
enum FACTOR {
    FACTOR_ZERO,
    FACTOR_ONE,
    FACTOR_SRC_COLOR,
    FACTOR_DST_COLOR,
    FACTOR_SRC_ALPHA,
};

enum TOPOLOGY { TOPOLOGY_TRIANGLE_LIST, TOPOLOGY_TRIANGLE_STRIP, TOPOLOGY_TRIANGLE_FAN };

enum TEXTURE {
    TEXTURE_1D,
    TEXTURE_2D,
    TEXTURE_3D,
    TEXTURE_CUBE,
    TEXTURE_2D_ARRAY,
    TEXTURE_CUBE_ARRAY
};

enum FORMAT {
    FORMAT_NONE                 = 0x00000000,
    FORMAT_R8Unorm              = 0x00000001,
    FORMAT_R8Snorm              = 0x00000002,
    FORMAT_R8Uint               = 0x00000003,
    FORMAT_R8Sint               = 0x00000004,
    FORMAT_R16Unorm             = 0x00000005,
    FORMAT_R16Snorm             = 0x00000006,
    FORMAT_R16Uint              = 0x00000007,
    FORMAT_R16Sint              = 0x00000008,
    FORMAT_R16Float             = 0x00000009,
    FORMAT_RG8Unorm             = 0x0000000A,
    FORMAT_RG8Snorm             = 0x0000000B,
    FORMAT_RG8Uint              = 0x0000000C,
    FORMAT_RG8Sint              = 0x0000000D,
    FORMAT_R32Float             = 0x0000000E,
    FORMAT_R32Uint              = 0x0000000F,
    FORMAT_R32Sint              = 0x00000010,
    FORMAT_RG16Unorm            = 0x00000011,
    FORMAT_RG16Snorm            = 0x00000012,
    FORMAT_RG16Uint             = 0x00000013,
    FORMAT_RG16Sint             = 0x00000014,
    FORMAT_RG16Float            = 0x00000015,
    FORMAT_RGBA8Unorm           = 0x00000016,
    FORMAT_RGBA8UnormSrgb       = 0x00000017,
    FORMAT_RGBA8Snorm           = 0x00000018,
    FORMAT_RGBA8Uint            = 0x00000019,
    FORMAT_RGBA8Sint            = 0x0000001A,
    FORMAT_BGRA8Unorm           = 0x0000001B,
    FORMAT_BGRA8UnormSrgb       = 0x0000001C,
    FORMAT_RGB10A2Uint          = 0x0000001D,
    FORMAT_RGB10A2Unorm         = 0x0000001E,
    FORMAT_RG11B10Ufloat        = 0x0000001F,
    FORMAT_RGB9E5Ufloat         = 0x00000020,
    FORMAT_RG32Float            = 0x00000021,
    FORMAT_RG32Uint             = 0x00000022,
    FORMAT_RG32Sint             = 0x00000023,
    FORMAT_RGBA16Unorm          = 0x00000024,
    FORMAT_RGBA16Snorm          = 0x00000025,
    FORMAT_RGBA16Uint           = 0x00000026,
    FORMAT_RGBA16Sint           = 0x00000027,
    FORMAT_RGBA16Float          = 0x00000028,
    FORMAT_RGBA32Float          = 0x00000029,
    FORMAT_RGBA32Uint           = 0x0000002A,
    FORMAT_RGBA32Sint           = 0x0000002B,
    FORMAT_Stencil8             = 0x0000002C,
    FORMAT_Depth16Unorm         = 0x0000002D,
    FORMAT_Depth24Plus          = 0x0000002E,
    FORMAT_Depth24PlusStencil8  = 0x0000002F,
    FORMAT_Depth32Float         = 0x00000030,
    FORMAT_Depth32FloatStencil8 = 0x00000031,
    FORMAT_BC1RGBAUnorm         = 0x00000032,
    FORMAT_BC1RGBAUnormSrgb     = 0x00000033,
    FORMAT_BC2RGBAUnorm         = 0x00000034,
    FORMAT_BC2RGBAUnormSrgb     = 0x00000035,
    FORMAT_BC3RGBAUnorm         = 0x00000036,
    FORMAT_BC3RGBAUnormSrgb     = 0x00000037,
    FORMAT_BC4RUnorm            = 0x00000038,
    FORMAT_BC4RSnorm            = 0x00000039,
    FORMAT_BC5RGUnorm           = 0x0000003A,
    FORMAT_BC5RGSnorm           = 0x0000003B,
    FORMAT_BC6HRGBUfloat        = 0x0000003C,
    FORMAT_BC6HRGBFloat         = 0x0000003D,
    FORMAT_BC7RGBAUnorm         = 0x0000003E,
    FORMAT_BC7RGBAUnormSrgb     = 0x0000003F,
    FORMAT_ETC2RGB8Unorm        = 0x00000040,
    FORMAT_ETC2RGB8UnormSrgb    = 0x00000041,
    FORMAT_ETC2RGB8A1Unorm      = 0x00000042,
    FORMAT_ETC2RGB8A1UnormSrgb  = 0x00000043,
    FORMAT_ETC2RGBA8Unorm       = 0x00000044,
    FORMAT_ETC2RGBA8UnormSrgb   = 0x00000045,
    FORMAT_EACR11Unorm          = 0x00000046,
    FORMAT_EACR11Snorm          = 0x00000047,
    FORMAT_EACRG11Unorm         = 0x00000048,
    FORMAT_EACRG11Snorm         = 0x00000049,
    FORMAT_ASTC4x4Unorm         = 0x0000004A,
    FORMAT_ASTC4x4UnormSrgb     = 0x0000004B,
    FORMAT_ASTC5x4Unorm         = 0x0000004C,
    FORMAT_ASTC5x4UnormSrgb     = 0x0000004D,
    FORMAT_ASTC5x5Unorm         = 0x0000004E,
    FORMAT_ASTC5x5UnormSrgb     = 0x0000004F,
    FORMAT_ASTC6x5Unorm         = 0x00000050,
    FORMAT_ASTC6x5UnormSrgb     = 0x00000051,
    FORMAT_ASTC6x6Unorm         = 0x00000052,
    FORMAT_ASTC6x6UnormSrgb     = 0x00000053,
    FORMAT_ASTC8x5Unorm         = 0x00000054,
    FORMAT_ASTC8x5UnormSrgb     = 0x00000055,
    FORMAT_ASTC8x6Unorm         = 0x00000056,
    FORMAT_ASTC8x6UnormSrgb     = 0x00000057,
    FORMAT_ASTC8x8Unorm         = 0x00000058,
    FORMAT_ASTC8x8UnormSrgb     = 0x00000059,
    FORMAT_ASTC10x5Unorm        = 0x0000005A,
    FORMAT_ASTC10x5UnormSrgb    = 0x0000005B,
    FORMAT_ASTC10x6Unorm        = 0x0000005C,
    FORMAT_ASTC10x6UnormSrgb    = 0x0000005D,
    FORMAT_ASTC10x8Unorm        = 0x0000005E,
    FORMAT_ASTC10x8UnormSrgb    = 0x0000005F,
    FORMAT_ASTC10x10Unorm       = 0x00000060,
    FORMAT_ASTC10x10UnormSrgb   = 0x00000061,
    FORMAT_ASTC12x10Unorm       = 0x00000062,
    FORMAT_ASTC12x10UnormSrgb   = 0x00000063,
    FORMAT_ASTC12x12Unorm       = 0x00000064,
    FORMAT_ASTC12x12UnormSrgb   = 0x00000065,

    FORMAT_ValidCount,
    FORMAT_Force32 = 0x7FFFFFFF
};

enum USAGE_FLAGS {
    USAGE_NONE                     = 0,
    USAGE_SAMPLED                  = 0x01,
    USAGE_STORAGE                  = 0x02,
    USAGE_COLOR_ATTACHMENT         = 0x04,
    USAGE_DEPTH_STENCIL_ATTACHMENT = 0x08,
    USAGE_TRANSFER_SRC             = 0x10,
    USAGE_TRANSFER_DST             = 0x20,
};

enum STAGE {
    STAGE_NONE = 0,
    STAGE_TRANSFER,
    STAGE_COMPUTE,
    STAGE_RASTER_COLOR_OUT,
    STAGE_PIXEL_SHADER,
    STAGE_VERTEX_SHADER,
};

enum HAZARD_FLAGS {
    HAZARD_DRAW_ARGUMENTS = 0x1,
    HAZARD_DESCRIPTORS    = 0x2,
    HAZARD_DEPTH_STENCIL  = 0x4
};

enum SIGNAL {
    SIGNAL_ATOMIC_SET,
    SIGNAL_ATOMIC_MAX,
    SIGNAL_ATOMIC_OR,
};

enum LOAD_OP {
    LOAD_OP_UNDEFINED,
    LOAD_OP_LOAD,
    LOAD_OP_CLEAR,
};

enum STORE_OP {
    STORE_OP_UNDEFINED,
    STORE_OP_STORE,
    STORE_OP_DISCARD,
};

enum QUEUE_TYPE {
    QUEUE_DEFAULT,
    QUEUE_COMPUTE,
    QUEUE_TRANSFER,
};

enum PRESENT_MODE {
    PRESENT_MODE_IMMEDIATE,
    PRESENT_MODE_MAILBOX,
    PRESENT_MODE_FIFO,
    PRESENT_MODE_FIFO_RELAXED,

    PRESENT_MODE_VALID_COUNT,
};

// Custom allocation callback - essentially a realloc function but not exactly
// the same as the C version.
// - ptr is null iff old_size is 0
// - When new_size != 0 and old_size == 0, should behave like malloc
// - If new_size == 0, the function must return null; if old_size != 0, it
// should free the block pointed to by ptr. It is the responsibility of the
// function to copy old_size bytes of memory from ptr to the returned pointer.
typedef MemoryBlock (*ProcAllocatorCallback)(void*    userdata,
                                             void*    ptr,
                                             uint32_t old_size,
                                             uint32_t new_size);
typedef void (*ProcLogCallback)(LogLevel lvl, Span<const char> message, void* userdata);

// Structs
struct Dimension3D {
    uint32_t x, y, z;
};

struct Stencil {
    OP      test        = OP_ALWAYS;
    OP      failOp      = OP_KEEP;
    OP      passOp      = OP_KEEP;
    OP      depthFailOp = OP_KEEP;
    uint8_t reference   = 0;
};

struct DeviceDesc {
    GpuPreference gpu_preference = GpuPreference_Discrete;

    uintptr_t native_window_handle   = 0;
    uintptr_t native_instance_handle = 0;

    ProcLogCallback       log_callback   = nullptr;
    void*                 log_userdata   = nullptr;
    LogLevel              log_level      = LogLevel_Off;
    ProcAllocatorCallback alloc_callback = nullptr;
    void*                 alloc_userdata = nullptr;
};

struct GpuDepthStencilDesc {
    DEPTH_FLAGS depthMode            = DEPTH_NONE;
    OP          depthTest            = OP_ALWAYS;
    float       depthBias            = 0.0f;
    float       depthBiasSlopeFactor = 0.0f;
    float       depthBiasClamp       = 0.0f;
    uint8_t     stencilReadMask      = 0xff;
    uint8_t     stencilWriteMask     = 0xff;
    Stencil     stencilFront;
    Stencil     stencilBack;
};

struct BlendDesc {
    BLEND   colorOp        = BLEND_ADD;
    FACTOR  srcColorFactor = FACTOR_ONE;
    FACTOR  dstColorFactor = FACTOR_ZERO;
    BLEND   alphaOp        = BLEND_ADD;
    FACTOR  srcAlphaFactor = FACTOR_ONE;
    FACTOR  dstAlphaFactor = FACTOR_ZERO;
    uint8_t colorWriteMask = 0xf;
};

struct ColorTarget {
    FORMAT  format    = FORMAT_NONE;
    uint8_t writeMask = 0xf;
};

struct RasterDesc {
    TOPOLOGY                topology                  = TOPOLOGY_TRIANGLE_LIST;
    CULL                    cull                      = CULL_NONE;
    bool                    alphaToCoverage           = false;
    bool                    supportDualSourceBlending = false;
    uint8_t                 sampleCount               = 1;
    FORMAT                  depthFormat               = FORMAT_NONE;
    FORMAT                  stencilFormat             = FORMAT_NONE;
    Span<const ColorTarget> colorTargets              = {};
    BlendDesc               blendstate                = {};
};

struct RenderAttachment {
    OP load_op;
    OP store_op;
};

struct RenderPassDesc {
    Span<const RenderAttachment> color_attachments;
};

struct GpuTextureDesc {
    TEXTURE     type = TEXTURE_2D;
    Dimension3D dimensions;
    uint32_t    mipCount    = 1;
    uint32_t    layerCount  = 1;
    uint32_t    sampleCount = 1;
    FORMAT      format      = FORMAT_NONE;
    USAGE_FLAGS usage       = USAGE_NONE;
};

struct TextureViewDesc {
    FORMAT   format     = FORMAT_NONE;
    uint8_t  baseMip    = 0;
    uint8_t  mipCount   = 0;
    uint16_t baseLayer  = 0;
    uint16_t layerCount = 0;
};

struct GpuTextureSizeAlign {
    size_t size;
    size_t align;
};

struct ShaderSource {
    ByteSpan         spirv;
    Span<const char> entry_point;
};

struct SurfaceCapabilities {
    USAGE_FLAGS              usages;
    Span<const FORMAT>       formats;
    Span<const PRESENT_MODE> present_modes;
};

struct SurfaceConfiguration {
    FORMAT       format;
    USAGE_FLAGS  usages;
    uint32_t     width;
    uint32_t     height;
    PRESENT_MODE present_mode;
};

struct SurfaceTextureInfo {
    enum {
        STATUS_SUCCESS,
        STATUS_SUBOPTIMAL,
        STATUS_OUT_OF_DATE,
        STATUS_ERROR,
    } status;

    Handle<Texture> texture;

    // Semaphore needs to be waited on before the texture can safely be used.
    Handle<Semaphore> acquire_semaphore;
};

struct SemaphoreInfo {
    Handle<Semaphore> semaphore;
    uint64_t          wait_value;
    STAGE stage = STAGE_NONE;  // Ignored on signal operations, what stage must be blocked on the
                               // wait operation
};

// Initialization
class Device {
   public:
    static Device create(const DeviceDesc& desc);
    Device() = default;
    ~Device();
    Device(const Device&) = delete;
    Device(Device&& other) : impl(std::exchange(other.impl, nullptr)) {}
    Device& operator=(const Device&) = delete;
    Device& operator=(Device&& other) {
        impl = std::exchange(other.impl, impl);
        return *this;
    }

    // Surface:
    SurfaceCapabilities get_surface_capabilities();
    bool                configure_surface(const SurfaceConfiguration& config);
    void                unconfigure_surface();
    SurfaceTextureInfo  get_current_texture();
    void                present(Handle<Queue> queue);

    // Buffers:
    Handle<Buffer> malloc(size_t bytes, MEMORY memory = MEMORY_DEFAULT);
    Handle<Buffer> malloc(size_t bytes, size_t align, MEMORY memory = MEMORY_DEFAULT);
    void           free(Handle<Buffer> buffer);
    GpuPtr         getDevicePointer(Handle<Buffer> buffer);

    // Textures:
    Handle<Texture>     createTexture(const GpuTextureDesc& desc);
    Handle<TextureHeap> createTextureHeap(size_t size);

    uint32_t createTextureView(Handle<TextureHeap> heap,
                               Handle<Texture>     texture,
                               TextureViewDesc     desc);
    uint32_t createRWTextureView(Handle<TextureHeap> heap,
                                 Handle<Texture>     texture,
                                 TextureViewDesc     desc);
    void     free(Handle<Texture>);
    void     free(Handle<TextureHeap>);
    void     freeTextureView(Handle<TextureHeap> heap, uint32_t view);

    // Pipelines
    Handle<Pipeline> createComputePipeline(ByteSpan computeIR);
    Handle<Pipeline> createGraphicsPipeline(ShaderSource      vertex,
                                            ShaderSource      fragment,
                                            const RasterDesc& desc);
    Handle<Pipeline> createGraphicsMeshletPipeline(ByteSpan   meshletIR,
                                                   ByteSpan   pixelIR,
                                                   RasterDesc desc);
    void             freePipeline(Handle<Pipeline> pipeline);

    // State objects
    Handle<DepthStencilState> createDepthStencilState(GpuDepthStencilDesc desc);
    Handle<BlendState>        createBlendState(BlendDesc desc);
    void                      freeDepthStencilState(Handle<DepthStencilState> state);
    void                      freeBlendState(Handle<BlendState> state);

    // Queue
    Handle<Queue>         getQueue(QUEUE_TYPE type = QUEUE_DEFAULT);
    Handle<CommandBuffer> startCommandRecording(Handle<Queue> queue);

    // TODO: May want to wrap these args in a struct.
    void submit(Handle<Queue>                     queue,
                Span<const Handle<CommandBuffer>> commandBuffers,
                Span<const SemaphoreInfo>         wait_semaphores,
                Span<const SemaphoreInfo>         signal_semaphores = {});
    void cancel(Handle<Queue> queue, Span<const Handle<CommandBuffer>> commandBuffers);

    // Semaphores
    Handle<Semaphore> createSemaphore(uint64_t initValue);
    void              waitSemaphore(Handle<Semaphore> sema, uint64_t value);
    void              destroySemaphore(Handle<Semaphore> sema);

   private:
    struct Impl;
    Impl* impl = nullptr;

    Device(Impl* impl) : impl{impl} {};
};

template <>
class Handle<CommandBuffer> {
   public:
    // Commands
    void memcpy(GpuPtr destGpu, GpuPtr srcGpu, size_t size);
    void copyToTexture(GpuPtr destGpu, GpuPtr srcGpu, Handle<Texture> texture);
    void copyFromTexture(GpuPtr destGpu, GpuPtr srcGpu, Handle<Texture> texture);

    void setActiveTextureHeapPtr(GpuPtr ptrGpu);

    void barrier(STAGE before, STAGE after, HAZARD_FLAGS hazards = HAZARD_FLAGS(0));

#if 0
// NOTE: Not sure this is implementable on top of vulkan right now. 
// Vulkan Events aren't sophisticated enough to do this API, but we could probably come up with a simpler one
void gpuSignalAfter(Handle<CommandBuffer> cb, STAGE before, GpuPtr ptrGpu,
                    uint64_t value, SIGNAL signal);
void gpuWaitBefore(Handle<CommandBuffer> cb, STAGE after, GpuPtr ptrGpu,
                   uint64_t value, OP op,
                   HAZARD_FLAGS hazards = HAZARD_FLAGS(0), uint64_t mask = ~0);
#endif

    void setPipeline(Handle<Pipeline> pipeline);
    void setDepthStencilState(Handle<DepthStencilState> state);
    void setBlendState(Handle<BlendState> state);

    void setTextureHeap(Handle<TextureHeap> heap);

    void dispatch(GpuPtr dataGpu, const Dimension3D& gridDimensions);
    void dispatchIndirect(GpuPtr dataGpu, GpuPtr gridDimensionsGpu);

    void beginRenderPass(RenderPassDesc desc);
    void endRenderPass();

    void draw(GpuPtr   vertexDataGpu,
              GpuPtr   fragmentDataGpu,
              uint32_t vertexCount,
              uint32_t instanceCount,
              uint32_t firstVertex   = 0,
              uint32_t firstInstance = 0);
    void drawIndexedInstanced(GpuPtr   vertexDataGpu,
                              GpuPtr   pixelDataGpu,
                              GpuPtr   indicesGpu,
                              uint32_t indexCount,
                              uint32_t instanceCount);
    void drawIndexedInstancedIndirect(GpuPtr vertexDataGpu,
                                      GpuPtr pixelDataGpu,
                                      GpuPtr indicesGpu,
                                      GpuPtr argsGpu);
    void drawIndexedInstancedIndirectMulti(GpuPtr   dataVxGpu,
                                           uint32_t vxStride,
                                           GpuPtr   dataPxGpu,
                                           uint32_t pxStride,
                                           GpuPtr   argsGpu,
                                           GpuPtr   drawCountGpu);

    void drawMeshlets(GpuPtr meshletDataGpu, GpuPtr pixelDataGpu, const Dimension3D& dim);
    void drawMeshletsIndirect(GpuPtr meshletDataGpu, GpuPtr pixelDataGpu, GpuPtr dimGpu);

   private:
    struct Impl;
    Impl* impl;
};

}  // namespace loon::gpu
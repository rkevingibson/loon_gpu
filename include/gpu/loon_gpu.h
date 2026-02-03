#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <utility>


#ifdef __cpp_concepts
#    define REQUIRES(x) requires x
#else
#    define REQUIRES(x)
#endif

// Helpers for flags - can use enum class for type safety while still allowing binary ops.
#define LOON_BITWISE_BINARY_OP(name, op)                                                           \
    inline constexpr name operator op(name lhs, name rhs) {                                        \
        using U = std::underlying_type_t<name>;                                                    \
        return name(static_cast<U>(lhs) op static_cast<U>(rhs));                                   \
    }

#define LOON_BITWISE_ASSIGNMENT_OP(name, op)                                                       \
    inline constexpr name operator op##=(name lhs, name rhs) {                                     \
        lhs = lhs op rhs;                                                                          \
        return lhs;                                                                                \
    }

#define LOON_DEFINE_BITWISE_OPS(name)                                                              \
    LOON_BITWISE_BINARY_OP(name, |);                                                               \
    LOON_BITWISE_BINARY_OP(name, &);                                                               \
    LOON_BITWISE_BINARY_OP(name, ^);                                                               \
    LOON_BITWISE_ASSIGNMENT_OP(name, |);                                                           \
    LOON_BITWISE_ASSIGNMENT_OP(name, &);                                                           \
    LOON_BITWISE_ASSIGNMENT_OP(name, ^);


namespace loon::gpu {

constexpr size_t kMaxTextureHeapSize = 32ull * 1024;
constexpr size_t kMaxNumTextureHeaps = 1024;

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

    constexpr Span(const Span<T>& src) noexcept = default;
    Span& operator=(const Span<T>& src)         = default;

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

    Span<unsigned char> as_bytes() {
        return Span<unsigned char>((unsigned char*)m_ptr, m_len * sizeof(T));
    }

    template <typename U>
    Span<U> cast() const {
        return Span<U>((U*)m_ptr, m_len * sizeof(T) / sizeof(U));
    }

   protected:
    T*     m_ptr{nullptr};
    size_t m_len{0};
};

inline constexpr Span<const char> operator""_sv(const char* val, size_t len) {
    return Span<const char>(val, len);
}

// A lightweight replacement for std::function.
// Tries to avoid heap allocations by being "fat", it can store reasonably sized lambdas without
// allocating, and will fail statically if the lambda is too big.
// Adapted from https://github.com/jlaumon/Bedrock/blob/main/Bedrock/Function.h
template <class Res, class... Args>
class Function {
   public:
    Function() = default;
    ~Function() {
        if (m_vtable) { m_vtable->destruct(m_storage); }
    }
    Function(const Function&)            = delete;
    Function& operator=(const Function&) = delete;

    Function(Function&& other) {
        if (other.m_vtable) {
            m_vtable = std::exchange(other.m_vtable, nullptr);
            m_vtable->move(m_storage, other.m_storage);
        }
    }

    Function& operator=(Function&& other) {
        if (m_vtable) { m_vtable->destruct(m_storage); }
        if (other.m_vtable) {
            m_vtable = std::exchange(other.m_vtable, nullptr);
            m_vtable->move(m_storage, other.m_storage);
        }
        return *this;
    }

    template <class F>
    Function(F&& fn)
        REQUIRES((!std::is_same_v<std::remove_reference_t<std::remove_cv_t<F>>, Function>)) {
        using T = std::decay_t<F>;
        static VTable table{
            .invoke =
                [](void* data, Args&&... args) {
                    reinterpret_cast<T*>(data)->operator()(std::forward<Args>(args)...);
                },
            .destruct = [](void* data) { reinterpret_cast<T*>(data)->~T(); },
            .move =
                [](void* dst, void* src) {
                    ::new (reinterpret_cast<T*>(dst)) T(std::move(*reinterpret_cast<T*>(src)));
                },
        };

        m_vtable = &table;
        static_assert(sizeof(F) < kStorageSize, "Lambda must be smaller than kStorageSize bytes");
        static_assert(std::is_move_constructible_v<F>, "Lambda must be move constructible");
        ::new (reinterpret_cast<F*>(m_storage)) F(std::forward<F>(fn));
    }

    Res operator()(Args&&... args) {
        if (m_vtable) { m_vtable->invoke(m_storage, std::forward<Args>(args)...); }
    }

    operator bool() const { return m_vtable != nullptr; }

   private:
    struct VTable {
        using InvokeFunc      = Res (*)(void*, Args&&...);
        using DestructFunc    = void (*)(void*);
        using MoveFunc        = void (*)(void* dst, void* src);
        InvokeFunc   invoke   = nullptr;
        DestructFunc destruct = nullptr;
        MoveFunc     move     = nullptr;
    };

    static constexpr size_t kStorageSize = 64 - sizeof(VTable*);
    VTable*                 m_vtable     = nullptr;
    uint8_t                 m_storage[kStorageSize];
};

typedef struct MemoryBlock {
    void*    ptr;
    uint32_t len;
} MemoryBlock;

// Opaque handles

template <class T>
struct Handle {
    uint64_t  h = 0;
    constexpr operator bool() const noexcept { return h != 0; }
};

struct Device;
struct Pipeline;
struct Buffer;
struct Texture;
struct TextureHeap;
struct TextureView;
struct DepthStencilState;
struct Queue;
struct CommandBuffer;
struct Semaphore;

using GpuPtr = uint64_t;

// MARK: Enums

enum class LogLevel : uint8_t {
    Off,
    Error,
    Warning,
    Info,
    Debug,
};

enum class GpuPreference : uint8_t {
    Discrete = 0,
    Integrated,
};

enum class Memory : uint8_t {
    Default,
    Gpu,
    Readback,
};

enum class Cull : uint8_t {
    CCW,
    CW,
    All,
    None,
};

enum class DepthFlags : uint8_t {
    None  = 0,
    Read  = 0x1,
    Write = 0x2,
};
LOON_DEFINE_BITWISE_OPS(DepthFlags);

enum class Op : uint8_t {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
    Keep,
};

enum class Blend : uint8_t {
    Add,
    Subtract,
    RevSubtract,
    Min,
    Max,
};

enum class Factor : uint8_t {
    Zero,
    One,
    SrcColor,
    DstColor,
    SrcAlpha,
};

enum class Topology : uint8_t {
    TriangleList,
    TriangleStrip,
    TriangleFan,
};

enum class TextureType : uint8_t {
    Tex1D,
    Tex2D,
    Tex3D,
    TexCube,
    Tex2DArray,
    TexCubeArray,
};

enum class Format : uint32_t {
    None                 = 0x00000000,
    R8Unorm              = 0x00000001,
    R8Snorm              = 0x00000002,
    R8Uint               = 0x00000003,
    R8Sint               = 0x00000004,
    R16Unorm             = 0x00000005,
    R16Snorm             = 0x00000006,
    R16Uint              = 0x00000007,
    R16Sint              = 0x00000008,
    R16Float             = 0x00000009,
    RG8Unorm             = 0x0000000A,
    RG8Snorm             = 0x0000000B,
    RG8Uint              = 0x0000000C,
    RG8Sint              = 0x0000000D,
    R32Float             = 0x0000000E,
    R32Uint              = 0x0000000F,
    R32Sint              = 0x00000010,
    RG16Unorm            = 0x00000011,
    RG16Snorm            = 0x00000012,
    RG16Uint             = 0x00000013,
    RG16Sint             = 0x00000014,
    RG16Float            = 0x00000015,
    RGBA8Unorm           = 0x00000016,
    RGBA8UnormSrgb       = 0x00000017,
    RGBA8Snorm           = 0x00000018,
    RGBA8Uint            = 0x00000019,
    RGBA8Sint            = 0x0000001A,
    BGRA8Unorm           = 0x0000001B,
    BGRA8UnormSrgb       = 0x0000001C,
    RGB10A2Uint          = 0x0000001D,
    RGB10A2Unorm         = 0x0000001E,
    RG11B10Ufloat        = 0x0000001F,
    RGB9E5Ufloat         = 0x00000020,
    RG32Float            = 0x00000021,
    RG32Uint             = 0x00000022,
    RG32Sint             = 0x00000023,
    RGBA16Unorm          = 0x00000024,
    RGBA16Snorm          = 0x00000025,
    RGBA16Uint           = 0x00000026,
    RGBA16Sint           = 0x00000027,
    RGBA16Float          = 0x00000028,
    RGBA32Float          = 0x00000029,
    RGBA32Uint           = 0x0000002A,
    RGBA32Sint           = 0x0000002B,
    Stencil8             = 0x0000002C,
    Depth16Unorm         = 0x0000002D,
    Depth24Plus          = 0x0000002E,
    Depth24PlusStencil8  = 0x0000002F,
    Depth32Float         = 0x00000030,
    Depth32FloatStencil8 = 0x00000031,
    BC1RGBAUnorm         = 0x00000032,
    BC1RGBAUnormSrgb     = 0x00000033,
    BC2RGBAUnorm         = 0x00000034,
    BC2RGBAUnormSrgb     = 0x00000035,
    BC3RGBAUnorm         = 0x00000036,
    BC3RGBAUnormSrgb     = 0x00000037,
    BC4RUnorm            = 0x00000038,
    BC4RSnorm            = 0x00000039,
    BC5RGUnorm           = 0x0000003A,
    BC5RGSnorm           = 0x0000003B,
    BC6HRGBUfloat        = 0x0000003C,
    BC6HRGBFloat         = 0x0000003D,
    BC7RGBAUnorm         = 0x0000003E,
    BC7RGBAUnormSrgb     = 0x0000003F,
    ETC2RGB8Unorm        = 0x00000040,
    ETC2RGB8UnormSrgb    = 0x00000041,
    ETC2RGB8A1Unorm      = 0x00000042,
    ETC2RGB8A1UnormSrgb  = 0x00000043,
    ETC2RGBA8Unorm       = 0x00000044,
    ETC2RGBA8UnormSrgb   = 0x00000045,
    EACR11Unorm          = 0x00000046,
    EACR11Snorm          = 0x00000047,
    EACRG11Unorm         = 0x00000048,
    EACRG11Snorm         = 0x00000049,
    ASTC4x4Unorm         = 0x0000004A,
    ASTC4x4UnormSrgb     = 0x0000004B,
    ASTC5x4Unorm         = 0x0000004C,
    ASTC5x4UnormSrgb     = 0x0000004D,
    ASTC5x5Unorm         = 0x0000004E,
    ASTC5x5UnormSrgb     = 0x0000004F,
    ASTC6x5Unorm         = 0x00000050,
    ASTC6x5UnormSrgb     = 0x00000051,
    ASTC6x6Unorm         = 0x00000052,
    ASTC6x6UnormSrgb     = 0x00000053,
    ASTC8x5Unorm         = 0x00000054,
    ASTC8x5UnormSrgb     = 0x00000055,
    ASTC8x6Unorm         = 0x00000056,
    ASTC8x6UnormSrgb     = 0x00000057,
    ASTC8x8Unorm         = 0x00000058,
    ASTC8x8UnormSrgb     = 0x00000059,
    ASTC10x5Unorm        = 0x0000005A,
    ASTC10x5UnormSrgb    = 0x0000005B,
    ASTC10x6Unorm        = 0x0000005C,
    ASTC10x6UnormSrgb    = 0x0000005D,
    ASTC10x8Unorm        = 0x0000005E,
    ASTC10x8UnormSrgb    = 0x0000005F,
    ASTC10x10Unorm       = 0x00000060,
    ASTC10x10UnormSrgb   = 0x00000061,
    ASTC12x10Unorm       = 0x00000062,
    ASTC12x10UnormSrgb   = 0x00000063,
    ASTC12x12Unorm       = 0x00000064,
    ASTC12x12UnormSrgb   = 0x00000065,

    ValidCount,
};

enum class UsageFlags : uint16_t {
    None                   = 0,
    Sampled                = 0x01,
    Storage                = 0x02,
    ColorAttachment        = 0x04,
    DepthStencilAttachment = 0x08,
    TransferSrc            = 0x10,
    TransferDst            = 0x20,
};
LOON_DEFINE_BITWISE_OPS(UsageFlags);

enum StageFlags : uint16_t {
    None           = 0,
    Transfer       = 0x01,
    Compute        = 0x02,
    RasterColorOut = 0x04,
    PixelShader    = 0x08,
    VertexShader   = 0x10,
    Host           = 0x20,
};
LOON_DEFINE_BITWISE_OPS(StageFlags);

enum class Layout : uint8_t {
    DontCare = 0,
    General,
    Attachment,
    Present,
};

enum class HazardFlags : uint8_t {
    DrawArguments = 0x1,
    Descriptors   = 0x2,
    DepthStencil  = 0x4,
};
LOON_DEFINE_BITWISE_OPS(HazardFlags);

enum class LoadOp : uint8_t {
    Undefined,
    Load,
    Clear,
};

enum class StoreOp : uint8_t {
    Undefined,
    Store,
    Discard,
};

enum class QueueType : uint8_t {
    Default,
    Compute,
    Transfer,

    ValidCount,
};

enum class PresentMode : uint8_t {
    Immediate,
    Mailbox,
    Fifo,
    FifoRelaxed,

    ValidCount,
};

enum class SurfaceStatus : uint8_t {
    Success,
    Suboptimal,
    OutOfDate,
    Error,
};

enum class SamplerCoords : uint8_t {
    Normalized,
    Pixel,
};

enum class SamplerFilter : uint8_t {
    Nearest,
    Linear,
};

enum class SamplerAddressing : uint8_t {
    ClampToEdge,
    Repeat,
    Mirrored,
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

// MARK: Structs

struct Dimension2D {
    uint32_t x, y;
};

struct Dimension3D {
    uint32_t x, y, z;
};

struct Rect2D {
    uint32_t offset_x = 0;
    uint32_t offset_y = 0;
    uint32_t width;
    uint32_t height;
};

struct Color {
    uint32_t r;
    uint32_t g;
    uint32_t b;
    uint32_t a;
};

struct Stencil {
    Op      test          = Op::Always;
    Op      fail_op       = Op::Keep;
    Op      pass_op       = Op::Keep;
    Op      depth_fail_op = Op::Keep;
    uint8_t reference     = 0;
};

struct SamplerDesc {
    SamplerCoords     coord          = SamplerCoords::Normalized;
    SamplerFilter     filter         = SamplerFilter::Nearest;
    SamplerAddressing address        = SamplerAddressing::ClampToEdge;
    float             max_anisotropy = 0.0f;
};

static constexpr SamplerDesc kDefaultSamplers[] = {SamplerDesc{
    .coord          = SamplerCoords::Normalized,
    .filter         = SamplerFilter::Linear,
    .address        = SamplerAddressing::ClampToEdge,
    .max_anisotropy = 8.0f,
}};

struct DeviceDesc {
    GpuPreference gpu_preference = GpuPreference::Discrete;

    uintptr_t native_window_handle   = 0;
    uintptr_t native_instance_handle = 0;

    ProcLogCallback       log_callback   = nullptr;
    void*                 log_userdata   = nullptr;
    LogLevel              log_level      = LogLevel::Off;
    ProcAllocatorCallback alloc_callback = nullptr;
    void*                 alloc_userdata = nullptr;

    Span<const SamplerDesc> samplers = kDefaultSamplers;
};

struct DepthStencilDesc {
    DepthFlags depth_mode              = DepthFlags::None;
    Op         depth_test              = Op::Always;
    float      depth_bias              = 0.0f;
    float      depth_bias_slope_factor = 0.0f;
    float      depth_bias_clamp        = 0.0f;
    uint8_t    stencil_read_mask       = 0xff;
    uint8_t    stencil_write_mask      = 0xff;
    Stencil    stencil_front;
    Stencil    stencil_bsack;
};

struct BlendDesc {
    Blend   color_op         = Blend::Add;
    Factor  src_color_factor = Factor::One;
    Factor  dst_color_factor = Factor::Zero;
    Blend   alpha_op         = Blend::Add;
    Factor  src_alpha_factor = Factor::One;
    Factor  dst_alpha_factor = Factor::Zero;
    uint8_t color_write_mask = 0xf;
};

struct ColorTarget {
    Format  format     = Format::None;
    uint8_t write_mask = 0xf;
};

struct RasterDesc {
    Topology                topology                     = Topology::TriangleList;
    Cull                    cull                         = Cull::None;
    bool                    alpha_to_coverage            = false;
    bool                    support_dual_source_blending = false;
    uint8_t                 sample_count                 = 1;
    Format                  depth_format                 = Format::None;
    Format                  stencil_format               = Format::None;
    Span<const ColorTarget> color_targets                = {};
    BlendDesc               blendstate                   = {};
};

struct RenderAttachment {
    Handle<TextureView> texture_view = {0};
    LoadOp              load_op;
    StoreOp             store_op;
    Color               clear_color;
};

struct RenderPassDesc {
    Span<const RenderAttachment> color_attachments;
    RenderAttachment             depth_attachment;
    Rect2D                       render_area;
};

struct TextureDesc {
    TextureType type = TextureType::Tex2D;
    Dimension3D dimensions;
    uint32_t    mip_count    = 1;
    uint32_t    layer_count  = 1;
    uint32_t    sample_count = 1;
    Format      format       = Format::None;
    UsageFlags  usage        = UsageFlags::None;
};

struct TextureViewDesc {
    Format   format      = Format::None;
    uint8_t  base_mip    = 0;
    uint8_t  mip_count   = 1;
    uint16_t base_layer  = 0;
    uint16_t layer_count = 1;
};

struct TextureSizeAlign {
    size_t size;
    size_t align;
};

struct ShaderSource {
    Span<uint8_t>    spirv;
    Span<const char> entry_point;
};

struct SurfaceCapabilities {
    UsageFlags              usages;
    Span<const Format>      formats;
    Span<const PresentMode> present_modes;
};

struct SurfaceConfiguration {
    Format      format;
    UsageFlags  usages;
    uint32_t    width;
    uint32_t    height;
    PresentMode present_mode;
};

struct SurfaceTextureInfo {
    SurfaceStatus   status;
    Handle<Texture> texture;

    // Semaphore needs to be waited on before the texture can safely be used.
    Handle<Semaphore> acquire_semaphore;
};

struct SemaphoreInfo {
    Handle<Semaphore> semaphore;
    uint64_t          value;
    StageFlags        stage = None;  // Ignored on signal operations, what stage must be blocked on
                                     // the wait operation
};

struct TextureTransition {
    Handle<Texture> texture;
    Layout          old_layout = Layout::DontCare;
    Layout          new_layout = Layout::General;
};

struct BufferToTextureCopyInfo {
    Dimension2D buffer_image_size;
    Dimension3D image_offset{0, 0, 0};
    Dimension3D image_extent;
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

    void wait_for_device_idle();

    // Surface:
    SurfaceCapabilities get_surface_capabilities();
    bool                configure_surface(const SurfaceConfiguration& config);
    void                unconfigure_surface();
    SurfaceTextureInfo  get_current_texture();
    SurfaceStatus       present(Handle<Queue> queue);

    // Buffers:
    Handle<Buffer> malloc(size_t bytes, Memory memory = Memory::Default);
    Handle<Buffer> malloc(size_t bytes, size_t align, Memory memory = Memory::Default);
    void           free(Handle<Buffer> buffer);
    GpuPtr         get_device_pointer(Handle<Buffer> buffer);
    void*          get_host_pointer(Handle<Buffer> buffer);

    // Textures:
    Handle<Texture>     create_texture(const TextureDesc& desc);
    Handle<TextureHeap> create_texture_heap(size_t size);

    Handle<TextureView> create_texture_view(Handle<Texture> texture, TextureViewDesc desc);

    uint32_t add_texture_view_to_heap(Handle<TextureHeap>, Handle<TextureView>);
    void     remove_texture_view_from_heap(Handle<TextureHeap>, uint32_t);

    void free(Handle<Texture>);
    void free(Handle<TextureHeap>);
    void free(Handle<TextureView>);

    // Pipelines
    Handle<Pipeline> create_compute_pipeline(ShaderSource computeIR);
    Handle<Pipeline> create_graphics_pipeline(ShaderSource      vertex,
                                              ShaderSource      fragment,
                                              const RasterDesc& desc);
    void             free(Handle<Pipeline> pipeline);

    // State objects
    Handle<DepthStencilState> create_depth_stencil_state(DepthStencilDesc desc);
    void                      free_depth_stencil_state(Handle<DepthStencilState> state);

    // Queue
    Handle<Queue> get_queue(QueueType type = QueueType::Default);
    CommandBuffer start_command_recording(Handle<Queue> queue);

    // TODO: May want to wrap these args in a struct.
    void submit(Handle<Queue>             queue,
                Span<const CommandBuffer> commandBuffers,
                Span<const SemaphoreInfo> wait_semaphores,
                Span<const SemaphoreInfo> signal_semaphores = {});
    void cancel(Handle<Queue> queue, Span<const Handle<CommandBuffer>> commandBuffers);
    void on_submitted_work_completed(Handle<Queue> queue, Function<void>&& fn);
    void process_events(Handle<Queue> queue);

    // Semaphores
    Handle<Semaphore> create_semaphore(uint64_t initValue);
    void              wait_semaphore(Handle<Semaphore> sema, uint64_t value);
    void              free(Handle<Semaphore> sema);

   private:
    struct Impl;
    Impl* impl = nullptr;
    friend class CommandBuffer;

    Device(Impl* impl) : impl{impl} {};
};

class CommandBuffer {
   public:
    CommandBuffer()                                = default;
    CommandBuffer(const CommandBuffer&)            = default;
    CommandBuffer& operator=(const CommandBuffer&) = default;

    // Commands
    void memcpy(GpuPtr destGpu, GpuPtr srcGpu, size_t size);
    void copy_to_texture(GpuPtr src, Handle<Texture> texture, const BufferToTextureCopyInfo& info);
    void copy_from_texture(GpuPtr destGpu, GpuPtr srcGpu, Handle<Texture> texture);

    void set_texture_heap(Handle<TextureHeap> heap);

    void barrier(StageFlags                    before,
                 StageFlags                    after,
                 Span<const TextureTransition> image_transitions = {},
                 HazardFlags                   hazards           = HazardFlags(0));

    void set_pipeline(Handle<Pipeline> pipeline);
    void set_depth_stencil_State(Handle<DepthStencilState> state);

    void dispatch(GpuPtr dataGpu, const Dimension3D& gridDimensions);
    void dispatch_indirect(GpuPtr dataGpu, GpuPtr gridDimensionsGpu);

    void begin_render_pass(RenderPassDesc desc);
    void end_render_pass();

    void draw(GpuPtr   vertexDataGpu,
              GpuPtr   fragmentDataGpu,
              uint32_t vertexCount,
              uint32_t instanceCount);
    void draw_indexed_instanced(GpuPtr   vertexDataGpu,
                                GpuPtr   pixelDataGpu,
                                GpuPtr   indicesGpu,
                                uint32_t indexCount,
                                uint32_t instanceCount);
    void draw_indexed_instanced_indirect(GpuPtr vertexDataGpu,
                                         GpuPtr pixelDataGpu,
                                         GpuPtr indicesGpu,
                                         GpuPtr argsGpu);
    void draw_indexed_instanced_indirect_multi(GpuPtr   vertexDataGpu,
                                               GpuPtr   pixelDataGpu,
                                               GpuPtr   indicesGpu,
                                               GpuPtr   argsGpu,
                                               GpuPtr   drawCountGpu,
                                               uint32_t maxDraws);

   private:
    void set_compute_ptr(GpuPtr dataGpu);
    void set_graphics_ptrs(GpuPtr vertexDataGpu, GpuPtr fragmentDataGpu);

    friend class Device::Impl;
    CommandBuffer(void* buffer, void* device) : buffer{buffer}, device(device) {};
    void* buffer;
    void* device;
};

extern template class Span<const char>;
extern template class Span<uint8_t>;
extern template class Span<const gpu::SamplerDesc>;
extern template class Span<const gpu::ColorTarget>;
extern template class Span<const gpu::RenderAttachment>;
extern template class Span<const gpu::Format>;
extern template class Span<const gpu::PresentMode>;
extern template class Span<const gpu::CommandBuffer>;
extern template class Span<const gpu::SemaphoreInfo>;
extern template class Span<const Handle<gpu::CommandBuffer>>;
extern template class Span<const gpu::TextureTransition>;
extern template class Function<void>;
extern template class Function<void>;

}  // namespace loon::gpu
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
    inline constexpr name operator op## = (name lhs, name rhs) {                                   \
        lhs = lhs op rhs;                                                                          \
        return lhs;                                                                                \
    }

#define LOON_BITWISE_BOOL_OP(name)                                                                 \
    inline constexpr bool any(name x) {                                                            \
        return static_cast<std::underlying_type_t<name>>(x) != 0;                                  \
    }

#define LOON_DEFINE_BITWISE_OPS(name)                                                              \
    LOON_BITWISE_BINARY_OP(name, |);                                                               \
    LOON_BITWISE_BINARY_OP(name, &);                                                               \
    LOON_BITWISE_BINARY_OP(name, ^);                                                               \
    LOON_BITWISE_ASSIGNMENT_OP(name, |);                                                           \
    LOON_BITWISE_ASSIGNMENT_OP(name, &);                                                           \
    LOON_BITWISE_ASSIGNMENT_OP(name, ^);                                                           \
    LOON_BITWISE_BOOL_OP(name);

namespace loon::gpu {

constexpr size_t kMaxTextureHeapSize = 32ull * 1024;
constexpr size_t kMaxNumSamplers     = 4000;
constexpr size_t kMaxNumTextureHeaps = 1024;
/**
 * @brief A non-owning view class. Used heavily for passing array arguments.
 * This is analogous to std::span, but included here for pre-c++20 support.
 * Also supports initialization from an initializer list, or even a single object (a span of 1), if
 * the span is a const view. This is very useful for creating spans of temporary objects for
 * passing to functions, but you need to be careful - lifetimes of temporaries only last until the
 * end of the statement they're in.
 * @tparam T The type you have a view over
 */
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

    /**
     * @brief Construct a span from a pointer and length.
     *
     */
    constexpr Span(T* ptr, size_t len) noexcept : m_ptr{ptr}, m_len{len} {}

    /**
     * @brief Construct a span from a pair of pointers. end is a pointer one-past-the-end.
     *
     */
    constexpr Span(T* begin, T* end) noexcept :
        m_ptr{begin}, m_len{static_cast<size_t>(end - begin)} {}

    /**
     * @brief Construct a span from a C-style array. Length is automatically inferred.
     */
    template <size_t N>
    constexpr Span(T (&a)[N]) noexcept : Span(a, N) {}

    /**
     * @brief Construct from an initializer list.
     * Only acceptable in the case where a Span is being passed as a function argument, otherwise
     * the initializer list will not outlive the Span, and you end up with a dangling pointer. e.g.
     * for a function void Process(rkg::span<const float> x);
     * We can do
     * ```
     * Process({1, 2, 3});
     * ```
     * But not:
     * ```
     * Span args = {1,2,3}; // Dangling pointer after this line.
     * Process(args);
     * ```
     */
    constexpr Span(std::initializer_list<T> v) noexcept REQUIRES(is_const<T>) :
        Span(v.begin(), v.size()) {}

    /**
     * @brief Construct from a single value.
     * Same rules as for initializer list, useful for passing one
     * element to a function which takes a span.
     */
    constexpr Span(const T& v) noexcept REQUIRES(is_const<T>) : Span(&v, 1) {}

    /**
     * @brief Construct span of const objects from const span of non-const objects.
     *
     */
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

    Span<const uint8_t> as_bytes() const {
        return Span<const uint8_t>((const uint8_t*)m_ptr, m_len * sizeof(T));
    }

    template <typename U>
    Span<U> cast() const {
        return Span<U>((U*)m_ptr, m_len * sizeof(T) / sizeof(U));
    }

   private:
    T*     m_ptr{nullptr};
    size_t m_len{0};
};

inline constexpr Span<const char> operator""_sv(const char* val, size_t len) {
    return Span<const char>(val, len);
}

/**
 * @brief A lightweight replacement for std::function.
 * Tries to avoid heap allocations by being "fat", it can store reasonably sized lambdas without
 * allocating, and will fail statically if the lambda is too big. Adapted from
 * https://github.com/jlaumon/Bedrock/blob/main/Bedrock/Function.h
 */
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
    uint64_t       h = 0;
    constexpr      operator bool() const noexcept { return h != 0; }
    constexpr bool operator==(const Handle& other) { return other.h == h; }
};

/** @defgroup Device Device
 */

/**
 * @brief Opaque object representing a logical GPU device.
 *
 */
typedef struct DeviceImpl*        Device;
typedef struct QueueImpl*         Queue;
typedef struct CommandBufferImpl* CommandBuffer;
struct Pipeline;
struct Texture;
struct TextureHeap;
struct DepthStencilState;
struct Semaphore;

using GpuPtr      = uint64_t;
using TextureView = uint64_t;
using Sampler     = uint64_t;

// MARK: Enums

/**
 * @brief Logging level
 *
 */
enum class LogLevel : uint8_t {
    Off,
    Error,
    Warning,
    Info,
    Debug,
};

/**
 * @brief Preferred GPU, used when creating a device.
 *
 */
enum class GpuPreference : uint8_t {
    Discrete = 0,
    Integrated,
};

/**
 * @brief
 *
 */
enum class Backend {
    Vulkan,
    Metal,
};

/**
 * @brief Type of memory to allocate.
 *
 */
enum class Memory : uint8_t {
    Default,   ///< CPU visible memory, optimized for writing to from the CPU and reading from GPU
    Gpu,       ///< GPU-only memory, not visible from the CPU
    Readback,  ///< CPU visible memory, optimized for reading from the CPU.
};

enum class FrontFace : uint8_t {
    CCW = 0,  ///< Counter-clockwise
    CW,       ///< Clockwise
};

enum class Cull : uint8_t {
    Front,
    Back,
    None,
};

enum class DepthFlags : uint8_t {
    None  = 0,
    Read  = 0x1,
    Write = 0x2,
};
LOON_DEFINE_BITWISE_OPS(DepthFlags);

/**
 * @brief Comparison operation for depth and stencil testing
 *
 */
enum class Op : uint8_t {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
};

/**
 * @brief Operations for stencil buffers
 *
 */
enum class StencilOp : uint8_t {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
};

/**
 * @brief Operations for color/alpha blending.
 *
 */
enum class Blend : uint8_t {
    Add,
    Subtract,
    RevSubtract,
    Min,
    Max,
};

/**
 * @brief Blend factors for color/alpha blending.
 *
 */
enum class Factor : uint8_t {
    Zero,
    One,
    SrcColor,
    DstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
};

/**
 * @brief Input primitive to be used for a render pass.
 *
 */
enum class Topology : uint8_t {
    TriangleList,
    TriangleStrip,
};

enum class TextureType : uint8_t {
    Tex1D,
    Tex2D,
    Tex3D,
    TexCube,
    Tex2DArray,
    TexCubeArray,
};

/**
 * @brief Texture formats
 *
 */
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

enum class StageFlags : uint16_t {
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
};

enum class HazardFlags : uint8_t {
    None          = 0x0,
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
    Default,  ///< Queue capable of doing graphics, compute and transfer work
    // TODO: Enable multiple queue types. Right now, we only support a single global queue.
    // Compute,   ///< Dedicated compute-only queue
    // Transfer,  ///< Dedicated transfer-only queue

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
    Normalized,  ///< Coordinates lie in [0,1] range
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

enum class SpecializationConstantType : uint8_t {
    UInt8,
    UInt16,
    UInt32,
    Int8,
    Int16,
    Int32,
    Boolean,
    Float32,
};

enum class IndexType : uint8_t {
    UInt16,
    UInt32,
};

enum class ErrorType : uint8_t {
    NoError,
    Validation,
    OutOfMemory,
    Internal,
    Unknown,
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
typedef void (*ProcErrorCallback)(ErrorType type, Span<const char> message, void* userdata);

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
    Op        test          = Op::Always;
    StencilOp fail_op       = StencilOp::Keep;
    StencilOp pass_op       = StencilOp::Keep;
    StencilOp depth_fail_op = StencilOp::Keep;
    uint8_t   reference     = 0;
};

struct SamplerDesc {
    SamplerCoords     coord          = SamplerCoords::Normalized;
    SamplerFilter     filter         = SamplerFilter::Nearest;
    SamplerAddressing address        = SamplerAddressing::ClampToEdge;
    float             max_anisotropy = 1.0f;
};

/**
 * @brief Descriptor for initializing a device.
 *
 */
struct DeviceDesc {
    GpuPreference gpu_preference
        = GpuPreference::Discrete;  ///< Preferred GPU, if more than one valid device is available.

    uintptr_t native_window_handle = 0;    ///< Handle for the platform-specific window. HWND on
                                           ///< windows, CAMetalLayer* on MacOS, Window on XLib
    uintptr_t native_instance_handle = 0;  ///< Handle for the platform-specific instance. HINSTANCE
                                           ///< on windows, Display* on XLib. Ignored on MacOS.

    ///@{
    /** @name Logging callback */
    ProcLogCallback log_callback = nullptr;
    void*           log_userdata = nullptr;
    LogLevel        log_level    = LogLevel::Off;
    ///@}

    ///@{
    /** @name Allocation callback */
    ProcAllocatorCallback alloc_callback = nullptr;
    void*                 alloc_userdata = nullptr;
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
    Stencil    stencil_back;
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
    Format    format     = Format::None;
    BlendDesc blendstate = {};
};

struct RasterDesc {
    Topology                topology          = Topology::TriangleList;
    bool                    alpha_to_coverage = false;
    uint8_t                 sample_count      = 1;
    Format                  depth_format      = Format::None;
    Format                  stencil_format    = Format::None;
    Span<const ColorTarget> color_targets     = {};
};

struct RenderAttachment {
    Handle<Texture> texture = {0};
    LoadOp          load_op;
    StoreOp         store_op;
    Color           clear_color;
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

struct TextureHeapDesc {
    uint32_t texture_count    = 0;
    uint32_t rw_texture_count = 0;
    uint32_t sampler_count    = 0;
};

struct TextureViewDesc {
    Handle<Texture> texture;
    Format          format      = Format::None;
    uint8_t         base_mip    = 0;
    uint8_t         mip_count   = 1;
    uint16_t        base_layer  = 0;
    uint16_t        layer_count = 1;
};

struct TextureSizeAlign {
    size_t size;
    size_t align;
};

struct SpecializationConstant {
    uint32_t constant_id;
    union {
        uint64_t int_val;
        float    float_val;
        bool     bool_val;
    };
    SpecializationConstantType type;
};

struct ShaderSource {
    Span<const uint8_t> spirv;
    Span<const char>    entry_point;
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
};

struct SemaphoreInfo {
    Handle<Semaphore> semaphore;
    uint64_t          value;
    StageFlags        stage = StageFlags::None;  // Ignored on signal operations, what stage must be
                                                 // blocked on the wait operation
};

struct TextureTransition {
    Handle<Texture> texture;
    Layout          old_layout = Layout::DontCare;
    Layout          new_layout = Layout::General;
};

struct BufferToTextureCopyInfo {
    Dimension3D image_extent;
    uint32_t    source_row_pixels_stride
        = 0;  ///< Number of pixels between subsequent rows of data in the source buffer. If 0,
              ///< treated as equal to image_extent.x. Otherwise, should be >= image_extent.x
    uint32_t source_plane_rows_stride
        = 0;  ///< Number of rows in a plane of image in the source buffer. If 0, treated as equal
              ///< to image_extent.y. Otherwise, should be >= image_extent.y.
    Dimension3D destination_image_offset{0, 0, 0};

    uint8_t base_mip   = 0;
    uint8_t base_layer = 0;
};

struct DrawIndexedInstancedInfo {
    GpuPtr    vertexDataGpu;
    GpuPtr    fragmentDataGpu;
    GpuPtr    indicesGpu;
    uint32_t  indexCount;
    uint32_t  instanceCount = 1;
    IndexType type          = IndexType::UInt16;
};

struct DrawIndexedIndirectInfo {
    GpuPtr    vertexDataGpu;
    GpuPtr    fragmentDataGpu;
    GpuPtr    indicesGpu;
    GpuPtr    argsGpu;
    IndexType type = IndexType::UInt16;
};

struct MultiDrawIndirectInfo {
    GpuPtr    vertexDataGpu;
    GpuPtr    pixelDataGpu;
    GpuPtr    indicesGpu;
    GpuPtr    argsGpu;
    GpuPtr    drawCountGpu;
    uint32_t  maxDraws;
    IndexType type = IndexType::UInt16;
};

struct alignas(8) DrawIndexedIndirectGpuArgs {
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int32_t  vertex_offset;
    uint32_t first_instance;
};

/**
 * @addtogroup Device
 * @{
 */
/**
 * @brief Create a device object
 *
 * @return Opaque pointer to a device object.
 */
Device create_device(const DeviceDesc&);

/**
 * @brief Destroy a device object.
 * Note: this will flush the gpu, and destroy any created resources.
 */
void destroy_device(Device d);

/**
 * @brief Get the current backend for the device.
 * Currently this is statically determined, but in the future we may support selection at device
 * creation.
 */
Backend device_backend();

/**
 * @brief Block until any pending work on the GPU is completed.
 *
 * @param d
 */
void device_wait_for_idle(Device d);

/** @defgroup Surface Surface Management
 *
 * Surface management functions.
 * While surfaces are strictly optional, there currently is some command buffer management that is
 * tied to the frame loop implied by `present()` calls. See
 * [github](https://github.com/rkevingibson/loon_gpu/issues/1) for details.
 */

/** @ingroup Surface
 * @brief Get the surface capabilities for a device.
 *
 * @return SurfaceCapabilities
 */
SurfaceCapabilities get_surface_capabilities(Device d);


/** @ingroup Surface
 * @brief Configures the presentation surface.
 * Must be called before `get_current_texture()`
 * @return bool indicating if configuration succeeds. Errors will be logged.
 */
bool configure_surface(Device d, const SurfaceConfiguration& config);

/** @ingroup Surface
 * @brief Reset the surface configuration
 *
 * @param d
 */
void unconfigure_surface(Device d);

/** @ingroup Surface
 * @brief Get the current texture for presentation this frame.
 * Should only be called once per frame, and as late in the frame as possible to maximize GPU
 * utilization.
 * @return SurfaceTextureInfo
 */
SurfaceTextureInfo get_current_texture(Device d);

/** @ingroup Surface
 * @brief Present the current surface image on the provided queue.
 *
 * @return SurfaceStatus
 */
SurfaceStatus present(Device d, Queue queue);

/** @defgroup res Resources
 *
 * Resource Management Functions
 */

/// @{

/**
 * @brief Allocate memory with the specified size and type.
 *
 */
GpuPtr malloc(Device d, size_t bytes, Memory memory = Memory::Default);

/**
 * @brief Allocate memory with the specified size, alignment and type.
 *
 */
GpuPtr malloc(Device d, size_t bytes, size_t align, Memory memory = Memory::Default);

/**
 * @brief Free a pointer allocated with malloc
 *
 */
void free(Device d, GpuPtr ptr);

/**
 * @brief Get the corresponding CPU-side pointer for a GPU pointer.
 *
 */
void* get_host_pointer(Device d, GpuPtr ptr);

/**
 * @brief Get the texture size align object
 *
 * @param d
 * @param desc
 * @return TextureSizeAlign
 */
TextureSizeAlign get_texture_size_align(Device d, const TextureDesc& desc);

/**
 * @brief Create a texture object.
 * Create a new new texture with the provided descriptor. If location != 0, then the texture is
 * allocated at that location in Gpu memory. Note that textures can only be created in memory
 * allocated with Memory::Gpu - no host-visible textures can be created. If location == 0, memory is
 * allocated for the texture.
 * @param d
 * @param desc
 * @param location
 * @return Handle<Texture>
 */
Handle<Texture> create_texture(Device d, const TextureDesc& desc, GpuPtr location = 0);

Handle<TextureHeap> create_texture_heap(Device d, const TextureHeapDesc& desc);

/**
 * @brief
 *
 * @param d
 */
void free(Device d, Handle<Texture>);

/**
 * @brief
 *
 * @param d
 */
void free(Device d, Handle<TextureHeap>);

/**
 * @brief
 *
 * @param d
 * @param desc
 * @return TextureView
 */
TextureView add_texture_view_to_heap(Device d, Handle<TextureHeap>, const TextureViewDesc& desc);

TextureView add_rw_texture_view_to_heap(Device d, Handle<TextureHeap>, const TextureViewDesc& desc);

Sampler add_sampler_to_heap(Device d, Handle<TextureHeap>, const SamplerDesc& sampler);

void remove_rw_texture_view_from_heap(Device d, Handle<TextureHeap>, TextureView);

void remove_sampler_from_heap(Device d, Handle<TextureHeap>, Sampler);

/**
 * @brief
 *
 * @param d
 */
void remove_texture_view_from_heap(Device d, Handle<TextureHeap>, TextureView);


// State objects
/**
 * @brief Create a depth stencil state object
 *
 * @param d
 * @param desc
 * @return Handle<DepthStencilState>
 */
Handle<DepthStencilState> create_depth_stencil_state(Device d, const DepthStencilDesc& desc);

/**
 * @brief
 *
 * @param d
 * @param state
 */
void free_depth_stencil_state(Device d, Handle<DepthStencilState> state);

/// @}

/**
 * @defgroup Pipelines Pipelines
 *
 * @brief Create GPU pipelines from shader source and descriptors.
 *
 */
/// @{

/**
 * @brief Create a compute pipeline object
 *
 * @param d
 * @param compute
 * @param constants A list of specialization constants for the shaders
 * @return Handle<Pipeline>
 */
Handle<Pipeline> create_compute_pipeline(Device                             d,
                                         ShaderSource                       compute,
                                         Span<const SpecializationConstant> constants = {});

/**
 * @brief Create a graphics pipeline object
 *
 * @param d
 * @param vertex
 * @param fragment
 * @param desc
 * @param constants A list of specialization constants for the shaders
 * @return Handle<Pipeline>
 */
Handle<Pipeline> create_graphics_pipeline(Device                             d,
                                          ShaderSource                       vertex,
                                          ShaderSource                       fragment,
                                          const RasterDesc&                  desc,
                                          Span<const SpecializationConstant> constants = {});

/**
 * @brief
 *
 * @param d
 * @param pipeline
 */
void free(Device d, Handle<Pipeline> pipeline);

///@}

/**
 * @brief Create a semaphore object
 *
 * @param d
 * @param initValue
 * @return Handle<Semaphore>
 */
Handle<Semaphore> create_semaphore(Device d, uint64_t initValue);

/**
 * @brief Block the current thread until the semaphore value is greater or equal than the provided
 * value.
 *
 * @param d
 * @param sema
 * @param value
 */
void wait_semaphore(Device d, Handle<Semaphore> sema, uint64_t value);

/**
 * @brief Destroy the provided semaphore.
 *
 * @param d
 * @param sema
 */
void free(Device d, Handle<Semaphore> sema);

/**
 * @}
 *
 */

/**
 * @defgroup queue Queue
 * @brief A command queue for submitting work to the GPU.
 * Queues are used to record and submit commands to the GPU.
 * NOTE: Currently we only support a single queue per device. Support for multiple queues is
 * planned.
 * Commands submitted on different queues are not ordered with respect to each other, but
 * commands submitted to a specific queue will start in submission order (though may complete or
 * execute out of order). To synchronize between queues, use `Semaphores`. Note that generally
 * queues are not thread safe - functions on queues should only be called on one thread at a time.
 * The exception to this is `queue_start_command_recording()`, which can be safely called
 * concurrently by multiple threads.
 */
/// @{

/**
 * @brief Get a queue from the device.
 * Will do some initialization on the first call for a given `QueueType`, but subsequent calls will
 * be fast.
 *
 * @return The requested queue, or nullptr if the queue type is not supported on the hardware.
 */
Queue get_queue(Device d, QueueType type = QueueType::Default);

/**
 * @brief Get a command buffer for the queue.
 * Safe to call from multiple threads.
 */
CommandBuffer queue_start_command_recording(Queue q);

/**
 * @brief Submit a set of command buffers to the queue, synchronized with provided semaphores.
 *
 * @param q
 * @param command_buffers
 * @param wait_semaphores
 * @param signal_semaphores
 */
void queue_submit(Queue                     q,
                  Span<const CommandBuffer> command_buffers,
                  Span<const SemaphoreInfo> wait_semaphores   = {},
                  Span<const SemaphoreInfo> signal_semaphores = {});

/**
 * @brief
 *
 * @param q
 * @param command_buffers
 */
void queue_cancel(Queue q, Span<const Handle<CommandBuffer>> command_buffers);

/**
 * @brief Enqueue a callback function that will be called when any currently submitted work is
 * completed on this queue. Note that there are no background threads being created for this
 * callback - the fn will not be called until `queue_process_events()` is called.
 *
 * @param q
 * @param fn
 */
void queue_on_submitted_work_completed(Queue q, Function<void>&& fn);

/**
 * @brief Call any `queue_on_submitted_work_completed()` callbacks that are ready.
 *
 * @param q
 */
void queue_process_events(Queue q);

/// @}

/** @defgroup cmd Command Buffers
 * Command Buffers represent a packet of commands that will be executed on a GPU.
 * You get a command buffer by calling `queue_start_command_recording()`.
 * Command buffers can be recorded in parallel, but functions that operate on command buffers are
 * NOT synchronized - each command buffer should only be recorded by one thread at a time.
 * Command buffers must be submitted to the same queue they are started on. Before submission,
 * `cmd_finalize()` must be called.
 */

/// @{
/**
 * @brief Copy `size` bytes of memory from `src` to `dest`.
 *
 */
void cmd_memcpy(CommandBuffer cmd, GpuPtr destGpu, GpuPtr srcGpu, size_t size);

/**
 * @brief Copy from gpu memory into a texture object.
 *
 * @param cmd
 * @param src
 * @param texture
 * @param info
 */
void cmd_copy_to_texture(CommandBuffer                  cmd,
                         GpuPtr                         src,
                         Handle<Texture>                texture,
                         const BufferToTextureCopyInfo& info);

/**
 * @brief Copy from a texture to gpu memory.
 *
 * @param cmd
 * @param destGpu
 * @param srcGpu
 * @param texture
 */
void cmd_copy_from_texture(CommandBuffer   cmd,
                           GpuPtr          destGpu,
                           GpuPtr          srcGpu,
                           Handle<Texture> texture);

/**
 * @brief
 *
 * @param cmd
 * @param heap
 */
void cmd_set_texture_heap(CommandBuffer cmd, Handle<TextureHeap> heap);

/**
 * @brief
 *
 * @param cmd
 * @param before
 * @param after
 * @param image_transitions
 * @param hazards
 */
void cmd_barrier(CommandBuffer                 cmd,
                 StageFlags                    before,
                 StageFlags                    after,
                 Span<const TextureTransition> image_transitions = {},
                 HazardFlags                   hazards           = HazardFlags(0));

/**
 * @brief
 *
 * @param cmd
 * @param pipeline
 */
void cmd_set_pipeline(CommandBuffer cmd, Handle<Pipeline> pipeline);

/**
 * @brief
 *
 * @param cmd
 * @param state
 */
void cmd_set_depth_stencil_state(CommandBuffer cmd, Handle<DepthStencilState> state);

/**
 * @brief Set the current viewport
 *
 * @param cmd
 * @param rect
 */
void cmd_set_viewport(CommandBuffer cmd, const Rect2D& rect);

/**
 * @brief
 *
 * @param cmd
 * @param rect
 */
void cmd_set_scissor_rect(CommandBuffer cmd, const Rect2D& rect);

/**
 * @brief
 *
 * @param cmd
 * @param dataGpu
 * @param gridDimensions
 */
void cmd_dispatch(CommandBuffer cmd, GpuPtr dataGpu, const Dimension3D& gridDimensions);

/**
 * @brief
 *
 * @param cmd
 * @param dataGpu
 * @param gridDimensionsGpu
 */
void cmd_dispatch_indirect(CommandBuffer cmd, GpuPtr dataGpu, GpuPtr gridDimensionsGpu);

/**
 * @brief
 *
 * @param cmd
 * @param desc
 */
void cmd_begin_render_pass(CommandBuffer cmd, RenderPassDesc desc);

/**
 * @brief
 *
 * @param cmd
 */
void cmd_end_render_pass(CommandBuffer cmd);

/**
 * @brief Set front face winding direction.
 *
 */
void cmd_set_front_face(CommandBuffer cmd, FrontFace front);

/**
 * @brief Set backface culling mode
 *
 */
void cmd_set_cull_mode(CommandBuffer cmd, Cull cull);

/**
 * @brief
 *
 * @param cmd
 * @param vertexDataGpu
 * @param fragmentDataGpu
 * @param vertexCount
 * @param instanceCount
 */
void cmd_draw(CommandBuffer cmd,
              GpuPtr        vertexDataGpu,
              GpuPtr        fragmentDataGpu,
              uint32_t      vertexCount,
              uint32_t      instanceCount);

/**
 * @brief
 *
 * @param cmd
 * @param args
 */
void cmd_draw_indexed_instanced(CommandBuffer cmd, const DrawIndexedInstancedInfo& args);

/**
 * @brief
 *
 * @param cmd
 * @param args
 */
void cmd_draw_indexed_instanced_indirect(CommandBuffer cmd, const DrawIndexedIndirectInfo& args);

/**
 * @brief
 *
 * @param cmd
 * @param args
 */
void cmd_draw_indexed_instanced_indirect_multi(CommandBuffer                cmd,
                                               const MultiDrawIndirectInfo& args);

/**
 * @brief
 *
 * @param cmd
 */
void cmd_wait_for_surface_texture(CommandBuffer cmd);

/**
 * @brief
 *
 * @param cmd
 */
void cmd_signal_surface_texture(CommandBuffer cmd);

/**
 * @brief
 *
 * @param cmd
 */
void cmd_finalize(CommandBuffer cmd);
/// @}

extern template class Span<const char>;
extern template class Span<uint8_t>;
extern template class Span<const ColorTarget>;
extern template class Span<const RenderAttachment>;
extern template class Span<const Format>;
extern template class Span<const PresentMode>;
extern template class Span<const CommandBuffer>;
extern template class Span<const SemaphoreInfo>;
extern template class Span<const TextureTransition>;
extern template class Span<const SpecializationConstant>;
extern template class Function<void>;

}  // namespace loon::gpu
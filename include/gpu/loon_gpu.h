#pragma once
#include <cstddef>
#include <cstdint>

namespace loon::gpu {

template <class T> class Span {};
using ByteSpan = Span<uint8_t>;

struct StringView {
  const char *data;
  size_t length;
};

inline constexpr StringView operator""_sv(const char *val, size_t len) {
  return StringView{
      .data = val,
      .length = len,
  };
}

#define WGPU_STRCAT_IMPL(a, b) a##b
#define WGPU_STRCAT(a, b) WGPU_STRCAT_IMPL(a, b)
#define WGPU_FN_TYPEDEF(name) WGPU_STRCAT(WGPUProc, name)

// #define X_WGPU_FUNCTION(snake_name, pascal_name)                                                   \
//     extern WGPU_FN_TYPEDEF(pascal_name) WGPU_STRCAT(wgpu, pascal_name);
// WGPU_FUNCTION_LIST
// #undef X_WGPU_FUNCTION

// Extension structs, to get more control over the underlying implementation
typedef enum WGPULoonSType {
  WGPUSType_LoonInstanceConfiguration = 0x00420001,
  WGPUSType_LoonForce32 = 0x7FFFFFFF,
} WGPULoonSType;

typedef struct WGPULoonMemoryBlock {
  void *ptr;
  uint32_t len;
} WGPULoonMemoryBlock;

typedef enum WGPULoonLogLevel {
  WGPULoonLogLevel_Off = 0x00000000,
  WGPULoonLogLevel_Error = 0x00000001,
  WGPULoonLogLevel_Warning = 0x00000002,
  WGPULoonLogLevel_Info = 0x00000003,
  WGPULoonLogLevel_Debug = 0x00000004,
  WGPULoonLogLevel_Force32 = 0x7FFFFFFF,
} WGPULoonLogLevel;

// Custom allocation callback - essentially a realloc function but not exactly
// the same as the C version.
// - ptr is null iff old_size is 0
// - When new_size != 0 and old_size == 0, should behave like malloc
// - If new_size == 0, the function must return null; if old_size != 0, it
// should free the block pointed to by ptr. It is the responsibility of the
// function to copy old_size bytes of memory from ptr to the returned pointer.
typedef WGPULoonMemoryBlock (*WGPULoonProcAllocatorCallback)(void *userdata,
                                                             void *ptr,
                                                             uint32_t old_size,
                                                             uint32_t new_size);

typedef void (*WGPULoonProcLogCallback)(WGPULoonLogLevel lvl,
                                        StringView message, void *userdata);

// Chained to InstanceDescriptor to set memory allocation functions that will be
// used for all allocations in the instance.
//
typedef struct WGPULoonInstanceConfiguration {
  WGPULoonProcAllocatorCallback alloc;
  void *alloc_userdata;

  WGPULoonLogLevel log_level;
  WGPULoonProcLogCallback log;
  void *log_userdata;
} WGPULoonInstanceConfiguration;

// Opaque handles
struct Device;
struct GpuPipeline;
struct GpuBuffer;
struct GpuTexture;
struct GpuDepthStencilState;
struct GpuBlendState;
struct GpuQueue;
struct GpuCommandBuffer;
struct GpuSemaphore;

using GpuPtr = uint64_t;

// Enums
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
enum BLEND {
  BLEND_ADD,
  BLEND_SUBTRACT,
  BLEND_REV_SUBTRACT,
  BLEND_MIN,
  BLEND_MAX
};
enum FACTOR {
  FACTOR_ZERO,
  FACTOR_ONE,
  FACTOR_SRC_COLOR,
  FACTOR_DST_COLOR,
  FACTOR_SRC_ALPHA,
};

enum TOPOLOGY {
  TOPOLOGY_TRIANGLE_LIST,
  TOPOLOGY_TRIANGLE_STRIP,
  TOPOLOGY_TRIANGLE_FAN
};

enum TEXTURE {
  TEXTURE_1D,
  TEXTURE_2D,
  TEXTURE_3D,
  TEXTURE_CUBE,
  TEXTURE_2D_ARRAY,
  TEXTURE_CUBE_ARRAY
};

enum FORMAT {
  FORMAT_NONE,
  FORMAT_RGBA8_UNORM,
  FORMAT_D32_FLOAT,
  FORMAT_RG11B10_FLOAT,
  FORMAT_RGB10_A2_UNORM,
};

enum USAGE_FLAGS {
  USAGE_NONE = 0,
  USAGE_SAMPLED,
  USAGE_STORAGE,
  USAGE_COLOR_ATTACHMENT,
  USAGE_DEPTH_STENCIL_ATTACHMENT,
};

enum STAGE {
  STAGE_TRANSFER,
  STAGE_COMPUTE,
  STAGE_RASTER_COLOR_OUT,
  STAGE_PIXEL_SHADER,
  STAGE_VERTEX_SHADER,

};

enum HAZARD_FLAGS {
  HAZARD_DRAW_ARGUMENTS = 0x1,
  HAZARD_DESCRIPTORS = 0x2,
  HAZARD_DEPTH_STENCIL = 0x4
};

enum SIGNAL {
  SIGNAL_ATOMIC_SET,
  SIGNAL_ATOMIC_MAX,
  SIGNAL_ATOMIC_OR,
};

// Structs
struct Dimension3D {
  uint32_t x, y, z;
};

struct Stencil {
  OP test = OP_ALWAYS;
  OP failOp = OP_KEEP;
  OP passOp = OP_KEEP;
  OP depthFailOp = OP_KEEP;
  uint8_t reference = 0;
};

struct GpuDepthStencilDesc {
  DEPTH_FLAGS depthMode = DEPTH_NONE;
  OP depthTest = OP_ALWAYS;
  float depthBias = 0.0f;
  float depthBiasSlopeFactor = 0.0f;
  float depthBiasClamp = 0.0f;
  uint8_t stencilReadMask = 0xff;
  uint8_t stencilWriteMask = 0xff;
  Stencil stencilFront;
  Stencil stencilBack;
};

struct GpuBlendDesc {
  BLEND colorOp = BLEND_ADD;
  FACTOR srcColorFactor = FACTOR_ONE;
  FACTOR dstColorFactor = FACTOR_ZERO;
  BLEND alphaOp = BLEND_ADD;
  FACTOR srcAlphaFactor = FACTOR_ONE;
  FACTOR dstAlphaFactor = FACTOR_ZERO;
  uint8_t colorWriteMask = 0xf;
};

struct ColorTarget {
  FORMAT format = FORMAT_NONE;
  uint8_t writeMask = 0xf;
};

struct GpuRasterDesc {
  TOPOLOGY topology = TOPOLOGY_TRIANGLE_LIST;
  CULL cull = CULL_NONE;
  bool alphaToCoverage = false;
  bool supportDualSourceBlending = false;
  uint8_t sampleCount = 1;
  FORMAT depthFormat = FORMAT_NONE;
  FORMAT stencilFormat = FORMAT_NONE;
  Span<ColorTarget> colorTargets = {};
  GpuBlendDesc *blendstate = nullptr; // optional embedded blend state
};

struct GpuRenderPassDesc {};

struct GpuTextureDesc {
  TEXTURE type = TEXTURE_2D;
  Dimension3D dimensions;
  uint32_t mipCount = 1;
  uint32_t layerCount = 1;
  uint32_t sampleCount = 1;
  FORMAT format = FORMAT_NONE;
  USAGE_FLAGS usage = USAGE_NONE;
};

struct GpuViewDesc {
  FORMAT format = FORMAT_NONE;
  uint8_t baseMip = 0;
  uint8_t mipCount = 0;
  uint16_t baseLayer = 0;
  uint16_t layerCount = 0;
};

struct GpuTextureSizeAlign {
  size_t size;
  size_t align;
};

struct GpuTextureDescriptor {
  uint64_t data[4];
};

// Initialization
Device getDefaultDevice();

class Device {
public:
  // Buffers:
  GpuBuffer malloc(size_t bytes, MEMORY memory = MEMORY_DEFAULT);
  GpuBuffer malloc(size_t bytes, size_t align, MEMORY memory = MEMORY_DEFAULT);
  void free(GpuBuffer buffer);
  GpuPtr getDevicePointer(GpuBuffer buffer);

  // Textures:
  GpuTexture createTexture(const GpuTextureDesc &desc);
  GpuTextureDescriptor textureViewDescriptor(GpuTexture texture,
                                             GpuViewDesc desc);
  GpuTextureDescriptor RWTextureViewDescriptor(GpuTexture texture,
                                               GpuViewDesc desc);
  void free(GpuTexture);

  // Pipelines
  GpuPipeline createComputePipeline(ByteSpan computeIR);
  GpuPipeline createGraphicsPipeline(ByteSpan vertexIR, ByteSpan pixelIR,
                                     GpuRasterDesc desc);
  GpuPipeline createGraphicsMeshletPipeline(ByteSpan meshletIR,
                                            ByteSpan pixelIR,
                                            GpuRasterDesc desc);
  void freePipeline(GpuPipeline pipeline);

  // State objects
  GpuDepthStencilState createDepthStencilState(GpuDepthStencilDesc desc);
  GpuBlendState createBlendState(GpuBlendDesc desc);
  void freeDepthStencilState(GpuDepthStencilState state);
  void freeBlendState(GpuBlendState state);

  // Queue
  GpuQueue createQueue(/* DEVICE & QUEUE CREATION DETAILS OMITTED */);
  GpuCommandBuffer startCommandRecording(GpuQueue queue);
  void submit(GpuQueue queue, Span<GpuCommandBuffer> commandBuffers);
  void cancel(GpuQueue queue, Span<GpuCommandBuffer> commandBuffers);

  // Semaphores
  GpuSemaphore createSemaphore(uint64_t initValue);
  void waitSemaphore(GpuSemaphore sema, uint64_t value);
  void destroySemaphore(GpuSemaphore sema);

private:
  struct Impl;
  Impl *impl = nullptr;

  void chk(uint64_t result);
};

class CommandBuffer {
public:
  // Commands
  void memcpy(GpuCommandBuffer cb, GpuPtr destGpu, GpuPtr srcGpu, size_t size);
  void copyToTexture(GpuCommandBuffer cb, GpuPtr destGpu, GpuPtr srcGpu,
                     GpuTexture texture);
  void copyFromTexture(GpuCommandBuffer cb, GpuPtr destGpu, GpuPtr srcGpu,
                       GpuTexture texture);

  void setActiveTextureHeapPtr(GpuCommandBuffer cb, GpuPtr ptrGpu);

  void barrier(GpuCommandBuffer cb, STAGE before, STAGE after,
               HAZARD_FLAGS hazards = HAZARD_FLAGS(0));

#if 0
// NOTE: Not sure this is implementable on top of vulkan right now. 
// Vulkan Events aren't sophisticated enough to do this API, but we could probably come up with a simpler one
void gpuSignalAfter(GpuCommandBuffer cb, STAGE before, GpuPtr ptrGpu,
                    uint64_t value, SIGNAL signal);
void gpuWaitBefore(GpuCommandBuffer cb, STAGE after, GpuPtr ptrGpu,
                   uint64_t value, OP op,
                   HAZARD_FLAGS hazards = HAZARD_FLAGS(0), uint64_t mask = ~0);
#endif

  void setPipeline(GpuCommandBuffer cb, GpuPipeline pipeline);
  void setDepthStencilState(GpuCommandBuffer cb, GpuDepthStencilState state);
  void setBlendState(GpuCommandBuffer cb, GpuBlendState state);

  void dispatch(GpuCommandBuffer cb, GpuPtr dataGpu,
                const Dimension3D &gridDimensions);
  void dispatchIndirect(GpuCommandBuffer cb, GpuPtr dataGpu,
                        GpuPtr gridDimensionsGpu);

  void beginRenderPass(GpuCommandBuffer cb, GpuRenderPassDesc desc);
  void endRenderPass(GpuCommandBuffer cb);

  void drawIndexedInstanced(GpuCommandBuffer cb, GpuPtr vertexDataGpu,
                            GpuPtr pixelDataGpu, GpuPtr indicesGpu,
                            uint32_t indexCount, uint32_t instanceCount);
  void drawIndexedInstancedIndirect(GpuCommandBuffer cb, GpuPtr vertexDataGpu,
                                    GpuPtr pixelDataGpu, GpuPtr indicesGpu,
                                    GpuPtr argsGpu);
  void drawIndexedInstancedIndirectMulti(GpuCommandBuffer cb, GpuPtr dataVxGpu,
                                         uint32_t vxStride, GpuPtr dataPxGpu,
                                         uint32_t pxStride, GpuPtr argsGpu,
                                         GpuPtr drawCountGpu);

  void drawMeshlets(GpuCommandBuffer cb, GpuPtr meshletDataGpu,
                    GpuPtr pixelDataGpu, const Dimension3D &dim);
  void drawMeshletsIndirect(GpuCommandBuffer cb, GpuPtr meshletDataGpu,
                            GpuPtr pixelDataGpu, GpuPtr dimGpu);

private:
  struct Impl;
  Impl *impl;
};

} // namespace loon::gpu
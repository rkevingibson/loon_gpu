#include "validation.h"

#include "device.h"
#include "objects.h"

struct TextureFormatInfo {
    WGPUTextureFormat format;
    bool              render_attachment;
    bool              blendable;
};

static constexpr TextureFormatInfo kTextureFormatInfo[] = {
    {.format = WGPUTextureFormat_Undefined, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_R8Unorm, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_R8Snorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_R8Uint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_R8Sint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_R16Unorm, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_R16Snorm, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_R16Uint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_R16Sint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_R16Float, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RG8Unorm, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RG8Snorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_RG8Uint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RG8Sint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_R32Float, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_R32Uint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_R32Sint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RG16Unorm, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RG16Snorm, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RG16Uint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RG16Sint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RG16Float, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA8Unorm, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA8UnormSrgb, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA8Snorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_RGBA8Uint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA8Sint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_BGRA8Unorm, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_BGRA8UnormSrgb, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGB10A2Uint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGB10A2Unorm, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RG11B10Ufloat, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_RGB9E5Ufloat, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_RG32Float, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RG32Uint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RG32Sint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA16Unorm, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA16Snorm, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA16Uint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA16Sint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA16Float, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA32Float, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA32Uint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_RGBA32Sint, .render_attachment = true, .blendable = false},
    {.format = WGPUTextureFormat_Stencil8, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_Depth16Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_Depth24Plus, .render_attachment = false, .blendable = false},
    {.format            = WGPUTextureFormat_Depth24PlusStencil8,
     .render_attachment = false,
     .blendable         = false},
    {.format = WGPUTextureFormat_Depth32Float, .render_attachment = false, .blendable = false},
    {.format            = WGPUTextureFormat_Depth32FloatStencil8,
     .render_attachment = false,
     .blendable         = false},
    {.format = WGPUTextureFormat_BC1RGBAUnorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC1RGBAUnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC2RGBAUnorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC2RGBAUnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC3RGBAUnorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC3RGBAUnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC4RUnorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC4RSnorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC5RGUnorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC5RGSnorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC6HRGBUfloat, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC6HRGBFloat, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC7RGBAUnorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_BC7RGBAUnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ETC2RGB8Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ETC2RGB8UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ETC2RGB8A1Unorm, .render_attachment = false, .blendable = false},
    {.format            = WGPUTextureFormat_ETC2RGB8A1UnormSrgb,
     .render_attachment = false,
     .blendable         = false},
    {.format = WGPUTextureFormat_ETC2RGBA8Unorm, .render_attachment = false, .blendable = false},
    {.format            = WGPUTextureFormat_ETC2RGBA8UnormSrgb,
     .render_attachment = false,
     .blendable         = false},
    {.format = WGPUTextureFormat_EACR11Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_EACR11Snorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_EACRG11Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_EACRG11Snorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC4x4Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC4x4UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC5x4Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC5x4UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC5x5Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC5x5UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC6x5Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC6x5UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC6x6Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC6x6UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC8x5Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC8x5UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC8x6Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC8x6UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC8x8Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC8x8UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC10x5Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC10x5UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC10x6Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC10x6UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC10x8Unorm, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC10x8UnormSrgb, .render_attachment = false, .blendable = false},
    {.format = WGPUTextureFormat_ASTC10x10Unorm, .render_attachment = false, .blendable = false},
    {.format            = WGPUTextureFormat_ASTC10x10UnormSrgb,
     .render_attachment = false,
     .blendable         = false},
    {.format = WGPUTextureFormat_ASTC12x10Unorm, .render_attachment = false, .blendable = false},
    {.format            = WGPUTextureFormat_ASTC12x10UnormSrgb,
     .render_attachment = false,
     .blendable         = false},
    {.format = WGPUTextureFormat_ASTC12x12Unorm, .render_attachment = false, .blendable = false},
    {.format            = WGPUTextureFormat_ASTC12x12UnormSrgb,
     .render_attachment = false,
     .blendable         = false},
};
static constexpr size_t kNumTextureFormatInfos
    = sizeof(kTextureFormatInfo) / sizeof(TextureFormatInfo);
static constexpr auto checkFormatInfos = []() -> size_t {
    for (size_t i = 0; i < kNumTextureFormatInfos; ++i) {
        if (kTextureFormatInfo[i].format != i) { return i; }
    }
    return ~0;
};
static_assert(checkFormatInfos() == ~0);

namespace webgpu {


bool validate(WGPUSurface surface, WGPUSurfaceConfiguration const* config) {
    // TODO: Validation following https://webgpu-native.github.io/webgpu-headers/Surfaces.html
    (void)surface;
    (void)config;
    return true;
}


bool validate(WGPUSurface surface, WGPUAdapter adapter, WGPUSurfaceCapabilities* capabilities) {
    if (capabilities == nullptr || capabilities->nextInChain != nullptr) {
        // Unknown nextInChain - we don't support any for this struct
        return false;
    }

    if (surface->instance != adapter->instance) { return false; }
    return true;
}

bool validate(WGPUDevice device, WGPURenderPipelineDescriptor const* descriptor) {
    if (descriptor == nullptr) {
        device->error(WGPUErrorType_Validation,
                      "Invalid argument: null WGPURenderPipelineDescriptor used"_wsv);
        return false;
    }

    const size_t vertex_buffer_count = descriptor->vertex.bufferCount;
    if (vertex_buffer_count > kMaxVertexBuffers) {
        device->error(WGPUErrorType_Validation,
                      "Too many vertex buffers in wgpuDeviceCreateRenderPipeline"_wsv);
        return false;
    }

    // Webgpu spec says we only support 1 or 4x multisampling.
    if (descriptor->multisample.count != 1 && descriptor->multisample.count != 4) {
        device->error(WGPUErrorType_Validation,
                      "WebGPU only supports multisample counts of 1 or 4."_wsv);
        return false;
    }

    // https://www.w3.org/TR/webgpu/#fragment-state
    if (descriptor->fragment) {
        const auto& fragment = *descriptor->fragment;
        if (fragment.targetCount > kMaxColorAttachments) {
            device->error(WGPUErrorType_Validation,
                          "Invalid GPUFragmentState: too many color attachments"_wsv);
            return false;
        }

        for (uint32_t index = 0; index < fragment.targetCount; ++index) {
            const auto& colorState = fragment.targets[index];

            if (colorState.format >= kNumTextureFormatInfos
                || !kTextureFormatInfo[colorState.format].render_attachment) {
                device->error(WGPUErrorType_Validation,
                              "Invalid GPUFragmentState: color format is not renderable"_wsv);
                return false;
            }

            if (colorState.writeMask > WGPUColorWriteMask_All) {
                device->error(WGPUErrorType_Validation,
                              "Invalid GPUFragmentState: invalid color write mask"_wsv);
                return false;
            }

            // TODO: More validation here, for blending, etc.
        }
    }

    return true;
}

bool validate(WGPUDevice device, WGPUPipelineLayoutDescriptor const* descriptor) {
    (void)device;
    (void)descriptor;
    return true;
}

bool validate(WGPUDevice device, WGPUBindGroupLayoutDescriptor const* descriptor) {
    (void)device;
    (void)descriptor;
    return true;
}

bool validate(WGPUDevice device, WGPUBufferDescriptor const* descriptor) {
    if (device->is_lost()) {
        device->error(WGPUErrorType_Validation, "Device lost"_wsv);
        return false;
    }

    static constexpr WGPUFlags kValidBufferUsageMask
        = WGPUBufferUsage_MapRead | WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc
          | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index | WGPUBufferUsage_Vertex
          | WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect
          | WGPUBufferUsage_QueryResolve;
    const auto usage = descriptor->usage;
    if (usage == 0 || (usage & ~kValidBufferUsageMask) != 0) {
        device->error(WGPUErrorType_Validation, "WGPUBufferDescriptor - invalid usage"_wsv);
        return false;
    }

    if ((usage & WGPUBufferUsage_MapRead)) {
        // Usage must have nothing else but copy_dst.
        static constexpr WGPUFlags kMapReadMask = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        if ((usage & WGPUBufferUsage_CopyDst) == 0 || (usage & ~kMapReadMask) != 0) {
            device->error(
                WGPUErrorType_Validation,
                "WGPUBufferDescriptor - invalid usage, MAP_READ must have nothing else but COPY_DST"_wsv);
            return false;
        }
    }

    if ((usage & WGPUBufferUsage_MapWrite)) {
        // Usage must have nothing else but copy_dst.
        static constexpr WGPUFlags kMapWriteMask
            = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite;
        if ((usage & WGPUBufferUsage_CopySrc) == 0 || (usage & ~kMapWriteMask) != 0) {
            device->error(
                WGPUErrorType_Validation,
                "WGPUBufferDescriptor - invalid usage, MAP_WRITE must have nothing else but COPY_SRC"_wsv);
            return false;
        }
    }

    // TODO: Check against maxBufferSize

    return true;
}

bool validate(WGPUCommandEncoder encoder, WGPURenderPassDescriptor const* descriptor) {
    (void)encoder;
    (void)descriptor;
    return true;
}

bool validate(WGPURenderPassEncoder renderPassEncoder) {
    (void)renderPassEncoder;
    return true;
}

bool validate(WGPURenderPassEncoder renderPassEncoder, WGPURenderPipeline pipeline) {
    (void)renderPassEncoder;
    (void)pipeline;
    return true;
}

bool validate_draw(WGPURenderPassEncoder renderPassEncoder,
                   uint32_t              vertexCount,
                   uint32_t              instanceCount,
                   uint32_t              firstVertex,
                   uint32_t              firstInstance) {
    (void)renderPassEncoder;
    (void)vertexCount;
    (void)instanceCount;
    (void)firstVertex;
    (void)firstInstance;
    return true;
}


bool validate_buffer_get_mapped_range(WGPUBuffer buffer, size_t offset, size_t size) {
    auto device = buffer->device;
    if (buffer->mapping.ptr == nullptr) {
        device->error(WGPUErrorType_Validation, "Attempting to access an unmapped buffer"_wsv);
        return false;
    }
    if (offset % 8 != 0 || size % 4 != 0) {
        device->error(WGPUErrorType_Validation, "Invalid offset or size alignment."_wsv);
        return false;
    }
    if (offset < buffer->mapping.offset
        || offset + size > buffer->mapping.offset + buffer->mapping.size) {
        device->error(WGPUErrorType_Validation,
                      "Attempting to read buffer outside of mapped range"_wsv);
        return false;
    }

    return true;
}

}  // namespace webgpu
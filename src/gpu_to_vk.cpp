#include "gpu_to_vk.h"

#include "gpu/loon_gpu.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan_core.h"

namespace loon::gpu {


VkFormat bridge(Format format) {
    switch (format) {
        case Format::None: return VK_FORMAT_UNDEFINED;
        case Format::R8Unorm: return VK_FORMAT_R8_UNORM;
        case Format::R8Snorm: return VK_FORMAT_R8_SNORM;
        case Format::R8Uint: return VK_FORMAT_R8_UINT;
        case Format::R8Sint: return VK_FORMAT_R8_SINT;
        case Format::R16Uint: return VK_FORMAT_R16_UINT;
        case Format::R16Sint: return VK_FORMAT_R16_SINT;
        case Format::R16Float: return VK_FORMAT_R16_SFLOAT;
        case Format::RG8Unorm: return VK_FORMAT_R8G8_UNORM;
        case Format::RG8Snorm: return VK_FORMAT_R8G8_SNORM;
        case Format::RG8Uint: return VK_FORMAT_R8G8_UINT;
        case Format::RG8Sint: return VK_FORMAT_R8G8_SINT;
        case Format::R32Float: return VK_FORMAT_R32_SFLOAT;
        case Format::R32Uint: return VK_FORMAT_R32_UINT;
        case Format::R32Sint: return VK_FORMAT_R32_SINT;
        case Format::RG16Uint: return VK_FORMAT_R16G16_UINT;
        case Format::RG16Sint: return VK_FORMAT_R16G16_SINT;
        case Format::RG16Float: return VK_FORMAT_R16G16_SFLOAT;
        case Format::RGBA8Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::RGBA8UnormSrgb: return VK_FORMAT_R8G8B8A8_SRGB;
        case Format::RGBA8Snorm: return VK_FORMAT_R8G8B8A8_SNORM;
        case Format::RGBA8Uint: return VK_FORMAT_R8G8B8A8_UINT;
        case Format::RGBA8Sint: return VK_FORMAT_R8G8B8A8_SINT;
        case Format::BGRA8Unorm: return VK_FORMAT_B8G8R8A8_UNORM;
        case Format::BGRA8UnormSrgb: return VK_FORMAT_B8G8R8A8_SRGB;
        case Format::RGB10A2Uint: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
        case Format::RGB10A2Unorm: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case Format::RG11B10Ufloat: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case Format::RGB9E5Ufloat: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
        case Format::RG32Float: return VK_FORMAT_R32G32_SFLOAT;
        case Format::RG32Uint: return VK_FORMAT_R32G32_UINT;
        case Format::RG32Sint: return VK_FORMAT_R32G32_SINT;
        case Format::RGBA16Uint: return VK_FORMAT_R16G16B16A16_UINT;
        case Format::RGBA16Sint: return VK_FORMAT_R16G16B16A16_SINT;
        case Format::RGBA16Float: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case Format::RGBA32Float: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case Format::RGBA32Uint: return VK_FORMAT_R32G32B32A32_UINT;
        case Format::RGBA32Sint: return VK_FORMAT_R32G32B32A32_SINT;
        case Format::Stencil8: return VK_FORMAT_S8_UINT;
        case Format::Depth16Unorm: return VK_FORMAT_D16_UNORM;
        case Format::Depth24Plus: return VK_FORMAT_X8_D24_UNORM_PACK32;
        case Format::Depth24PlusStencil8: return VK_FORMAT_D24_UNORM_S8_UINT;
        case Format::Depth32Float: return VK_FORMAT_D32_SFLOAT;
        case Format::Depth32FloatStencil8: return VK_FORMAT_D32_SFLOAT_S8_UINT;
        case Format::BC1RGBAUnorm: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case Format::BC1RGBAUnormSrgb: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
        case Format::BC2RGBAUnorm: return VK_FORMAT_BC2_UNORM_BLOCK;
        case Format::BC2RGBAUnormSrgb: return VK_FORMAT_BC2_SRGB_BLOCK;
        case Format::BC3RGBAUnorm: return VK_FORMAT_BC3_UNORM_BLOCK;
        case Format::BC3RGBAUnormSrgb: return VK_FORMAT_BC3_SRGB_BLOCK;
        case Format::BC4RUnorm: return VK_FORMAT_BC4_UNORM_BLOCK;
        case Format::BC4RSnorm: return VK_FORMAT_BC4_SNORM_BLOCK;
        case Format::BC5RGUnorm: return VK_FORMAT_BC5_UNORM_BLOCK;
        case Format::BC5RGSnorm: return VK_FORMAT_BC5_SNORM_BLOCK;
        case Format::BC6HRGBUfloat: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        case Format::BC6HRGBFloat: return VK_FORMAT_BC6H_SFLOAT_BLOCK;
        case Format::BC7RGBAUnorm: return VK_FORMAT_BC7_UNORM_BLOCK;
        case Format::BC7RGBAUnormSrgb: return VK_FORMAT_BC7_SRGB_BLOCK;
        case Format::ETC2RGB8Unorm: return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
        case Format::ETC2RGB8UnormSrgb: return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
        case Format::ETC2RGB8A1Unorm: return VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;
        case Format::ETC2RGB8A1UnormSrgb: return VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;
        case Format::ETC2RGBA8Unorm: return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
        case Format::ETC2RGBA8UnormSrgb: return VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
        case Format::EACR11Unorm: return VK_FORMAT_EAC_R11_UNORM_BLOCK;
        case Format::EACR11Snorm: return VK_FORMAT_EAC_R11_SNORM_BLOCK;
        case Format::EACRG11Unorm: return VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
        case Format::EACRG11Snorm: return VK_FORMAT_EAC_R11G11_SNORM_BLOCK;
        case Format::ASTC4x4Unorm: return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
        case Format::ASTC4x4UnormSrgb: return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
        case Format::ASTC5x4Unorm: return VK_FORMAT_ASTC_5x4_UNORM_BLOCK;
        case Format::ASTC5x4UnormSrgb: return VK_FORMAT_ASTC_5x4_SRGB_BLOCK;
        case Format::ASTC5x5Unorm: return VK_FORMAT_ASTC_5x5_UNORM_BLOCK;
        case Format::ASTC5x5UnormSrgb: return VK_FORMAT_ASTC_5x5_SRGB_BLOCK;
        case Format::ASTC6x5Unorm: return VK_FORMAT_ASTC_6x5_UNORM_BLOCK;
        case Format::ASTC6x5UnormSrgb: return VK_FORMAT_ASTC_6x5_SRGB_BLOCK;
        case Format::ASTC6x6Unorm: return VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
        case Format::ASTC6x6UnormSrgb: return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;
        case Format::ASTC8x5Unorm: return VK_FORMAT_ASTC_8x5_UNORM_BLOCK;
        case Format::ASTC8x5UnormSrgb: return VK_FORMAT_ASTC_8x5_SRGB_BLOCK;
        case Format::ASTC8x6Unorm: return VK_FORMAT_ASTC_8x6_UNORM_BLOCK;
        case Format::ASTC8x6UnormSrgb: return VK_FORMAT_ASTC_8x6_SRGB_BLOCK;
        case Format::ASTC8x8Unorm: return VK_FORMAT_ASTC_8x8_UNORM_BLOCK;
        case Format::ASTC8x8UnormSrgb: return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
        case Format::ASTC10x5Unorm: return VK_FORMAT_ASTC_10x5_UNORM_BLOCK;
        case Format::ASTC10x5UnormSrgb: return VK_FORMAT_ASTC_10x5_SRGB_BLOCK;
        case Format::ASTC10x6Unorm: return VK_FORMAT_ASTC_10x6_UNORM_BLOCK;
        case Format::ASTC10x6UnormSrgb: return VK_FORMAT_ASTC_10x6_SRGB_BLOCK;
        case Format::ASTC10x8Unorm: return VK_FORMAT_ASTC_10x8_UNORM_BLOCK;
        case Format::ASTC10x8UnormSrgb: return VK_FORMAT_ASTC_10x8_SRGB_BLOCK;
        case Format::ASTC10x10Unorm: return VK_FORMAT_ASTC_10x10_UNORM_BLOCK;
        case Format::ASTC10x10UnormSrgb: return VK_FORMAT_ASTC_10x10_SRGB_BLOCK;
        case Format::ASTC12x10Unorm: return VK_FORMAT_ASTC_12x10_UNORM_BLOCK;
        case Format::ASTC12x10UnormSrgb: return VK_FORMAT_ASTC_12x10_SRGB_BLOCK;
        case Format::ASTC12x12Unorm: return VK_FORMAT_ASTC_12x12_UNORM_BLOCK;
        case Format::ASTC12x12UnormSrgb: return VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
        default: return VK_FORMAT_MAX_ENUM;
    }
}

VkImageAspectFlags aspects_for_format(Format format) {
    switch (format) {
        case Format::Stencil8: return VK_IMAGE_ASPECT_STENCIL_BIT;
        case Format::Depth16Unorm: return VK_IMAGE_ASPECT_DEPTH_BIT;
        case Format::Depth24Plus: return VK_IMAGE_ASPECT_DEPTH_BIT;
        case Format::Depth24PlusStencil8:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        case Format::Depth32Float: return VK_IMAGE_ASPECT_DEPTH_BIT;
        case Format::Depth32FloatStencil8:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        default: return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

Format bridge(VkFormat format) {
    switch (format) {
        case VK_FORMAT_UNDEFINED: return Format::None;
        case VK_FORMAT_R8_UNORM: return Format::R8Unorm;
        case VK_FORMAT_R8_SNORM: return Format::R8Snorm;
        case VK_FORMAT_R8_UINT: return Format::R8Uint;
        case VK_FORMAT_R8_SINT: return Format::R8Sint;
        case VK_FORMAT_R16_UINT: return Format::R16Uint;
        case VK_FORMAT_R16_SINT: return Format::R16Sint;
        case VK_FORMAT_R16_SFLOAT: return Format::R16Float;
        case VK_FORMAT_R8G8_UNORM: return Format::RG8Unorm;
        case VK_FORMAT_R8G8_SNORM: return Format::RG8Snorm;
        case VK_FORMAT_R8G8_UINT: return Format::RG8Uint;
        case VK_FORMAT_R8G8_SINT: return Format::RG8Sint;
        case VK_FORMAT_R32_SFLOAT: return Format::R32Float;
        case VK_FORMAT_R32_UINT: return Format::R32Uint;
        case VK_FORMAT_R32_SINT: return Format::R32Sint;
        case VK_FORMAT_R16G16_UINT: return Format::RG16Uint;
        case VK_FORMAT_R16G16_SINT: return Format::RG16Sint;
        case VK_FORMAT_R16G16_SFLOAT: return Format::RG16Float;
        case VK_FORMAT_R8G8B8A8_UNORM: return Format::RGBA8Unorm;
        case VK_FORMAT_R8G8B8A8_SRGB: return Format::RGBA8UnormSrgb;
        case VK_FORMAT_R8G8B8A8_SNORM: return Format::RGBA8Snorm;
        case VK_FORMAT_R8G8B8A8_UINT: return Format::RGBA8Uint;
        case VK_FORMAT_R8G8B8A8_SINT: return Format::RGBA8Sint;
        case VK_FORMAT_B8G8R8A8_UNORM: return Format::BGRA8Unorm;
        case VK_FORMAT_B8G8R8A8_SRGB: return Format::BGRA8UnormSrgb;
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: return Format::RGB10A2Uint;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return Format::RGB10A2Unorm;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return Format::RG11B10Ufloat;
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return Format::RGB9E5Ufloat;
        case VK_FORMAT_R32G32_SFLOAT: return Format::RG32Float;
        case VK_FORMAT_R32G32_UINT: return Format::RG32Uint;
        case VK_FORMAT_R32G32_SINT: return Format::RG32Sint;
        case VK_FORMAT_R16G16B16A16_UINT: return Format::RGBA16Uint;
        case VK_FORMAT_R16G16B16A16_SINT: return Format::RGBA16Sint;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return Format::RGBA16Float;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return Format::RGBA32Float;
        case VK_FORMAT_R32G32B32A32_UINT: return Format::RGBA32Uint;
        case VK_FORMAT_R32G32B32A32_SINT: return Format::RGBA32Sint;
        case VK_FORMAT_S8_UINT: return Format::Stencil8;
        case VK_FORMAT_D16_UNORM: return Format::Depth16Unorm;
        case VK_FORMAT_X8_D24_UNORM_PACK32: return Format::Depth24Plus;
        case VK_FORMAT_D24_UNORM_S8_UINT: return Format::Depth24PlusStencil8;
        case VK_FORMAT_D32_SFLOAT: return Format::Depth32Float;
        case VK_FORMAT_D32_SFLOAT_S8_UINT: return Format::Depth32FloatStencil8;
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return Format::BC1RGBAUnorm;
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return Format::BC1RGBAUnormSrgb;
        case VK_FORMAT_BC2_UNORM_BLOCK: return Format::BC2RGBAUnorm;
        case VK_FORMAT_BC2_SRGB_BLOCK: return Format::BC2RGBAUnormSrgb;
        case VK_FORMAT_BC3_UNORM_BLOCK: return Format::BC3RGBAUnorm;
        case VK_FORMAT_BC3_SRGB_BLOCK: return Format::BC3RGBAUnormSrgb;
        case VK_FORMAT_BC4_UNORM_BLOCK: return Format::BC4RUnorm;
        case VK_FORMAT_BC4_SNORM_BLOCK: return Format::BC4RSnorm;
        case VK_FORMAT_BC5_UNORM_BLOCK: return Format::BC5RGUnorm;
        case VK_FORMAT_BC5_SNORM_BLOCK: return Format::BC5RGSnorm;
        case VK_FORMAT_BC6H_UFLOAT_BLOCK: return Format::BC6HRGBUfloat;
        case VK_FORMAT_BC6H_SFLOAT_BLOCK: return Format::BC6HRGBFloat;
        case VK_FORMAT_BC7_UNORM_BLOCK: return Format::BC7RGBAUnorm;
        case VK_FORMAT_BC7_SRGB_BLOCK: return Format::BC7RGBAUnormSrgb;
        case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK: return Format::ETC2RGB8Unorm;
        case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK: return Format::ETC2RGB8UnormSrgb;
        case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK: return Format::ETC2RGB8A1Unorm;
        case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK: return Format::ETC2RGB8A1UnormSrgb;
        case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK: return Format::ETC2RGBA8Unorm;
        case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK: return Format::ETC2RGBA8UnormSrgb;
        case VK_FORMAT_EAC_R11_UNORM_BLOCK: return Format::EACR11Unorm;
        case VK_FORMAT_EAC_R11_SNORM_BLOCK: return Format::EACR11Snorm;
        case VK_FORMAT_EAC_R11G11_UNORM_BLOCK: return Format::EACRG11Unorm;
        case VK_FORMAT_EAC_R11G11_SNORM_BLOCK: return Format::EACRG11Snorm;
        case VK_FORMAT_ASTC_4x4_UNORM_BLOCK: return Format::ASTC4x4Unorm;
        case VK_FORMAT_ASTC_4x4_SRGB_BLOCK: return Format::ASTC4x4UnormSrgb;
        case VK_FORMAT_ASTC_5x4_UNORM_BLOCK: return Format::ASTC5x4Unorm;
        case VK_FORMAT_ASTC_5x4_SRGB_BLOCK: return Format::ASTC5x4UnormSrgb;
        case VK_FORMAT_ASTC_5x5_UNORM_BLOCK: return Format::ASTC5x5Unorm;
        case VK_FORMAT_ASTC_5x5_SRGB_BLOCK: return Format::ASTC5x5UnormSrgb;
        case VK_FORMAT_ASTC_6x5_UNORM_BLOCK: return Format::ASTC6x5Unorm;
        case VK_FORMAT_ASTC_6x5_SRGB_BLOCK: return Format::ASTC6x5UnormSrgb;
        case VK_FORMAT_ASTC_6x6_UNORM_BLOCK: return Format::ASTC6x6Unorm;
        case VK_FORMAT_ASTC_6x6_SRGB_BLOCK: return Format::ASTC6x6UnormSrgb;
        case VK_FORMAT_ASTC_8x5_UNORM_BLOCK: return Format::ASTC8x5Unorm;
        case VK_FORMAT_ASTC_8x5_SRGB_BLOCK: return Format::ASTC8x5UnormSrgb;
        case VK_FORMAT_ASTC_8x6_UNORM_BLOCK: return Format::ASTC8x6Unorm;
        case VK_FORMAT_ASTC_8x6_SRGB_BLOCK: return Format::ASTC8x6UnormSrgb;
        case VK_FORMAT_ASTC_8x8_UNORM_BLOCK: return Format::ASTC8x8Unorm;
        case VK_FORMAT_ASTC_8x8_SRGB_BLOCK: return Format::ASTC8x8UnormSrgb;
        case VK_FORMAT_ASTC_10x5_UNORM_BLOCK: return Format::ASTC10x5Unorm;
        case VK_FORMAT_ASTC_10x5_SRGB_BLOCK: return Format::ASTC10x5UnormSrgb;
        case VK_FORMAT_ASTC_10x6_UNORM_BLOCK: return Format::ASTC10x6Unorm;
        case VK_FORMAT_ASTC_10x6_SRGB_BLOCK: return Format::ASTC10x6UnormSrgb;
        case VK_FORMAT_ASTC_10x8_UNORM_BLOCK: return Format::ASTC10x8Unorm;
        case VK_FORMAT_ASTC_10x8_SRGB_BLOCK: return Format::ASTC10x8UnormSrgb;
        case VK_FORMAT_ASTC_10x10_UNORM_BLOCK: return Format::ASTC10x10Unorm;
        case VK_FORMAT_ASTC_10x10_SRGB_BLOCK: return Format::ASTC10x10UnormSrgb;
        case VK_FORMAT_ASTC_12x10_UNORM_BLOCK: return Format::ASTC12x10Unorm;
        case VK_FORMAT_ASTC_12x10_SRGB_BLOCK: return Format::ASTC12x10UnormSrgb;
        case VK_FORMAT_ASTC_12x12_UNORM_BLOCK: return Format::ASTC12x12Unorm;
        case VK_FORMAT_ASTC_12x12_SRGB_BLOCK: return Format::ASTC12x12UnormSrgb;
        default: return Format(~0);
    }
}

VkPrimitiveTopology bridge(Topology topo) {
    switch (topo) {
        case Topology::TriangleList: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case Topology::TriangleStrip: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case Topology::TriangleFan: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        default: return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    }
}

PresentMode bridge(VkPresentModeKHR mode) {
    switch (mode) {
        case VK_PRESENT_MODE_IMMEDIATE_KHR: return PresentMode::Immediate;
        case VK_PRESENT_MODE_MAILBOX_KHR: return PresentMode::Mailbox;
        case VK_PRESENT_MODE_FIFO_KHR: return PresentMode::Fifo;
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return PresentMode::FifoRelaxed;
        default: return PresentMode::ValidCount;
    }
}

VkPresentModeKHR bridge(PresentMode mode) {
    switch (mode) {
        case PresentMode::Immediate: return VK_PRESENT_MODE_IMMEDIATE_KHR;
        case PresentMode::Mailbox: return VK_PRESENT_MODE_MAILBOX_KHR;
        case PresentMode::Fifo: return VK_PRESENT_MODE_FIFO_KHR;
        case PresentMode::FifoRelaxed: return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        default: return VK_PRESENT_MODE_MAX_ENUM_KHR;
    }
}

VkPipelineStageFlags2 bridge_pipeline_stage(StageFlags stage) {
    VkPipelineStageFlags2 out = 0;
    out |= (stage & Transfer) ? VK_PIPELINE_STAGE_2_TRANSFER_BIT : 0;
    out |= (stage & Compute) ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : 0;
    out |= (stage & RasterColorOut) ? VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT : 0;
    out |= (stage & PixelShader) ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                              | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                                              | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
                                        : 0;
    out |= (stage & VertexShader)
               ? VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT
               : 0;
    out |= (stage & Host) ? VK_PIPELINE_STAGE_2_HOST_BIT : 0;
    return out;
}

UsageFlags bridge_usage_flags(VkImageUsageFlags flags) {
    UsageFlags usage = UsageFlags::None;
    usage |= (flags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ? UsageFlags::TransferSrc
                                                       : UsageFlags::None;
    usage |= (flags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) ? UsageFlags::TransferDst
                                                       : UsageFlags::None;
    usage |= (flags & VK_IMAGE_USAGE_SAMPLED_BIT) ? UsageFlags::Sampled : UsageFlags::None;
    usage |= (flags & VK_IMAGE_USAGE_STORAGE_BIT) ? UsageFlags::Storage : UsageFlags::None;
    usage |= (flags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ? UsageFlags::ColorAttachment
                                                           : UsageFlags::None;
    usage |= (flags & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                 ? UsageFlags::DepthStencilAttachment
                 : UsageFlags::None;
    // NOTE: Should VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT also count as a render attachment
    // usage?
    return UsageFlags(usage);
}

VkImageUsageFlags bridge_usage_flags(UsageFlags usage) {
    VkImageUsageFlags flags = 0;
    flags |= bool(usage & UsageFlags::TransferSrc) ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0;
    flags |= bool(usage & UsageFlags::TransferDst) ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0;
    flags |= bool(usage & UsageFlags::Sampled) ? VK_IMAGE_USAGE_SAMPLED_BIT : 0;
    flags |= bool(usage & UsageFlags::Storage) ? VK_IMAGE_USAGE_STORAGE_BIT : 0;
    flags |= bool(usage & UsageFlags::ColorAttachment) ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0;
    flags |= bool(usage & UsageFlags::DepthStencilAttachment)
                 ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                 : 0;
    return flags;
}

VkImageType bridge(TextureType tex) {
    switch (tex) {
        case TextureType::Tex1D: return VK_IMAGE_TYPE_1D;
        case TextureType::Tex2D: return VK_IMAGE_TYPE_2D;
        case TextureType::Tex3D: return VK_IMAGE_TYPE_3D;
        case TextureType::TexCube:
        case TextureType::Tex2DArray:
        case TextureType::TexCubeArray: return VK_IMAGE_TYPE_2D;
    }
    return VK_IMAGE_TYPE_MAX_ENUM;
}
VkImageViewType bridge_view_type(TextureType tex) {
    switch (tex) {
        case TextureType::Tex1D: return VK_IMAGE_VIEW_TYPE_1D;
        case TextureType::Tex2D: return VK_IMAGE_VIEW_TYPE_2D;
        case TextureType::Tex3D: return VK_IMAGE_VIEW_TYPE_3D;
        case TextureType::TexCube: return VK_IMAGE_VIEW_TYPE_CUBE;
        case TextureType::Tex2DArray: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        case TextureType::TexCubeArray: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    }
    return VK_IMAGE_VIEW_TYPE_MAX_ENUM;
}

VkBlendFactor bridge(Factor factor) {
    switch (factor) {
        case Factor::Zero: return VK_BLEND_FACTOR_ZERO;
        case Factor::One: return VK_BLEND_FACTOR_ONE;
        case Factor::SrcColor: return VK_BLEND_FACTOR_SRC_COLOR;
        case Factor::DstColor: return VK_BLEND_FACTOR_DST_COLOR;
        case Factor::SrcAlpha: return VK_BLEND_FACTOR_SRC_ALPHA;
    }
}

VkBlendOp bridge(Blend op) {
    switch (op) {
        case Blend::Add: return VK_BLEND_OP_ADD;
        case Blend::Subtract: return VK_BLEND_OP_SUBTRACT;
        case Blend::RevSubtract: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case Blend::Min: return VK_BLEND_OP_MIN;
        case Blend::Max: return VK_BLEND_OP_MAX;
    }
}

VkPipelineColorBlendAttachmentState bridge(const BlendDesc& state) {
    return VkPipelineColorBlendAttachmentState{
        .blendEnable         = true,
        .srcColorBlendFactor = bridge(state.src_color_factor),
        .dstColorBlendFactor = bridge(state.dst_color_factor),
        .colorBlendOp        = bridge(state.color_op),
        .srcAlphaBlendFactor = bridge(state.src_alpha_factor),
        .dstAlphaBlendFactor = bridge(state.dst_alpha_factor),
        .alphaBlendOp        = bridge(state.alpha_op),
        .colorWriteMask      = state.color_write_mask,
    };
}

VkAttachmentLoadOp bridge(LoadOp op) {
    switch (op) {
        case LoadOp::Undefined: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        case LoadOp::Load: return VK_ATTACHMENT_LOAD_OP_LOAD;
        case LoadOp::Clear: return VK_ATTACHMENT_LOAD_OP_CLEAR;
    }
}

VkAttachmentStoreOp bridge(StoreOp op) {
    switch (op) {
        case StoreOp::Undefined: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
        case StoreOp::Store: return VK_ATTACHMENT_STORE_OP_STORE;
        case StoreOp::Discard: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
    }
}

VkCompareOp bridge(Op op) {
    switch (op) {
        case Op::Never: return VK_COMPARE_OP_NEVER;
        case Op::Less: return VK_COMPARE_OP_LESS;
        case Op::Equal: return VK_COMPARE_OP_EQUAL;
        case Op::LessEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
        case Op::Greater: return VK_COMPARE_OP_GREATER;
        case Op::NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
        case Op::GreaterEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case Op::Always: return VK_COMPARE_OP_ALWAYS;
        default: return VK_COMPARE_OP_MAX_ENUM;
    }
}

VkImageLayout bridge(Layout layout) {
    switch (layout) {
        case Layout::DontCare: return VK_IMAGE_LAYOUT_UNDEFINED;
        case Layout::General: return VK_IMAGE_LAYOUT_GENERAL;
        case Layout::Attachment: return VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
        case Layout::Present: return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }
}

}  // namespace loon::gpu
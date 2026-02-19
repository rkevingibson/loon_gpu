#include "gpu_to_mtl.h"

#include "Metal.hpp"
#include "gpu/loon_gpu.h"

namespace loon::gpu {
MTL::PixelFormat bridge(Format f) {
    switch (f) {
        case Format::R8Unorm: return MTL::PixelFormatR8Unorm;
        case Format::R8Snorm: return MTL::PixelFormatR8Snorm;
        case Format::R8Uint: return MTL::PixelFormatR8Uint;
        case Format::R8Sint: return MTL::PixelFormatR8Sint;
        case Format::R16Unorm: return MTL::PixelFormatR16Unorm;
        case Format::R16Snorm: return MTL::PixelFormatR16Snorm;
        case Format::R16Uint: return MTL::PixelFormatR16Uint;
        case Format::R16Sint: return MTL::PixelFormatR16Sint;
        case Format::R16Float: return MTL::PixelFormatR16Float;
        case Format::RG8Unorm: return MTL::PixelFormatRG8Unorm;
        case Format::RG8Snorm: return MTL::PixelFormatRG8Snorm;
        case Format::RG8Uint: return MTL::PixelFormatRG8Uint;
        case Format::RG8Sint: return MTL::PixelFormatRG8Sint;
        case Format::R32Float: return MTL::PixelFormatR32Float;
        case Format::R32Uint: return MTL::PixelFormatR32Uint;
        case Format::R32Sint: return MTL::PixelFormatR32Sint;
        case Format::RG16Unorm: return MTL::PixelFormatRG16Unorm;
        case Format::RG16Snorm: return MTL::PixelFormatRG16Snorm;
        case Format::RG16Uint: return MTL::PixelFormatRG16Uint;
        case Format::RG16Sint: return MTL::PixelFormatRG16Sint;
        case Format::RG16Float: return MTL::PixelFormatRG16Float;
        case Format::RGBA8Unorm: return MTL::PixelFormatRGBA8Unorm;
        case Format::RGBA8UnormSrgb: return MTL::PixelFormatRGBA8Unorm_sRGB;
        case Format::RGBA8Snorm: return MTL::PixelFormatRGBA8Snorm;
        case Format::RGBA8Uint: return MTL::PixelFormatRGBA8Uint;
        case Format::RGBA8Sint: return MTL::PixelFormatRGBA8Sint;
        case Format::BGRA8Unorm: return MTL::PixelFormatBGRA8Unorm;
        case Format::BGRA8UnormSrgb: return MTL::PixelFormatBGRA8Unorm_sRGB;
        case Format::RGB10A2Uint: return MTL::PixelFormatRGB10A2Uint;
        case Format::RGB10A2Unorm: return MTL::PixelFormatRGB10A2Unorm;
        case Format::RG11B10Ufloat: return MTL::PixelFormatRG11B10Float;
        case Format::RGB9E5Ufloat: return MTL::PixelFormatRGB9E5Float;
        case Format::RG32Float: return MTL::PixelFormatRG32Float;
        case Format::RG32Uint: return MTL::PixelFormatRG32Uint;
        case Format::RG32Sint: return MTL::PixelFormatRG32Sint;
        case Format::RGBA16Unorm: return MTL::PixelFormatRGBA16Unorm;
        case Format::RGBA16Snorm: return MTL::PixelFormatRGBA16Snorm;
        case Format::RGBA16Uint: return MTL::PixelFormatRGBA16Uint;
        case Format::RGBA16Sint: return MTL::PixelFormatRGBA16Sint;
        case Format::RGBA16Float: return MTL::PixelFormatRGBA16Float;
        case Format::RGBA32Float: return MTL::PixelFormatRGBA32Float;
        case Format::RGBA32Uint: return MTL::PixelFormatRGBA32Uint;
        case Format::RGBA32Sint: return MTL::PixelFormatRGBA32Sint;
        case Format::Stencil8: return MTL::PixelFormatStencil8;
        case Format::Depth16Unorm: return MTL::PixelFormatDepth16Unorm;
        case Format::Depth24PlusStencil8: return MTL::PixelFormatDepth24Unorm_Stencil8;
        case Format::Depth32Float: return MTL::PixelFormatDepth32Float;
        case Format::Depth32FloatStencil8: return MTL::PixelFormatDepth32Float_Stencil8;
        case Format::BC1RGBAUnorm: return MTL::PixelFormatBC1_RGBA;
        case Format::BC1RGBAUnormSrgb: return MTL::PixelFormatBC1_RGBA_sRGB;
        case Format::BC2RGBAUnorm: return MTL::PixelFormatBC2_RGBA;
        case Format::BC2RGBAUnormSrgb: return MTL::PixelFormatBC2_RGBA_sRGB;
        case Format::BC3RGBAUnorm: return MTL::PixelFormatBC3_RGBA;
        case Format::BC3RGBAUnormSrgb: return MTL::PixelFormatBC3_RGBA_sRGB;
        case Format::BC4RUnorm: return MTL::PixelFormatBC4_RUnorm;
        case Format::BC4RSnorm: return MTL::PixelFormatBC4_RSnorm;
        case Format::BC5RGUnorm: return MTL::PixelFormatBC5_RGUnorm;
        case Format::BC5RGSnorm: return MTL::PixelFormatBC5_RGSnorm;
        case Format::BC6HRGBUfloat: return MTL::PixelFormatBC6H_RGBUfloat;
        case Format::BC6HRGBFloat: return MTL::PixelFormatBC6H_RGBFloat;
        case Format::BC7RGBAUnorm: return MTL::PixelFormatBC7_RGBAUnorm;
        case Format::BC7RGBAUnormSrgb: return MTL::PixelFormatBC7_RGBAUnorm_sRGB;
        case Format::ETC2RGB8Unorm: return MTL::PixelFormatETC2_RGB8;
        case Format::ETC2RGB8UnormSrgb: return MTL::PixelFormatETC2_RGB8_sRGB;
        case Format::ETC2RGB8A1Unorm: return MTL::PixelFormatETC2_RGB8A1;
        case Format::ETC2RGB8A1UnormSrgb: return MTL::PixelFormatETC2_RGB8A1_sRGB;
        case Format::EACR11Unorm: return MTL::PixelFormatEAC_R11Unorm;
        case Format::EACR11Snorm: return MTL::PixelFormatEAC_R11Snorm;
        case Format::EACRG11Unorm: return MTL::PixelFormatEAC_RG11Unorm;
        case Format::EACRG11Snorm: return MTL::PixelFormatEAC_RG11Snorm;
        case Format::ASTC4x4Unorm: return MTL::PixelFormatASTC_4x4_LDR;
        case Format::ASTC4x4UnormSrgb: return MTL::PixelFormatASTC_4x4_sRGB;
        case Format::ASTC5x4Unorm: return MTL::PixelFormatASTC_5x4_LDR;
        case Format::ASTC5x4UnormSrgb: return MTL::PixelFormatASTC_5x4_sRGB;
        case Format::ASTC5x5Unorm: return MTL::PixelFormatASTC_5x5_LDR;
        case Format::ASTC5x5UnormSrgb: return MTL::PixelFormatASTC_5x5_sRGB;
        case Format::ASTC6x5Unorm: return MTL::PixelFormatASTC_6x5_LDR;
        case Format::ASTC6x5UnormSrgb: return MTL::PixelFormatASTC_6x5_sRGB;
        case Format::ASTC6x6Unorm: return MTL::PixelFormatASTC_6x6_LDR;
        case Format::ASTC6x6UnormSrgb: return MTL::PixelFormatASTC_6x6_sRGB;
        case Format::ASTC8x5Unorm: return MTL::PixelFormatASTC_8x5_LDR;
        case Format::ASTC8x5UnormSrgb: return MTL::PixelFormatASTC_8x5_sRGB;
        case Format::ASTC8x6Unorm: return MTL::PixelFormatASTC_8x6_LDR;
        case Format::ASTC8x6UnormSrgb: return MTL::PixelFormatASTC_8x6_sRGB;
        case Format::ASTC8x8Unorm: return MTL::PixelFormatASTC_8x8_LDR;
        case Format::ASTC8x8UnormSrgb: return MTL::PixelFormatASTC_8x8_sRGB;
        case Format::ASTC10x5Unorm: return MTL::PixelFormatASTC_10x5_LDR;
        case Format::ASTC10x5UnormSrgb: return MTL::PixelFormatASTC_10x5_sRGB;
        case Format::ASTC10x6Unorm: return MTL::PixelFormatASTC_10x6_LDR;
        case Format::ASTC10x6UnormSrgb: return MTL::PixelFormatASTC_10x6_sRGB;
        case Format::ASTC10x8Unorm: return MTL::PixelFormatASTC_10x8_LDR;
        case Format::ASTC10x8UnormSrgb: return MTL::PixelFormatASTC_10x8_sRGB;
        case Format::ASTC10x10Unorm: return MTL::PixelFormatASTC_10x10_LDR;
        case Format::ASTC10x10UnormSrgb: return MTL::PixelFormatASTC_10x10_sRGB;
        case Format::ASTC12x10Unorm: return MTL::PixelFormatASTC_12x10_LDR;
        case Format::ASTC12x10UnormSrgb: return MTL::PixelFormatASTC_12x10_sRGB;
        case Format::ASTC12x12Unorm: return MTL::PixelFormatASTC_12x12_LDR;
        case Format::ASTC12x12UnormSrgb: return MTL::PixelFormatASTC_12x12_sRGB;
        default: return MTL::PixelFormatInvalid;
    }
}

MTL::TextureType bridge(TextureType t) {
    switch (t) {
        case TextureType::Tex1D: return MTL::TextureType1D;
        case TextureType::Tex2D: return MTL::TextureType2D;
        case TextureType::Tex3D: return MTL::TextureType3D;
        case TextureType::TexCube: return MTL::TextureTypeCube;
        case TextureType::Tex2DArray: return MTL::TextureType2DArray;
        case TextureType::TexCubeArray: return MTL::TextureTypeCubeArray;
    }
}

MTL::TextureUsage bridge_texture_usage(UsageFlags u) {
    MTL::TextureUsage usages = 0;
    if ((u & UsageFlags::Sampled) != UsageFlags::None) { usages |= MTL::TextureUsageShaderRead; }
    if ((u & UsageFlags::Storage) != UsageFlags::None) {
        usages |= MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead
                  | MTL::TextureUsageShaderAtomic;
    }
    if ((u & UsageFlags::ColorAttachment) != UsageFlags::None
        || (u & UsageFlags::DepthStencilAttachment) != UsageFlags::None) {
        usages |= MTL::TextureUsageRenderTarget;
    }
    // TODO: TransferSrc/Dst, do they need shader write/read?
    // TODO: PixelFormatView might be needed as a default?
    return usages;
}

}  // namespace loon::gpu
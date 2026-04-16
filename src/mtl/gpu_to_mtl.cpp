#include "gpu_to_mtl.h"

#include "Metal.hpp"
#include "gpu/loon_gpu.h"

namespace loon::gpu {
MTL::PixelFormat bridge(Format f) {
    switch (f) {
        case Format::None: return MTL::PixelFormatInvalid;
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
    static constexpr struct {
        UsageFlags        in;
        MTL::TextureUsage out;
    } kMapping[] = {
        {UsageFlags::Sampled, MTL::TextureUsageShaderRead},
        {UsageFlags::Storage,
         MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead
             | MTL::TextureUsageShaderAtomic},
        {UsageFlags::ColorAttachment, MTL::TextureUsageRenderTarget},
        {UsageFlags::DepthStencilAttachment, MTL::TextureUsageRenderTarget},
    };
    // TODO: TransferSrc/Dst, do they need shader write/read?
    // TODO: PixelFormatView might be needed as a default?

    MTL::TextureUsage out = 0;
    for (const auto& entry : kMapping) { out |= any(u & entry.in) ? entry.out : 0; }
    return out;
}

MTL::PrimitiveTopologyClass bridge_topology_class(Topology t) {
    switch (t) {
        case Topology::TriangleList:
        case Topology::TriangleStrip: return MTL::PrimitiveTopologyClassTriangle;
    }
}

MTL::BlendOperation bridge(Blend op) {
    switch (op) {
        case Blend::Add: return MTL::BlendOperationAdd;
        case Blend::Subtract: return MTL::BlendOperationSubtract;
        case Blend::RevSubtract: return MTL::BlendOperationReverseSubtract;
        case Blend::Min: return MTL::BlendOperationMin;
        case Blend::Max: return MTL::BlendOperationMax;
    }
}

MTL::BlendFactor bridge(Factor f) {
    switch (f) {
        case Factor::Zero: return MTL::BlendFactorZero;
        case Factor::One: return MTL::BlendFactorOne;
        case Factor::SrcColor: return MTL::BlendFactorSourceColor;
        case Factor::DstColor: return MTL::BlendFactorDestinationColor;
        case Factor::SrcAlpha: return MTL::BlendFactorSourceAlpha;
        case Factor::OneMinusSrcAlpha: return MTL::BlendFactorOneMinusSourceAlpha;
    }
}

MTL::CompareFunction bridge(Op op) {
    switch (op) {
        case Op::Never: return MTL::CompareFunctionNever;
        case Op::Less: return MTL::CompareFunctionLess;
        case Op::Equal: return MTL::CompareFunctionEqual;
        case Op::LessEqual: return MTL::CompareFunctionLessEqual;
        case Op::Greater: return MTL::CompareFunctionGreater;
        case Op::NotEqual: return MTL::CompareFunctionNotEqual;
        case Op::GreaterEqual: return MTL::CompareFunctionGreaterEqual;
        case Op::Always: return MTL::CompareFunctionAlways;
    }
}

MTL::StencilOperation bridge(StencilOp op) {
    switch (op) {
        case StencilOp::Keep: return MTL::StencilOperationKeep;
        case StencilOp::Zero: return MTL::StencilOperationZero;
        case StencilOp::Replace: return MTL::StencilOperationReplace;
        case StencilOp::IncrementClamp: return MTL::StencilOperationIncrementClamp;
        case StencilOp::DecrementClamp: return MTL::StencilOperationDecrementClamp;
        case StencilOp::Invert: return MTL::StencilOperationInvert;
        case StencilOp::IncrementWrap: return MTL::StencilOperationIncrementWrap;
        case StencilOp::DecrementWrap: return MTL::StencilOperationDecrementWrap;
    }
}

MTL::LoadAction bridge(LoadOp op) {
    switch (op) {
        case LoadOp::Undefined: return MTL::LoadActionDontCare;
        case LoadOp::Load: return MTL::LoadActionLoad;
        case LoadOp::Clear: return MTL::LoadActionClear;
    }
}

MTL::StoreAction bridge(StoreOp op) {
    switch (op) {
        case StoreOp::Undefined: return MTL::StoreActionDontCare;
        case StoreOp::Store: return MTL::StoreActionStore;
        case StoreOp::Discard: return MTL::StoreActionDontCare;
    }
}

MTL::SamplerMinMagFilter bridge_minmag(SamplerFilter f) {
    switch (f) {
        case SamplerFilter::Nearest: return MTL::SamplerMinMagFilterNearest;
        case SamplerFilter::Linear: return MTL::SamplerMinMagFilterLinear;
    }
}

MTL::SamplerMipFilter bridge_mip(SamplerFilter f) {
    switch (f) {
        case SamplerFilter::Nearest: return MTL::SamplerMipFilterNearest;
        case SamplerFilter::Linear: return MTL::SamplerMipFilterLinear;
    }
}

MTL::SamplerAddressMode bridge(SamplerAddressing a) {
    switch (a) {
        case SamplerAddressing::ClampToEdge: return MTL::SamplerAddressModeClampToEdge;
        case SamplerAddressing::Repeat: return MTL::SamplerAddressModeRepeat;
        case SamplerAddressing::Mirrored: return MTL::SamplerAddressModeMirrorRepeat;
    }
}

MTL::Stages bridge(StageFlags s) {
    static constexpr struct {
        StageFlags  in;
        MTL::Stages out;
    } kStageMap[] = {
        {StageFlags::Transfer, MTL::StageBlit},
        {StageFlags::Compute, MTL::StageDispatch},
        {StageFlags::RasterColorOut, MTL::StageFragment},
        {StageFlags::PixelShader, MTL::StageFragment},
        {StageFlags::VertexShader, MTL::StageVertex},
    };
    MTL::Stages out = 0;
    for (const auto entry : kStageMap) { out |= any(s & entry.in) ? entry.out : 0; }
    return out;
}

MTL::PrimitiveType bridge(Topology t) {
    switch (t) {
        case Topology::TriangleList: return MTL::PrimitiveTypeTriangle;
        case Topology::TriangleStrip: return MTL::PrimitiveTypeTriangleStrip;
    }
}

MTL::CullMode bridge(Cull c) {
    switch (c) {
        case Cull::Front: return MTL::CullModeFront;
        case Cull::Back: return MTL::CullModeBack;
        case Cull::None: return MTL::CullModeNone;
    }
}

}  // namespace loon::gpu
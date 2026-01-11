#include "gpu_to_vk.h"

#include "vulkan/vulkan_core.h"

namespace loon::gpu {


VkFormat bridge(FORMAT format) {
    switch (format) {
        case FORMAT_NONE: return VK_FORMAT_UNDEFINED;
        case FORMAT_R8Unorm: return VK_FORMAT_R8_UNORM;
        case FORMAT_R8Snorm: return VK_FORMAT_R8_SNORM;
        case FORMAT_R8Uint: return VK_FORMAT_R8_UINT;
        case FORMAT_R8Sint: return VK_FORMAT_R8_SINT;
        case FORMAT_R16Uint: return VK_FORMAT_R16_UINT;
        case FORMAT_R16Sint: return VK_FORMAT_R16_SINT;
        case FORMAT_R16Float: return VK_FORMAT_R16_SFLOAT;
        case FORMAT_RG8Unorm: return VK_FORMAT_R8G8_UNORM;
        case FORMAT_RG8Snorm: return VK_FORMAT_R8G8_SNORM;
        case FORMAT_RG8Uint: return VK_FORMAT_R8G8_UINT;
        case FORMAT_RG8Sint: return VK_FORMAT_R8G8_SINT;
        case FORMAT_R32Float: return VK_FORMAT_R32_SFLOAT;
        case FORMAT_R32Uint: return VK_FORMAT_R32_UINT;
        case FORMAT_R32Sint: return VK_FORMAT_R32_SINT;
        case FORMAT_RG16Uint: return VK_FORMAT_R16G16_UINT;
        case FORMAT_RG16Sint: return VK_FORMAT_R16G16_SINT;
        case FORMAT_RG16Float: return VK_FORMAT_R16G16_SFLOAT;
        case FORMAT_RGBA8Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
        case FORMAT_RGBA8UnormSrgb: return VK_FORMAT_R8G8B8A8_SRGB;
        case FORMAT_RGBA8Snorm: return VK_FORMAT_R8G8B8A8_SNORM;
        case FORMAT_RGBA8Uint: return VK_FORMAT_R8G8B8A8_UINT;
        case FORMAT_RGBA8Sint: return VK_FORMAT_R8G8B8A8_SINT;
        case FORMAT_BGRA8Unorm: return VK_FORMAT_B8G8R8A8_UNORM;
        case FORMAT_BGRA8UnormSrgb: return VK_FORMAT_B8G8R8A8_SRGB;
        case FORMAT_RGB10A2Uint: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
        case FORMAT_RGB10A2Unorm: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case FORMAT_RG11B10Ufloat: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case FORMAT_RGB9E5Ufloat: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
        case FORMAT_RG32Float: return VK_FORMAT_R32G32_SFLOAT;
        case FORMAT_RG32Uint: return VK_FORMAT_R32G32_UINT;
        case FORMAT_RG32Sint: return VK_FORMAT_R32G32_SINT;
        case FORMAT_RGBA16Uint: return VK_FORMAT_R16G16B16A16_UINT;
        case FORMAT_RGBA16Sint: return VK_FORMAT_R16G16B16A16_SINT;
        case FORMAT_RGBA16Float: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case FORMAT_RGBA32Float: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case FORMAT_RGBA32Uint: return VK_FORMAT_R32G32B32A32_UINT;
        case FORMAT_RGBA32Sint: return VK_FORMAT_R32G32B32A32_SINT;
        case FORMAT_Stencil8: return VK_FORMAT_S8_UINT;
        case FORMAT_Depth16Unorm: return VK_FORMAT_D16_UNORM;
        case FORMAT_Depth24Plus: return VK_FORMAT_X8_D24_UNORM_PACK32;
        case FORMAT_Depth24PlusStencil8: return VK_FORMAT_D24_UNORM_S8_UINT;
        case FORMAT_Depth32Float: return VK_FORMAT_D32_SFLOAT;
        case FORMAT_Depth32FloatStencil8: return VK_FORMAT_D32_SFLOAT_S8_UINT;
        case FORMAT_BC1RGBAUnorm: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case FORMAT_BC1RGBAUnormSrgb: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
        case FORMAT_BC2RGBAUnorm: return VK_FORMAT_BC2_UNORM_BLOCK;
        case FORMAT_BC2RGBAUnormSrgb: return VK_FORMAT_BC2_SRGB_BLOCK;
        case FORMAT_BC3RGBAUnorm: return VK_FORMAT_BC3_UNORM_BLOCK;
        case FORMAT_BC3RGBAUnormSrgb: return VK_FORMAT_BC3_SRGB_BLOCK;
        case FORMAT_BC4RUnorm: return VK_FORMAT_BC4_UNORM_BLOCK;
        case FORMAT_BC4RSnorm: return VK_FORMAT_BC4_SNORM_BLOCK;
        case FORMAT_BC5RGUnorm: return VK_FORMAT_BC5_UNORM_BLOCK;
        case FORMAT_BC5RGSnorm: return VK_FORMAT_BC5_SNORM_BLOCK;
        case FORMAT_BC6HRGBUfloat: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        case FORMAT_BC6HRGBFloat: return VK_FORMAT_BC6H_SFLOAT_BLOCK;
        case FORMAT_BC7RGBAUnorm: return VK_FORMAT_BC7_UNORM_BLOCK;
        case FORMAT_BC7RGBAUnormSrgb: return VK_FORMAT_BC7_SRGB_BLOCK;
        case FORMAT_ETC2RGB8Unorm: return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
        case FORMAT_ETC2RGB8UnormSrgb: return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
        case FORMAT_ETC2RGB8A1Unorm: return VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;
        case FORMAT_ETC2RGB8A1UnormSrgb: return VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;
        case FORMAT_ETC2RGBA8Unorm: return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
        case FORMAT_ETC2RGBA8UnormSrgb: return VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
        case FORMAT_EACR11Unorm: return VK_FORMAT_EAC_R11_UNORM_BLOCK;
        case FORMAT_EACR11Snorm: return VK_FORMAT_EAC_R11_SNORM_BLOCK;
        case FORMAT_EACRG11Unorm: return VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
        case FORMAT_EACRG11Snorm: return VK_FORMAT_EAC_R11G11_SNORM_BLOCK;
        case FORMAT_ASTC4x4Unorm: return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
        case FORMAT_ASTC4x4UnormSrgb: return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
        case FORMAT_ASTC5x4Unorm: return VK_FORMAT_ASTC_5x4_UNORM_BLOCK;
        case FORMAT_ASTC5x4UnormSrgb: return VK_FORMAT_ASTC_5x4_SRGB_BLOCK;
        case FORMAT_ASTC5x5Unorm: return VK_FORMAT_ASTC_5x5_UNORM_BLOCK;
        case FORMAT_ASTC5x5UnormSrgb: return VK_FORMAT_ASTC_5x5_SRGB_BLOCK;
        case FORMAT_ASTC6x5Unorm: return VK_FORMAT_ASTC_6x5_UNORM_BLOCK;
        case FORMAT_ASTC6x5UnormSrgb: return VK_FORMAT_ASTC_6x5_SRGB_BLOCK;
        case FORMAT_ASTC6x6Unorm: return VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
        case FORMAT_ASTC6x6UnormSrgb: return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;
        case FORMAT_ASTC8x5Unorm: return VK_FORMAT_ASTC_8x5_UNORM_BLOCK;
        case FORMAT_ASTC8x5UnormSrgb: return VK_FORMAT_ASTC_8x5_SRGB_BLOCK;
        case FORMAT_ASTC8x6Unorm: return VK_FORMAT_ASTC_8x6_UNORM_BLOCK;
        case FORMAT_ASTC8x6UnormSrgb: return VK_FORMAT_ASTC_8x6_SRGB_BLOCK;
        case FORMAT_ASTC8x8Unorm: return VK_FORMAT_ASTC_8x8_UNORM_BLOCK;
        case FORMAT_ASTC8x8UnormSrgb: return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
        case FORMAT_ASTC10x5Unorm: return VK_FORMAT_ASTC_10x5_UNORM_BLOCK;
        case FORMAT_ASTC10x5UnormSrgb: return VK_FORMAT_ASTC_10x5_SRGB_BLOCK;
        case FORMAT_ASTC10x6Unorm: return VK_FORMAT_ASTC_10x6_UNORM_BLOCK;
        case FORMAT_ASTC10x6UnormSrgb: return VK_FORMAT_ASTC_10x6_SRGB_BLOCK;
        case FORMAT_ASTC10x8Unorm: return VK_FORMAT_ASTC_10x8_UNORM_BLOCK;
        case FORMAT_ASTC10x8UnormSrgb: return VK_FORMAT_ASTC_10x8_SRGB_BLOCK;
        case FORMAT_ASTC10x10Unorm: return VK_FORMAT_ASTC_10x10_UNORM_BLOCK;
        case FORMAT_ASTC10x10UnormSrgb: return VK_FORMAT_ASTC_10x10_SRGB_BLOCK;
        case FORMAT_ASTC12x10Unorm: return VK_FORMAT_ASTC_12x10_UNORM_BLOCK;
        case FORMAT_ASTC12x10UnormSrgb: return VK_FORMAT_ASTC_12x10_SRGB_BLOCK;
        case FORMAT_ASTC12x12Unorm: return VK_FORMAT_ASTC_12x12_UNORM_BLOCK;
        case FORMAT_ASTC12x12UnormSrgb: return VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
        default: return VK_FORMAT_MAX_ENUM;
    }
}

FORMAT bridge(VkFormat format) {
    switch (format) {
        case VK_FORMAT_UNDEFINED: return FORMAT_NONE;
        case VK_FORMAT_R8_UNORM: return FORMAT_R8Unorm;
        case VK_FORMAT_R8_SNORM: return FORMAT_R8Snorm;
        case VK_FORMAT_R8_UINT: return FORMAT_R8Uint;
        case VK_FORMAT_R8_SINT: return FORMAT_R8Sint;
        case VK_FORMAT_R16_UINT: return FORMAT_R16Uint;
        case VK_FORMAT_R16_SINT: return FORMAT_R16Sint;
        case VK_FORMAT_R16_SFLOAT: return FORMAT_R16Float;
        case VK_FORMAT_R8G8_UNORM: return FORMAT_RG8Unorm;
        case VK_FORMAT_R8G8_SNORM: return FORMAT_RG8Snorm;
        case VK_FORMAT_R8G8_UINT: return FORMAT_RG8Uint;
        case VK_FORMAT_R8G8_SINT: return FORMAT_RG8Sint;
        case VK_FORMAT_R32_SFLOAT: return FORMAT_R32Float;
        case VK_FORMAT_R32_UINT: return FORMAT_R32Uint;
        case VK_FORMAT_R32_SINT: return FORMAT_R32Sint;
        case VK_FORMAT_R16G16_UINT: return FORMAT_RG16Uint;
        case VK_FORMAT_R16G16_SINT: return FORMAT_RG16Sint;
        case VK_FORMAT_R16G16_SFLOAT: return FORMAT_RG16Float;
        case VK_FORMAT_R8G8B8A8_UNORM: return FORMAT_RGBA8Unorm;
        case VK_FORMAT_R8G8B8A8_SRGB: return FORMAT_RGBA8UnormSrgb;
        case VK_FORMAT_R8G8B8A8_SNORM: return FORMAT_RGBA8Snorm;
        case VK_FORMAT_R8G8B8A8_UINT: return FORMAT_RGBA8Uint;
        case VK_FORMAT_R8G8B8A8_SINT: return FORMAT_RGBA8Sint;
        case VK_FORMAT_B8G8R8A8_UNORM: return FORMAT_BGRA8Unorm;
        case VK_FORMAT_B8G8R8A8_SRGB: return FORMAT_BGRA8UnormSrgb;
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: return FORMAT_RGB10A2Uint;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return FORMAT_RGB10A2Unorm;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return FORMAT_RG11B10Ufloat;
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return FORMAT_RGB9E5Ufloat;
        case VK_FORMAT_R32G32_SFLOAT: return FORMAT_RG32Float;
        case VK_FORMAT_R32G32_UINT: return FORMAT_RG32Uint;
        case VK_FORMAT_R32G32_SINT: return FORMAT_RG32Sint;
        case VK_FORMAT_R16G16B16A16_UINT: return FORMAT_RGBA16Uint;
        case VK_FORMAT_R16G16B16A16_SINT: return FORMAT_RGBA16Sint;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return FORMAT_RGBA16Float;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return FORMAT_RGBA32Float;
        case VK_FORMAT_R32G32B32A32_UINT: return FORMAT_RGBA32Uint;
        case VK_FORMAT_R32G32B32A32_SINT: return FORMAT_RGBA32Sint;
        case VK_FORMAT_S8_UINT: return FORMAT_Stencil8;
        case VK_FORMAT_D16_UNORM: return FORMAT_Depth16Unorm;
        case VK_FORMAT_X8_D24_UNORM_PACK32: return FORMAT_Depth24Plus;
        case VK_FORMAT_D24_UNORM_S8_UINT: return FORMAT_Depth24PlusStencil8;
        case VK_FORMAT_D32_SFLOAT: return FORMAT_Depth32Float;
        case VK_FORMAT_D32_SFLOAT_S8_UINT: return FORMAT_Depth32FloatStencil8;
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return FORMAT_BC1RGBAUnorm;
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return FORMAT_BC1RGBAUnormSrgb;
        case VK_FORMAT_BC2_UNORM_BLOCK: return FORMAT_BC2RGBAUnorm;
        case VK_FORMAT_BC2_SRGB_BLOCK: return FORMAT_BC2RGBAUnormSrgb;
        case VK_FORMAT_BC3_UNORM_BLOCK: return FORMAT_BC3RGBAUnorm;
        case VK_FORMAT_BC3_SRGB_BLOCK: return FORMAT_BC3RGBAUnormSrgb;
        case VK_FORMAT_BC4_UNORM_BLOCK: return FORMAT_BC4RUnorm;
        case VK_FORMAT_BC4_SNORM_BLOCK: return FORMAT_BC4RSnorm;
        case VK_FORMAT_BC5_UNORM_BLOCK: return FORMAT_BC5RGUnorm;
        case VK_FORMAT_BC5_SNORM_BLOCK: return FORMAT_BC5RGSnorm;
        case VK_FORMAT_BC6H_UFLOAT_BLOCK: return FORMAT_BC6HRGBUfloat;
        case VK_FORMAT_BC6H_SFLOAT_BLOCK: return FORMAT_BC6HRGBFloat;
        case VK_FORMAT_BC7_UNORM_BLOCK: return FORMAT_BC7RGBAUnorm;
        case VK_FORMAT_BC7_SRGB_BLOCK: return FORMAT_BC7RGBAUnormSrgb;
        case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK: return FORMAT_ETC2RGB8Unorm;
        case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK: return FORMAT_ETC2RGB8UnormSrgb;
        case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK: return FORMAT_ETC2RGB8A1Unorm;
        case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK: return FORMAT_ETC2RGB8A1UnormSrgb;
        case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK: return FORMAT_ETC2RGBA8Unorm;
        case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK: return FORMAT_ETC2RGBA8UnormSrgb;
        case VK_FORMAT_EAC_R11_UNORM_BLOCK: return FORMAT_EACR11Unorm;
        case VK_FORMAT_EAC_R11_SNORM_BLOCK: return FORMAT_EACR11Snorm;
        case VK_FORMAT_EAC_R11G11_UNORM_BLOCK: return FORMAT_EACRG11Unorm;
        case VK_FORMAT_EAC_R11G11_SNORM_BLOCK: return FORMAT_EACRG11Snorm;
        case VK_FORMAT_ASTC_4x4_UNORM_BLOCK: return FORMAT_ASTC4x4Unorm;
        case VK_FORMAT_ASTC_4x4_SRGB_BLOCK: return FORMAT_ASTC4x4UnormSrgb;
        case VK_FORMAT_ASTC_5x4_UNORM_BLOCK: return FORMAT_ASTC5x4Unorm;
        case VK_FORMAT_ASTC_5x4_SRGB_BLOCK: return FORMAT_ASTC5x4UnormSrgb;
        case VK_FORMAT_ASTC_5x5_UNORM_BLOCK: return FORMAT_ASTC5x5Unorm;
        case VK_FORMAT_ASTC_5x5_SRGB_BLOCK: return FORMAT_ASTC5x5UnormSrgb;
        case VK_FORMAT_ASTC_6x5_UNORM_BLOCK: return FORMAT_ASTC6x5Unorm;
        case VK_FORMAT_ASTC_6x5_SRGB_BLOCK: return FORMAT_ASTC6x5UnormSrgb;
        case VK_FORMAT_ASTC_6x6_UNORM_BLOCK: return FORMAT_ASTC6x6Unorm;
        case VK_FORMAT_ASTC_6x6_SRGB_BLOCK: return FORMAT_ASTC6x6UnormSrgb;
        case VK_FORMAT_ASTC_8x5_UNORM_BLOCK: return FORMAT_ASTC8x5Unorm;
        case VK_FORMAT_ASTC_8x5_SRGB_BLOCK: return FORMAT_ASTC8x5UnormSrgb;
        case VK_FORMAT_ASTC_8x6_UNORM_BLOCK: return FORMAT_ASTC8x6Unorm;
        case VK_FORMAT_ASTC_8x6_SRGB_BLOCK: return FORMAT_ASTC8x6UnormSrgb;
        case VK_FORMAT_ASTC_8x8_UNORM_BLOCK: return FORMAT_ASTC8x8Unorm;
        case VK_FORMAT_ASTC_8x8_SRGB_BLOCK: return FORMAT_ASTC8x8UnormSrgb;
        case VK_FORMAT_ASTC_10x5_UNORM_BLOCK: return FORMAT_ASTC10x5Unorm;
        case VK_FORMAT_ASTC_10x5_SRGB_BLOCK: return FORMAT_ASTC10x5UnormSrgb;
        case VK_FORMAT_ASTC_10x6_UNORM_BLOCK: return FORMAT_ASTC10x6Unorm;
        case VK_FORMAT_ASTC_10x6_SRGB_BLOCK: return FORMAT_ASTC10x6UnormSrgb;
        case VK_FORMAT_ASTC_10x8_UNORM_BLOCK: return FORMAT_ASTC10x8Unorm;
        case VK_FORMAT_ASTC_10x8_SRGB_BLOCK: return FORMAT_ASTC10x8UnormSrgb;
        case VK_FORMAT_ASTC_10x10_UNORM_BLOCK: return FORMAT_ASTC10x10Unorm;
        case VK_FORMAT_ASTC_10x10_SRGB_BLOCK: return FORMAT_ASTC10x10UnormSrgb;
        case VK_FORMAT_ASTC_12x10_UNORM_BLOCK: return FORMAT_ASTC12x10Unorm;
        case VK_FORMAT_ASTC_12x10_SRGB_BLOCK: return FORMAT_ASTC12x10UnormSrgb;
        case VK_FORMAT_ASTC_12x12_UNORM_BLOCK: return FORMAT_ASTC12x12Unorm;
        case VK_FORMAT_ASTC_12x12_SRGB_BLOCK: return FORMAT_ASTC12x12UnormSrgb;
        default: return FORMAT_Force32;
    }
}

VkPrimitiveTopology bridge(TOPOLOGY topo) {
    switch (topo) {
        case TOPOLOGY_TRIANGLE_LIST: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case TOPOLOGY_TRIANGLE_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case TOPOLOGY_TRIANGLE_FAN: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        default: return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    }
}

// VkCullModeFlags bridge(WGPUCullMode cull) {
//     switch (cull) {
//         case WGPUCullMode_Undefined:
//         case WGPUCullMode_None: return VK_CULL_MODE_NONE;
//         case WGPUCullMode_Front: return VK_CULL_MODE_FRONT_BIT;
//         case WGPUCullMode_Back: return VK_CULL_MODE_BACK_BIT;
//         default: return VK_CULL_MODE_FLAG_BITS_MAX_ENUM;
//     }
// }

// VkFrontFace bridge(WGPUFrontFace ff) {
//     switch (ff) {
//         case WGPUFrontFace_Undefined:
//         case WGPUFrontFace_CW: return VK_FRONT_FACE_CLOCKWISE;
//         case WGPUFrontFace_CCW: return VK_FRONT_FACE_COUNTER_CLOCKWISE;
//         case WGPUFrontFace_Force32:
//         default: return VK_FRONT_FACE_MAX_ENUM;
//     }
// }

// VkCompareOp bridge(WGPUCompareFunction fn) {
//     switch (fn) {
//         case WGPUCompareFunction_Undefined: return VK_COMPARE_OP_MAX_ENUM;
//         case WGPUCompareFunction_Never: return VK_COMPARE_OP_NEVER;
//         case WGPUCompareFunction_Less: return VK_COMPARE_OP_LESS;
//         case WGPUCompareFunction_Equal: return VK_COMPARE_OP_EQUAL;
//         case WGPUCompareFunction_LessEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
//         case WGPUCompareFunction_Greater: return VK_COMPARE_OP_GREATER;
//         case WGPUCompareFunction_NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
//         case WGPUCompareFunction_GreaterEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
//         case WGPUCompareFunction_Always: return VK_COMPARE_OP_ALWAYS;
//         default: return VK_COMPARE_OP_MAX_ENUM;
//     }
// }

PRESENT_MODE bridge(VkPresentModeKHR mode) {
    switch (mode) {
        case VK_PRESENT_MODE_IMMEDIATE_KHR: return PRESENT_MODE_IMMEDIATE;
        case VK_PRESENT_MODE_MAILBOX_KHR: return PRESENT_MODE_MAILBOX;
        case VK_PRESENT_MODE_FIFO_KHR: return PRESENT_MODE_FIFO;
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return PRESENT_MODE_FIFO_RELAXED;
        default: return PRESENT_MODE_VALID_COUNT;
    }
}

VkPresentModeKHR bridge(PRESENT_MODE mode) {
    switch (mode) {
        case PRESENT_MODE_IMMEDIATE: return VK_PRESENT_MODE_IMMEDIATE_KHR;
        case PRESENT_MODE_MAILBOX: return VK_PRESENT_MODE_MAILBOX_KHR;
        case PRESENT_MODE_FIFO: return VK_PRESENT_MODE_FIFO_KHR;
        case PRESENT_MODE_FIFO_RELAXED: return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        default: return VK_PRESENT_MODE_MAX_ENUM_KHR;
    }
}

// VkImageViewType bridge(WGPUTextureViewDimension dim) {
//     switch (dim) {
//         case WGPUTextureViewDimension_1D: return VK_IMAGE_VIEW_TYPE_1D;
//         case WGPUTextureViewDimension_2D: return VK_IMAGE_VIEW_TYPE_2D;
//         case WGPUTextureViewDimension_2DArray: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
//         case WGPUTextureViewDimension_Cube: return VK_IMAGE_VIEW_TYPE_CUBE;
//         case WGPUTextureViewDimension_CubeArray: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
//         case WGPUTextureViewDimension_3D: return VK_IMAGE_VIEW_TYPE_3D;
//         default: return VK_IMAGE_VIEW_TYPE_MAX_ENUM;
//     }
// }

// VkImageAspectFlags bridge(WGPUTextureAspect aspect) {
//     switch (aspect) {
//         case WGPUTextureAspect_Undefined: return VK_IMAGE_ASPECT_COLOR_BIT;
//         case WGPUTextureAspect_All: return VK_IMAGE_ASPECT_STENCIL_BIT |
//         VK_IMAGE_ASPECT_DEPTH_BIT; case WGPUTextureAspect_StencilOnly: return
//         VK_IMAGE_ASPECT_STENCIL_BIT; case WGPUTextureAspect_DepthOnly: return
//         VK_IMAGE_ASPECT_DEPTH_BIT; default: return VK_IMAGE_ASPECT_NONE;
//     }
// }

// VkIndexType bridge(WGPUIndexFormat fmt) {
//     switch (fmt) {
//         case WGPUIndexFormat_Uint16: return VK_INDEX_TYPE_UINT16;
//         case WGPUIndexFormat_Uint32: return VK_INDEX_TYPE_UINT32;
//         case WGPUIndexFormat_Force32:
//         case WGPUIndexFormat_Undefined:
//         default: return VK_INDEX_TYPE_MAX_ENUM;
//     }
// }

// VkShaderStageFlags bridge_shader_stage(WGPUShaderStage stage) {
//     VkShaderStageFlags out = 0;
//     out |= (stage & WGPUShaderStage_Vertex) ? VK_SHADER_STAGE_VERTEX_BIT : 0;
//     out |= (stage & WGPUShaderStage_Fragment) ? VK_SHADER_STAGE_FRAGMENT_BIT : 0;
//     out |= (stage & WGPUShaderStage_Compute) ? VK_SHADER_STAGE_COMPUTE_BIT : 0;
//     return out;
// }

VkPipelineStageFlags2 bridge_pipeline_stage(STAGE stage) {
    VkPipelineStageFlags2 out = 0;
    switch (stage) {
        case STAGE_NONE: return 0;
        case STAGE_TRANSFER: return VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        case STAGE_COMPUTE: return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        case STAGE_RASTER_COLOR_OUT: return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        case STAGE_PIXEL_SHADER: return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case STAGE_VERTEX_SHADER: return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    }
}

// VkSampleCountFlagBits bridge_sample_count(uint32_t sample_count) {
//     switch (sample_count) {
//         case 1: return VK_SAMPLE_COUNT_1_BIT;
//         case 2: return VK_SAMPLE_COUNT_2_BIT;
//         case 4: return VK_SAMPLE_COUNT_4_BIT;
//         case 8: return VK_SAMPLE_COUNT_8_BIT;
//         case 16: return VK_SAMPLE_COUNT_16_BIT;
//         case 32: return VK_SAMPLE_COUNT_32_BIT;
//     }
//     return VK_SAMPLE_COUNT_FLAG_BITS_MAX_ENUM;
// }

USAGE_FLAGS bridge_usage_flags(VkImageUsageFlags flags) {
    int usage = USAGE_NONE;
    usage |= (flags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ? USAGE_TRANSFER_SRC : 0;
    usage |= (flags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) ? USAGE_TRANSFER_DST : 0;
    usage |= (flags & VK_IMAGE_USAGE_SAMPLED_BIT) ? USAGE_SAMPLED : 0;
    usage |= (flags & VK_IMAGE_USAGE_STORAGE_BIT) ? USAGE_STORAGE : 0;
    usage |= (flags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ? USAGE_COLOR_ATTACHMENT : 0;
    usage |= (flags & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? USAGE_DEPTH_STENCIL_ATTACHMENT
                                                                   : 0;
    // NOTE: Should VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT also count as a render attachment
    // usage?
    return USAGE_FLAGS(usage);
}

VkImageUsageFlags bridge_usage_flags(USAGE_FLAGS usage) {
    VkImageUsageFlags flags = 0;
    flags |= (usage & USAGE_TRANSFER_SRC) ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0;
    flags |= (usage & USAGE_TRANSFER_DST) ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0;
    flags |= (usage & USAGE_SAMPLED) ? VK_IMAGE_USAGE_SAMPLED_BIT : 0;
    flags |= (usage & USAGE_STORAGE) ? VK_IMAGE_USAGE_STORAGE_BIT : 0;
    flags |= (usage & USAGE_COLOR_ATTACHMENT) ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0;
    flags |= (usage & USAGE_DEPTH_STENCIL_ATTACHMENT) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                                      : 0;
    return flags;
}

// VkBufferUsageFlags bridge_buffer_usage(WGPUBufferUsage usage) {
//     VkBufferUsageFlags flags = 0;
//     flags |= (usage & WGPUBufferUsage_CopySrc) ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0;
//     flags |= (usage & WGPUBufferUsage_CopyDst) ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0;
//     flags |= (usage & WGPUBufferUsage_Index) ? VK_BUFFER_USAGE_INDEX_BUFFER_BIT : 0;
//     flags |= (usage & WGPUBufferUsage_Vertex) ? VK_BUFFER_USAGE_VERTEX_BUFFER_BIT : 0;
//     flags |= (usage & WGPUBufferUsage_Uniform) ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : 0;
//     flags |= (usage & WGPUBufferUsage_Storage) ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0;
//     flags |= (usage & WGPUBufferUsage_Indirect) ? VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT : 0;
//     // TODO: Double check this one - I think that's all that's required.
//     flags |= (usage & WGPUBufferUsage_QueryResolve) ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0;
//     return flags;
// }

// Stack<WGPUCompositeAlphaMode, 5> bridge_composite_alpha_mode(VkCompositeAlphaFlagsKHR flags) {
//     Stack<WGPUCompositeAlphaMode, 5> composite_alpha_modes;
//     composite_alpha_modes.push(
//         WGPUCompositeAlphaMode_Auto);  // We'll support auto but just use opaque
//     if (flags & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
//         composite_alpha_modes.push(WGPUCompositeAlphaMode_Opaque);
//     }
//     if (flags & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) {
//         composite_alpha_modes.push(WGPUCompositeAlphaMode_Premultiplied);
//     }
//     if (flags & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) {
//         composite_alpha_modes.push(WGPUCompositeAlphaMode_Unpremultiplied);
//     }
//     if (flags & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
//         composite_alpha_modes.push(WGPUCompositeAlphaMode_Inherit);
//     }
//     return composite_alpha_modes;
// }

// VkCompositeAlphaFlagBitsKHR bridge_composite_alpha_mode(WGPUCompositeAlphaMode mode) {
//     switch (mode) {
//         case WGPUCompositeAlphaMode_Auto:
//         case WGPUCompositeAlphaMode_Opaque: return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
//         case WGPUCompositeAlphaMode_Premultiplied: return
//         VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR; case WGPUCompositeAlphaMode_Unpremultiplied:
//             return VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
//         case WGPUCompositeAlphaMode_Inherit: return VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
//         default: return VK_COMPOSITE_ALPHA_FLAG_BITS_MAX_ENUM_KHR;
//     }
// }

VkBlendFactor bridge(FACTOR factor) {
    switch (factor) {
        case FACTOR_ZERO: return VK_BLEND_FACTOR_ZERO;
        case FACTOR_ONE: return VK_BLEND_FACTOR_ONE;
        case FACTOR_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR;
        case FACTOR_DST_COLOR: return VK_BLEND_FACTOR_DST_COLOR;
        case FACTOR_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
    }
}

VkBlendOp bridge(BLEND op) {
    switch (op) {
        case BLEND_ADD: return VK_BLEND_OP_ADD;
        case BLEND_SUBTRACT: return VK_BLEND_OP_SUBTRACT;
        case BLEND_REV_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case BLEND_MIN: return VK_BLEND_OP_MIN;
        case BLEND_MAX: return VK_BLEND_OP_MAX;
    }
}


// VkColorComponentFlags bridge(WGPUColorWriteMask mask) {
//     VkColorComponentFlags result = 0;
//     result |= (mask & WGPUColorWriteMask_Red) ? VK_COLOR_COMPONENT_R_BIT : 0;
//     result |= (mask & WGPUColorWriteMask_Green) ? VK_COLOR_COMPONENT_G_BIT : 0;
//     result |= (mask & WGPUColorWriteMask_Blue) ? VK_COLOR_COMPONENT_B_BIT : 0;
//     result |= (mask & WGPUColorWriteMask_Alpha) ? VK_COLOR_COMPONENT_A_BIT : 0;
//     return result;
// }

VkPipelineColorBlendAttachmentState bridge(const BlendDesc& state) {
    return VkPipelineColorBlendAttachmentState{
        .blendEnable         = true,
        .srcColorBlendFactor = bridge(state.srcColorFactor),
        .dstColorBlendFactor = bridge(state.dstColorFactor),
        .colorBlendOp        = bridge(state.colorOp),
        .srcAlphaBlendFactor = bridge(state.srcAlphaFactor),
        .dstAlphaBlendFactor = bridge(state.dstAlphaFactor),
        .alphaBlendOp        = bridge(state.alphaOp),
        .colorWriteMask      = state.colorWriteMask,
    };
}

VkAttachmentLoadOp bridge(LOAD_OP op) {
    switch (op) {
        case LOAD_OP_UNDEFINED: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        case LOAD_OP_LOAD: return VK_ATTACHMENT_LOAD_OP_LOAD;
        case LOAD_OP_CLEAR: return VK_ATTACHMENT_LOAD_OP_CLEAR;
    }
}

VkAttachmentStoreOp bridge(STORE_OP op) {
    switch (op) {
        case STORE_OP_UNDEFINED: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
        case STORE_OP_STORE: return VK_ATTACHMENT_STORE_OP_STORE;
        case STORE_OP_DISCARD: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
    }
}

// VkStencilOp bridge(WGPUStencilOperation op) {
//     switch (op) {
//         case WGPUStencilOperation_Keep: return VK_STENCIL_OP_KEEP;
//         case WGPUStencilOperation_Zero: return VK_STENCIL_OP_ZERO;
//         case WGPUStencilOperation_Replace: return VK_STENCIL_OP_REPLACE;
//         case WGPUStencilOperation_Invert: return VK_STENCIL_OP_INVERT;
//         case WGPUStencilOperation_IncrementClamp: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
//         case WGPUStencilOperation_DecrementClamp: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
//         case WGPUStencilOperation_IncrementWrap: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
//         case WGPUStencilOperation_DecrementWrap: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
//         case WGPUStencilOperation_Force32:
//         default: return VK_STENCIL_OP_MAX_ENUM;
//     }
// }

// VkStencilOpState bridge(WGPUStencilFaceState state, uint32_t read_mask, uint32_t write_mask) {
//     return VkStencilOpState{
//         .failOp      = bridge(state.failOp),
//         .passOp      = bridge(state.passOp),
//         .depthFailOp = bridge(state.depthFailOp),
//         .compareOp   = bridge(state.compare),
//         .compareMask = read_mask,
//         .writeMask   = write_mask,
//         .reference   = 0,
//     };
// }

// VkPipelineDepthStencilStateCreateInfo bridge(const WGPUDepthStencilState& state) {
//     const bool depthTestEnabled = state.depthCompare != WGPUCompareFunction_Always
//                                   && state.depthCompare != WGPUCompareFunction_Undefined;

//     return VkPipelineDepthStencilStateCreateInfo{
//         .sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
//         .pNext                 = nullptr,
//         .flags                 = 0,
//         .depthTestEnable       = depthTestEnabled,
//         .depthWriteEnable      = (state.depthWriteEnabled == WGPUOptionalBool_True),
//         .depthCompareOp        = bridge(state.depthCompare),
//         .depthBoundsTestEnable = false,  // Will be dynamically controlled.
//         .stencilTestEnable     = true,
//         .front          = bridge(state.stencilFront, state.stencilReadMask,
//         state.stencilWriteMask), .back           = bridge(state.stencilBack,
//         state.stencilReadMask, state.stencilWriteMask), .minDepthBounds = 0, .maxDepthBounds = 1,
//     };
// }

// VkAttachmentLoadOp bridge(WGPULoadOp op) {
//     switch (op) {
//         case WGPULoadOp_Undefined: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
//         case WGPULoadOp_Load: return VK_ATTACHMENT_LOAD_OP_LOAD;
//         case WGPULoadOp_Clear: return VK_ATTACHMENT_LOAD_OP_CLEAR;
//         case WGPULoadOp_Force32: return VK_ATTACHMENT_LOAD_OP_MAX_ENUM;
//     }
// }

// VkAttachmentStoreOp bridge(WGPUStoreOp op) {
//     switch (op) {
//         case WGPUStoreOp_Undefined: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
//         case WGPUStoreOp_Store: return VK_ATTACHMENT_STORE_OP_STORE;
//         case WGPUStoreOp_Discard: return VK_ATTACHMENT_STORE_OP_NONE;
//         case WGPUStoreOp_Force32: return VK_ATTACHMENT_STORE_OP_MAX_ENUM;
//     }
// }

// VkClearColorValue bridge_clear_color_value(WGPUColor color, VkFormat format) {
//     const VkClearColorValue float_color{.float32 = {
//                                             static_cast<float>(color.r),
//                                             static_cast<float>(color.g),
//                                             static_cast<float>(color.b),
//                                             static_cast<float>(color.a),
//                                         },};
//     const VkClearColorValue sint_color{.int32 = {
//                                            static_cast<int32_t>(color.r),
//                                            static_cast<int32_t>(color.g),
//                                            static_cast<int32_t>(color.b),
//                                            static_cast<int32_t>(color.a),
//                                        }};
//     const VkClearColorValue uint_color{.uint32 = {
//                                            static_cast<uint32_t>(color.r),
//                                            static_cast<uint32_t>(color.g),
//                                            static_cast<uint32_t>(color.b),
//                                            static_cast<uint32_t>(color.a),
//                                        }};

//     switch (format) {
//         case VK_FORMAT_R8_UINT:
//         case VK_FORMAT_R8G8_UINT:
//         case VK_FORMAT_R8G8B8_UINT:
//         case VK_FORMAT_B8G8R8_UINT:
//         case VK_FORMAT_R8G8B8A8_UINT:
//         case VK_FORMAT_B8G8R8A8_UINT:
//         case VK_FORMAT_A8B8G8R8_UINT_PACK32:
//         case VK_FORMAT_A2R10G10B10_UINT_PACK32:
//         case VK_FORMAT_A2B10G10R10_UINT_PACK32:
//         case VK_FORMAT_R16_UINT:
//         case VK_FORMAT_R16G16_UINT:
//         case VK_FORMAT_R16G16B16_UINT:
//         case VK_FORMAT_R16G16B16A16_UINT:
//         case VK_FORMAT_R32_UINT:
//         case VK_FORMAT_R32G32_UINT:
//         case VK_FORMAT_R32G32B32_UINT:
//         case VK_FORMAT_R32G32B32A32_UINT:
//         case VK_FORMAT_R64_UINT:
//         case VK_FORMAT_R64G64_UINT:
//         case VK_FORMAT_R64G64B64_UINT:
//         case VK_FORMAT_R64G64B64A64_UINT: return uint_color;
//         case VK_FORMAT_R8_SINT:
//         case VK_FORMAT_R8G8_SINT:
//         case VK_FORMAT_R8G8B8_SINT:
//         case VK_FORMAT_B8G8R8_SINT:
//         case VK_FORMAT_R8G8B8A8_SINT:
//         case VK_FORMAT_B8G8R8A8_SINT:
//         case VK_FORMAT_A8B8G8R8_SINT_PACK32:
//         case VK_FORMAT_A2R10G10B10_SINT_PACK32:
//         case VK_FORMAT_A2B10G10R10_SINT_PACK32:
//         case VK_FORMAT_R16_SINT:
//         case VK_FORMAT_R16G16_SINT:
//         case VK_FORMAT_R16G16B16_SINT:
//         case VK_FORMAT_R16G16B16A16_SINT:
//         case VK_FORMAT_R32_SINT:
//         case VK_FORMAT_R32G32_SINT:
//         case VK_FORMAT_R32G32B32_SINT:
//         case VK_FORMAT_R32G32B32A32_SINT:
//         case VK_FORMAT_R64_SINT:
//         case VK_FORMAT_R64G64_SINT:
//         case VK_FORMAT_R64G64B64_SINT:
//         case VK_FORMAT_R64G64B64A64_SINT: return sint_color;
//         default: return float_color;
//     }
// }

// uint32_t texel_block_width(WGPUTextureFormat format) {
//     switch (format) {
//         case WGPUTextureFormat_Undefined:
//         case WGPUTextureFormat_R8Unorm:
//         case WGPUTextureFormat_R8Snorm:
//         case WGPUTextureFormat_R8Uint:
//         case WGPUTextureFormat_R8Sint:
//         case WGPUTextureFormat_R16Unorm:
//         case WGPUTextureFormat_R16Snorm:
//         case WGPUTextureFormat_R16Uint:
//         case WGPUTextureFormat_R16Sint:
//         case WGPUTextureFormat_R16Float:
//         case WGPUTextureFormat_RG8Unorm:
//         case WGPUTextureFormat_RG8Snorm:
//         case WGPUTextureFormat_RG8Uint:
//         case WGPUTextureFormat_RG8Sint:
//         case WGPUTextureFormat_R32Float:
//         case WGPUTextureFormat_R32Uint:
//         case WGPUTextureFormat_R32Sint:
//         case WGPUTextureFormat_RG16Unorm:
//         case WGPUTextureFormat_RG16Snorm:
//         case WGPUTextureFormat_RG16Uint:
//         case WGPUTextureFormat_RG16Sint:
//         case WGPUTextureFormat_RG16Float:
//         case WGPUTextureFormat_RGBA8Unorm:
//         case WGPUTextureFormat_RGBA8UnormSrgb:
//         case WGPUTextureFormat_RGBA8Snorm:
//         case WGPUTextureFormat_RGBA8Uint:
//         case WGPUTextureFormat_RGBA8Sint:
//         case WGPUTextureFormat_BGRA8Unorm:
//         case WGPUTextureFormat_BGRA8UnormSrgb:
//         case WGPUTextureFormat_RGB10A2Uint:
//         case WGPUTextureFormat_RGB10A2Unorm:
//         case WGPUTextureFormat_RG11B10Ufloat:
//         case WGPUTextureFormat_RGB9E5Ufloat:
//         case WGPUTextureFormat_RG32Float:
//         case WGPUTextureFormat_RG32Uint:
//         case WGPUTextureFormat_RG32Sint:
//         case WGPUTextureFormat_RGBA16Unorm:
//         case WGPUTextureFormat_RGBA16Snorm:
//         case WGPUTextureFormat_RGBA16Uint:
//         case WGPUTextureFormat_RGBA16Sint:
//         case WGPUTextureFormat_RGBA16Float:
//         case WGPUTextureFormat_RGBA32Float:
//         case WGPUTextureFormat_RGBA32Uint:
//         case WGPUTextureFormat_RGBA32Sint:
//         case WGPUTextureFormat_Stencil8:
//         case WGPUTextureFormat_Depth16Unorm:
//         case WGPUTextureFormat_Depth24Plus:
//         case WGPUTextureFormat_Depth24PlusStencil8:
//         case WGPUTextureFormat_Depth32Float:
//         case WGPUTextureFormat_Depth32FloatStencil8: return 1;
//         case WGPUTextureFormat_BC1RGBAUnorm:
//         case WGPUTextureFormat_BC1RGBAUnormSrgb:
//         case WGPUTextureFormat_BC2RGBAUnorm:
//         case WGPUTextureFormat_BC2RGBAUnormSrgb:
//         case WGPUTextureFormat_BC3RGBAUnorm:
//         case WGPUTextureFormat_BC3RGBAUnormSrgb:
//         case WGPUTextureFormat_BC4RUnorm:
//         case WGPUTextureFormat_BC4RSnorm:
//         case WGPUTextureFormat_BC5RGUnorm:
//         case WGPUTextureFormat_BC5RGSnorm:
//         case WGPUTextureFormat_BC6HRGBUfloat:
//         case WGPUTextureFormat_BC6HRGBFloat:
//         case WGPUTextureFormat_BC7RGBAUnorm:
//         case WGPUTextureFormat_BC7RGBAUnormSrgb:
//         case WGPUTextureFormat_ETC2RGB8Unorm:
//         case WGPUTextureFormat_ETC2RGB8UnormSrgb:
//         case WGPUTextureFormat_ETC2RGB8A1Unorm:
//         case WGPUTextureFormat_ETC2RGB8A1UnormSrgb:
//         case WGPUTextureFormat_ETC2RGBA8Unorm:
//         case WGPUTextureFormat_ETC2RGBA8UnormSrgb:
//         case WGPUTextureFormat_EACR11Unorm:
//         case WGPUTextureFormat_EACR11Snorm:
//         case WGPUTextureFormat_EACRG11Unorm:
//         case WGPUTextureFormat_EACRG11Snorm:
//         case WGPUTextureFormat_ASTC4x4Unorm:
//         case WGPUTextureFormat_ASTC4x4UnormSrgb: return 4;
//         case WGPUTextureFormat_ASTC5x4Unorm:
//         case WGPUTextureFormat_ASTC5x4UnormSrgb:
//         case WGPUTextureFormat_ASTC5x5Unorm:
//         case WGPUTextureFormat_ASTC5x5UnormSrgb: return 5;
//         case WGPUTextureFormat_ASTC6x5Unorm:
//         case WGPUTextureFormat_ASTC6x5UnormSrgb:
//         case WGPUTextureFormat_ASTC6x6Unorm:
//         case WGPUTextureFormat_ASTC6x6UnormSrgb: return 6;
//         case WGPUTextureFormat_ASTC8x5Unorm:
//         case WGPUTextureFormat_ASTC8x5UnormSrgb:
//         case WGPUTextureFormat_ASTC8x6Unorm:
//         case WGPUTextureFormat_ASTC8x6UnormSrgb:
//         case WGPUTextureFormat_ASTC8x8Unorm:
//         case WGPUTextureFormat_ASTC8x8UnormSrgb: return 8;
//         case WGPUTextureFormat_ASTC10x5Unorm:
//         case WGPUTextureFormat_ASTC10x5UnormSrgb:
//         case WGPUTextureFormat_ASTC10x6Unorm:
//         case WGPUTextureFormat_ASTC10x6UnormSrgb:
//         case WGPUTextureFormat_ASTC10x8Unorm:
//         case WGPUTextureFormat_ASTC10x8UnormSrgb:
//         case WGPUTextureFormat_ASTC10x10Unorm:
//         case WGPUTextureFormat_ASTC10x10UnormSrgb: return 10;
//         case WGPUTextureFormat_ASTC12x10Unorm:
//         case WGPUTextureFormat_ASTC12x10UnormSrgb:
//         case WGPUTextureFormat_ASTC12x12Unorm:
//         case WGPUTextureFormat_ASTC12x12UnormSrgb: return 12;
//         case WGPUTextureFormat_Force32: break;
//     }
//     return 1;
// }

// uint32_t texel_block_height(WGPUTextureFormat format) {
//     switch (format) {
//         case WGPUTextureFormat_Undefined:
//         case WGPUTextureFormat_R8Unorm:
//         case WGPUTextureFormat_R8Snorm:
//         case WGPUTextureFormat_R8Uint:
//         case WGPUTextureFormat_R8Sint:
//         case WGPUTextureFormat_R16Unorm:
//         case WGPUTextureFormat_R16Snorm:
//         case WGPUTextureFormat_R16Uint:
//         case WGPUTextureFormat_R16Sint:
//         case WGPUTextureFormat_R16Float:
//         case WGPUTextureFormat_RG8Unorm:
//         case WGPUTextureFormat_RG8Snorm:
//         case WGPUTextureFormat_RG8Uint:
//         case WGPUTextureFormat_RG8Sint:
//         case WGPUTextureFormat_R32Float:
//         case WGPUTextureFormat_R32Uint:
//         case WGPUTextureFormat_R32Sint:
//         case WGPUTextureFormat_RG16Unorm:
//         case WGPUTextureFormat_RG16Snorm:
//         case WGPUTextureFormat_RG16Uint:
//         case WGPUTextureFormat_RG16Sint:
//         case WGPUTextureFormat_RG16Float:
//         case WGPUTextureFormat_RGBA8Unorm:
//         case WGPUTextureFormat_RGBA8UnormSrgb:
//         case WGPUTextureFormat_RGBA8Snorm:
//         case WGPUTextureFormat_RGBA8Uint:
//         case WGPUTextureFormat_RGBA8Sint:
//         case WGPUTextureFormat_BGRA8Unorm:
//         case WGPUTextureFormat_BGRA8UnormSrgb:
//         case WGPUTextureFormat_RGB10A2Uint:
//         case WGPUTextureFormat_RGB10A2Unorm:
//         case WGPUTextureFormat_RG11B10Ufloat:
//         case WGPUTextureFormat_RGB9E5Ufloat:
//         case WGPUTextureFormat_RG32Float:
//         case WGPUTextureFormat_RG32Uint:
//         case WGPUTextureFormat_RG32Sint:
//         case WGPUTextureFormat_RGBA16Unorm:
//         case WGPUTextureFormat_RGBA16Snorm:
//         case WGPUTextureFormat_RGBA16Uint:
//         case WGPUTextureFormat_RGBA16Sint:
//         case WGPUTextureFormat_RGBA16Float:
//         case WGPUTextureFormat_RGBA32Float:
//         case WGPUTextureFormat_RGBA32Uint:
//         case WGPUTextureFormat_RGBA32Sint:
//         case WGPUTextureFormat_Stencil8:
//         case WGPUTextureFormat_Depth16Unorm:
//         case WGPUTextureFormat_Depth24Plus:
//         case WGPUTextureFormat_Depth24PlusStencil8:
//         case WGPUTextureFormat_Depth32Float:
//         case WGPUTextureFormat_Depth32FloatStencil8: return 1;
//         case WGPUTextureFormat_BC1RGBAUnorm:
//         case WGPUTextureFormat_BC1RGBAUnormSrgb:
//         case WGPUTextureFormat_BC2RGBAUnorm:
//         case WGPUTextureFormat_BC2RGBAUnormSrgb:
//         case WGPUTextureFormat_BC3RGBAUnorm:
//         case WGPUTextureFormat_BC3RGBAUnormSrgb:
//         case WGPUTextureFormat_BC4RUnorm:
//         case WGPUTextureFormat_BC4RSnorm:
//         case WGPUTextureFormat_BC5RGUnorm:
//         case WGPUTextureFormat_BC5RGSnorm:
//         case WGPUTextureFormat_BC6HRGBUfloat:
//         case WGPUTextureFormat_BC6HRGBFloat:
//         case WGPUTextureFormat_BC7RGBAUnorm:
//         case WGPUTextureFormat_BC7RGBAUnormSrgb:
//         case WGPUTextureFormat_ETC2RGB8Unorm:
//         case WGPUTextureFormat_ETC2RGB8UnormSrgb:
//         case WGPUTextureFormat_ETC2RGB8A1Unorm:
//         case WGPUTextureFormat_ETC2RGB8A1UnormSrgb:
//         case WGPUTextureFormat_ETC2RGBA8Unorm:
//         case WGPUTextureFormat_ETC2RGBA8UnormSrgb:
//         case WGPUTextureFormat_EACR11Unorm:
//         case WGPUTextureFormat_EACR11Snorm:
//         case WGPUTextureFormat_EACRG11Unorm:
//         case WGPUTextureFormat_EACRG11Snorm:
//         case WGPUTextureFormat_ASTC4x4Unorm:
//         case WGPUTextureFormat_ASTC4x4UnormSrgb:
//         case WGPUTextureFormat_ASTC5x4Unorm:
//         case WGPUTextureFormat_ASTC5x4UnormSrgb: return 4;
//         case WGPUTextureFormat_ASTC5x5Unorm:
//         case WGPUTextureFormat_ASTC5x5UnormSrgb:
//         case WGPUTextureFormat_ASTC6x5Unorm:
//         case WGPUTextureFormat_ASTC6x5UnormSrgb: return 5;
//         case WGPUTextureFormat_ASTC6x6Unorm:
//         case WGPUTextureFormat_ASTC6x6UnormSrgb: return 6;
//         case WGPUTextureFormat_ASTC8x5Unorm:
//         case WGPUTextureFormat_ASTC8x5UnormSrgb: return 5;
//         case WGPUTextureFormat_ASTC8x6Unorm:
//         case WGPUTextureFormat_ASTC8x6UnormSrgb: return 6;
//         case WGPUTextureFormat_ASTC8x8Unorm:
//         case WGPUTextureFormat_ASTC8x8UnormSrgb: return 8;
//         case WGPUTextureFormat_ASTC10x5Unorm:
//         case WGPUTextureFormat_ASTC10x5UnormSrgb: return 5;
//         case WGPUTextureFormat_ASTC10x6Unorm:
//         case WGPUTextureFormat_ASTC10x6UnormSrgb: return 6;
//         case WGPUTextureFormat_ASTC10x8Unorm:
//         case WGPUTextureFormat_ASTC10x8UnormSrgb: return 8;
//         case WGPUTextureFormat_ASTC10x10Unorm:
//         case WGPUTextureFormat_ASTC10x10UnormSrgb:
//         case WGPUTextureFormat_ASTC12x10Unorm:
//         case WGPUTextureFormat_ASTC12x10UnormSrgb: return 10;
//         case WGPUTextureFormat_ASTC12x12Unorm:
//         case WGPUTextureFormat_ASTC12x12UnormSrgb: return 12;
//         case WGPUTextureFormat_Force32: break; break;
//     }
//     return 1;
// }

// VkImageLayout image_layout_from_usage(ResourceUsage u) {
//     if ((u & kUsageAttachment)) {
//         return VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
//     } else if ((u & kUsageAttachmentRead) || (u & kUsageStorageRead)) {
//         return VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
//     } else if (u == kUsageUndefined) {
//         return VK_IMAGE_LAYOUT_UNDEFINED;
//     } else if (u == kUsagePresent) {
//         return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
//     } else {
//         return VK_IMAGE_LAYOUT_GENERAL;
//     }
// }

}  // namespace loon::gpu
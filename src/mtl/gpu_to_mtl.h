#pragma once

#include "Metal.hpp"
#include "gpu/loon_gpu.h"

namespace loon::gpu {

MTL::PixelFormat            bridge(Format f);
MTL::TextureType            bridge(TextureType t);
MTL::TextureUsage           bridge_texture_usage(UsageFlags u);
MTL::PrimitiveTopologyClass bridge_topology_class(Topology t);
MTL::BlendOperation         bridge(Blend op);
MTL::BlendFactor            bridge(Factor f);
MTL::CompareFunction        bridge(Op op);
MTL::StencilOperation       bridge(StencilOp op);
MTL::LoadAction             bridge(LoadOp op);
MTL::StoreAction            bridge(StoreOp op);
MTL::SamplerMinMagFilter    bridge_minmag(SamplerFilter f);
MTL::SamplerMipFilter       bridge_mip(SamplerFilter f);
MTL::SamplerAddressMode     bridge(SamplerAddressing a);
MTL::Stages                 bridge(StageFlags s);
MTL::PrimitiveType          bridge(Topology t);
MTL::CullMode               bridge(Cull c);

}  // namespace loon::gpu
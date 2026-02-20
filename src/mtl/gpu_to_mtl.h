#pragma once


#include "Metal.hpp"
#include "gpu/loon_gpu.h"

namespace loon::gpu {

MTL::PixelFormat            bridge(Format f);
MTL::TextureType            bridge(TextureType t);
MTL::TextureUsage           bridge_texture_usage(UsageFlags u);
MTL::PrimitiveTopologyClass bridge(Topology t);
MTL::BlendOperation         bridge(Blend op);
MTL::BlendFactor            bridge(Factor f);

}  // namespace loon::gpu
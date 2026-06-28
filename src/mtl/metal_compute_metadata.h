#pragma once

#include <gpu/loon_gpu.h>

#include "containers.h"
namespace loon::gpu {

struct ShaderMetadata {
    Dimension3D required_threadgroup_size;
};

ShaderMetadata parse_metadata(Span<const char> metal_source, Span<const char> entry_point);

}  // namespace loon::gpu

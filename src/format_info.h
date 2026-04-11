#pragma once

#include "gpu/loon_gpu.h"
namespace loon::gpu {

struct FormatInfo {
    Format  format;
    uint8_t block_size_bytes;
    uint8_t block_width  = 1;
    uint8_t block_height = 1;
};

FormatInfo get_format_info(Format f);

}  // namespace loon::gpu
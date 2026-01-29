#pragma once
#include <cstdint>

enum class VertexAttribute {
    kPosition,
    kColor,
    kUV,
};

enum class VertexAttributeFormat {
    kFloat2,
    kFloat3,
    kFloat4,
};

class Mesh {
   public:
    uint32_t              write_to(void* ptr, uint32_t buffer_size);
    uint32_t              offset(VertexAttribute attr);
    VertexAttributeFormat format(VertexAttribute attr);

   private:
};
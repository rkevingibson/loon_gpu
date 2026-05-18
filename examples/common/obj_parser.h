#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "box.h"
#include "geometry.h"
#include "string_view.h"

namespace loon {

enum class ObjErrorCode {
    Success = 0,
    LineOverflow,  // Trying to parse a line longer than RKG_OBJ_MAX_LINE_LENGTH, try
                   // increasing that (uses a bit more stack memory)
    ReadPastEOF,
    FileNotFound,
};
class LineReader {
   public:
    ~LineReader();

    static LineReader from_file(const char* filename);
    static LineReader from_mem(void* mem, size_t size);

    ObjErrorCode next();
    StringView   line() const;

   private:
    LineReader()         = default;
    char* m_buffer_begin = nullptr;
    char* m_buffer_end   = nullptr;
    char* m_line_begin   = nullptr;
    char* m_line_end     = nullptr;
    void* m_file         = nullptr;
};

struct ObjMesh {
    std::vector<geometry::float4> positions;
    std::vector<geometry::float3> texcoords;
    std::vector<geometry::float3> normals;

    std::vector<int8_t>  face_vertex_count;
    std::vector<int32_t> face_pos_indices;
    std::vector<int32_t> face_tex_indices;
    std::vector<int32_t> face_normal_indices;
};

Box<ObjMesh> obj_parse(LineReader& stream);

}  // namespace loon
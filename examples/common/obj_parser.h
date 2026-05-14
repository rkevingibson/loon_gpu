#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "box.h"
#include "geometry.h"

namespace loon {

enum class ObjErrorCode {
    Success = 0,
    LineOverflow,  // Trying to parse a line longer than RKG_OBJ_MAX_LINE_LENGTH, try
                   // increasing that (uses a bit more stack memory)
    ReadPastEOF,
};

struct ObjBufferedStream {
    const uint8_t* start;   // start of buffer
    const uint8_t* end;     // One past end of buffer
    const uint8_t* cursor;  // Cursor in buffer. Invariant: start <= cursor <= end

    ObjErrorCode error;

    ObjErrorCode (*refill)(ObjBufferedStream* s);
    // Precondition: cursor == end (buffer fully consumed)
    // Postcondition: start == cursor < end
    // Always returns "error"
};

void buffered_stream_from_mem(ObjBufferedStream* s, uint8_t* mem, size_t size);

struct ObjMesh {
    std::vector<geometry::float4> positions;
    std::vector<geometry::float3> texcoords;
    std::vector<geometry::float3> normals;

    std::vector<int8_t>  face_vertex_count;
    std::vector<int32_t> face_pos_indices;
    std::vector<int32_t> face_tex_indices;
    std::vector<int32_t> face_normal_indices;
};

Box<ObjMesh> obj_parse(ObjBufferedStream* stream);

// Internals:

#define LOON_OBJ_MAX_LINE_LENGTH 256
typedef struct rkg_buffered_line_reader {
    // Only used as-needed, if the underlying stream contains a partial line and needs to be
    // refilled, the line can be copied into here.
    char        line_buffer[LOON_OBJ_MAX_LINE_LENGTH];
    const char* line;
    size_t      line_len;


    // Invariant: stream->cursor always points to the start of the next line (after any \n, \r
    // characters)
    ObjBufferedStream* stream;
} rkg_buffered_line_reader;

// Advance the line reader, updating line_start and line_end and consuming the line from the stream.
ObjErrorCode rkg_buffered_line_reader_next(rkg_buffered_line_reader* reader);

}  // namespace loon
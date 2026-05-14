#include "obj_parser.h"

#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>

#include "string_view.h"

namespace loon {

static ObjErrorCode refill_zeros(ObjBufferedStream* s) {
    static const uint8_t zeros[256] = {0};
    s->start                        = zeros;
    s->end                          = zeros + sizeof(zeros);
    s->cursor                       = zeros;
    return s->error;
}

static ObjErrorCode fail(ObjBufferedStream* s, ObjErrorCode reason) {
    s->error  = reason;
    s->refill = refill_zeros;
    return s->refill(s);
}

static ObjErrorCode refill_mem_stream(ObjBufferedStream* s) {
    return fail(s, ObjErrorCode::ReadPastEOF);
}

void ObjBufferedStream_from_mem(ObjBufferedStream* s, uint8_t* mem, size_t size) {
    s->start  = mem;
    s->cursor = mem;
    s->end    = mem + size;
    s->error  = ObjErrorCode::Success;
    s->refill = refill_mem_stream;
}

ObjErrorCode rkg_buffered_line_reader_next(rkg_buffered_line_reader* reader) {
    char*          linebuf = reader->line_buffer;
    size_t         linelen = 0;
    const uint8_t *start, *chr, *end;
    do {
        start = reader->stream->cursor;
        end   = reader->stream->end;

        uintptr_t count = (end - start);
        chr             = static_cast<const uint8_t*>(memchr(start, '\n', count));
        if (chr == NULL) { chr = end; }

        uintptr_t len = (uintptr_t)(chr - start);
        linelen += len;

        if (chr == end)  // Need to do a copy
        {
            if (linelen > LOON_OBJ_MAX_LINE_LENGTH) { return ObjErrorCode::LineOverflow; }
            memcpy(linebuf, start, len);
            linebuf += len;
            reader->stream->cursor = reader->stream->end;
            const ObjErrorCode err = reader->stream->refill(reader->stream);
            if (err != ObjErrorCode::Success) { return err; }
        }
    } while (*chr != '\n');

    // Here, need to see if we ever copied. If we didn't, then linebuf should ==
    // reader->line_buffer, and we can just set it to cursor instead.
    reader->line           = (linebuf == reader->line_buffer) ? (const char*)reader->stream->cursor
                                                              : reader->line_buffer;
    reader->stream->cursor = chr + 1;

    if (linelen > 0 && reader->line[linelen - 1] == '\r')  // Windows-style newline.
        --linelen;

    reader->line_len = linelen;
    return ObjErrorCode::Success;
}

bool is_blank(char x) {
    return x == ' ' || x == '\t';
}

static StringView pop_next_token(StringView* line) {
    const char* begin = line->begin();
    const char* it    = begin;
    const char* end   = line->end();
    while (it != end && is_blank(*it)) ++it;

    const char* token_start = it;

    while (it != end && !is_blank(*it)) ++it;
    line->remove_prefix(it - begin);
    return StringView(token_start, it);
}

static void consume_leading_whitespace(StringView* span) {
    const char* it  = span->begin();
    const char* end = span->end();
    while (it != end && is_blank(*it)) { ++it; }
    span->remove_prefix(it - span->begin());
}

// Parse an int that's at the beginning of the token, updating tok to point to one after the parsed
// integer.
static int parse_int(StringView* tok, const char* buffer_end) {
    // No good native library solution, just roll our own naive one for now.
    // We know that since we're returning a 32-bit signed integer, we have a maximum of 10 digits
    // plus an optional hyphen. More commonly, we will have less than 8 digits, which would fit into
    // a uint64_t.
    (void)buffer_end;
    const char* it = tok->begin();

    int negmask = 1;
    if (*it == '-') {
        it++;
        negmask = -1;
    }

    int result = 0;
    {
        while (it != tok->end() && *it >= '0' && *it <= '9') {
            result *= 10;
            result += *it - '0';
            ++it;
        }
    }

    tok->remove_prefix(it - tok->begin());
    return result * negmask;
}

static void parse_position(StringView line, ObjMesh* mesh) {
    double p[4] = {0};

    for (int i = 0; i < 3; ++i) {
        const StringView tok = pop_next_token(&line);
        p[i]                 = atof(tok.begin());
    }

    const StringView tok = pop_next_token(&line);
    if (!tok.is_empty()) { p[3] = atof(tok.begin()); }
    auto& positions = mesh->positions;
    positions.push_back(geometry::float4{
        static_cast<float>(p[0]),
        static_cast<float>(p[1]),
        static_cast<float>(p[2]),
        static_cast<float>(p[3]),
    });
}

static void parse_texture_vert(StringView line, ObjMesh* mesh) {
    double p[3] = {0};
    // UVW coordinates. V,W are optional, default to 0.
    StringView tok = pop_next_token(&line);
    p[0]           = atof(tok.begin());

    tok = pop_next_token(&line);
    if (!tok.is_empty()) { p[1] = atof(tok.begin()); }

    tok = pop_next_token(&line);
    if (!tok.is_empty()) { p[2] = atof(tok.begin()); }

    auto& texcoords = mesh->texcoords;
    texcoords.push_back({
        static_cast<float>(p[0]),
        static_cast<float>(p[1]),
        static_cast<float>(p[2]),
    });
}

static void parse_normal_vert(StringView line, ObjMesh* mesh) {
    double p[3] = {0};

    for (int i = 0; i < 3; ++i) {
        StringView tok = pop_next_token(&line);
        p[i]           = atof(tok.begin());
    }

    auto& normals = mesh->normals;
    normals.push_back({
        static_cast<float>(p[0]),
        static_cast<float>(p[1]),
        static_cast<float>(p[2]),
    });
}

static void parse_space_vert(StringView line, ObjMesh* mesh) {
    // TODO: Support curves/surfaces.
    (void)line;
    (void)mesh;
}

static void parse_face(StringView line, const char* buffer_end, ObjMesh* mesh) {
    // Faces are the most complicated thing to parse.
    // The format may just be a list of numbers, but it may have slashes separating texture coords
    // and vertex normals. As well, the length of the list isn't known ahead of time. How to
    // best handle this? Simplest way is to just assume it has texture and normal,
    int8_t num_face_verts = 0;

    consume_leading_whitespace(&line);

    while (!line.is_empty()) {
        // Anything remaining is the normal index.
        int32_t pos_index    = parse_int(&line, buffer_end);
        int32_t tex_index    = 0;
        int32_t normal_index = 0;
        if (line.front() == '/') {
            line.remove_prefix(1);
            tex_index = parse_int(&line, buffer_end);
        }
        if (line.front() == '/') {
            line.remove_prefix(1);
            normal_index = parse_int(&line, buffer_end);
        }

        // Index fixup: Convert to 0-index, and negative values are relative to end of array.
        pos_index    = (pos_index < 0) ? pos_index + static_cast<int32_t>(mesh->positions.size())
                                       : pos_index - 1;
        tex_index    = (tex_index < 0) ? tex_index + static_cast<int32_t>(mesh->texcoords.size())
                                       : tex_index - 1;
        normal_index = (normal_index < 0)
                           ? normal_index + static_cast<int32_t>(mesh->normals.size())
                           : normal_index - 1;
        mesh->face_pos_indices.push_back(pos_index);
        mesh->face_tex_indices.push_back(tex_index);
        mesh->face_normal_indices.push_back(normal_index);

        ++num_face_verts;
        consume_leading_whitespace(&line);
    }
    mesh->face_vertex_count.push_back(num_face_verts);
}

#define TOKEN_AS_INT64_IMPL(t)                                                                     \
    ((uint64_t)((t)[0]) << 0ull | (uint64_t)((t)[1]) << 8ull | (uint64_t)((t)[2]) << 16ull |       \
     (uint64_t)((t)[3]) << 24ull | (uint64_t)((t)[4]) << 32ull | (uint64_t)((t)[5]) << 40ull |     \
     (uint64_t)((t)[6]) << 48ull | (uint64_t)((t)[7]) << 56ull)
#define TOKEN_AS_INT64(t) TOKEN_AS_INT64_IMPL(t "\0\0\0\0\0\0\0\0")
enum class TokenType : uint64_t {
    GEOMETRIC_VERT   = TOKEN_AS_INT64("v"),      // v
    TEXTURE_VERT     = TOKEN_AS_INT64("vt"),     // vt
    NORMAL_VERT      = TOKEN_AS_INT64("vn"),     // vn
    PARAM_SPACE_VERT = TOKEN_AS_INT64("vp"),     // vp
    POINT            = TOKEN_AS_INT64("p"),      // p
    LINE             = TOKEN_AS_INT64("l"),      // l
    FACE             = TOKEN_AS_INT64("f"),      // f
    CURVE            = TOKEN_AS_INT64("curv"),   // curv
    CURVE2D          = TOKEN_AS_INT64("curv2"),  // curv2
    SURFACE          = TOKEN_AS_INT64("surf"),   // surf
    GROUP_NAME       = TOKEN_AS_INT64("g"),      // g
    SMOOTHING_GROUP  = TOKEN_AS_INT64("s"),      // s
    MERGING_GROUP    = TOKEN_AS_INT64("mg"),     // mg
    OBJECT_NAME      = TOKEN_AS_INT64("o"),      // o
    MATERIAL         = TOKEN_AS_INT64("mtllib"),
};

Box<ObjMesh> rkg_obj_parse(ObjBufferedStream* stream) {
    Box<ObjMesh> mesh = make_box<ObjMesh>(ObjMesh{
        .positions           = {},
        .texcoords           = {},
        .normals             = {},
        .face_vertex_count   = {},
        .face_pos_indices    = {},
        .face_tex_indices    = {},
        .face_normal_indices = {},
    });

    rkg_buffered_line_reader reader = {
        .line_buffer = {},
        .line        = nullptr,
        .line_len    = 0,
        .stream      = stream,
    };

    // OBJ is a line-centric file format. The parsing of each line depends on the first token
    while (rkg_buffered_line_reader_next(&reader) == ObjErrorCode::Success) {
        StringView line = StringView(reader.line, reader.line_len);

        const StringView tok          = pop_next_token(&line);
        const size_t     token_length = tok.size();
        if (token_length == 0)  // Blank line
            continue;

        if (tok.front() == '#')  // Comment
            continue;

        // All of the supported tokens for lines are <8 characters long, so to match them we can
        // copy the token into a 64-bit int, and do a switch statement instead of strcmping.
        if (token_length > sizeof(uint64_t)) {
            // ERROR: Unexpected token.
            return NULL;
        }

        uint64_t    shift  = 0;
        uint64_t    token  = 0;
        const char* cursor = tok.begin();
        while (cursor != tok.end()) {
            token |= *cursor << shift;
            shift += 8;
            ++cursor;
        }

        switch (static_cast<TokenType>(token)) {
            case TokenType::GEOMETRIC_VERT: parse_position(line, mesh.get()); break;
            case TokenType::TEXTURE_VERT: parse_texture_vert(line, mesh.get()); break;
            case TokenType::NORMAL_VERT: parse_normal_vert(line, mesh.get()); break;
            case TokenType::PARAM_SPACE_VERT: parse_space_vert(line, mesh.get()); break;
            case TokenType::POINT: break;
            case TokenType::LINE: break;
            case TokenType::FACE: parse_face(line, (const char*)stream->end, mesh.get()); break;
            case TokenType::CURVE: break;
            case TokenType::CURVE2D: break;
            case TokenType::SURFACE: break;
            case TokenType::GROUP_NAME: break;
            case TokenType::SMOOTHING_GROUP: break;
            case TokenType::MERGING_GROUP: break;
            case TokenType::OBJECT_NAME: break;
            case TokenType::MATERIAL: {
                printf("Found material line");
            } break;
        }
    }


    return static_cast<Box<ObjMesh>&&>(mesh);
}

}  // namespace loon
#include "obj_parser.h"

#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>

#include <cstring>
#include <unordered_map>

#include "geometry.h"
#include "string_view.h"

namespace loon {
#define LOON_OBJ_MAX_LINE_LENGTH BUFSIZ

LineReader::~LineReader() {
    if (m_file) {
        fclose(static_cast<FILE*>(m_file));
        free(m_buffer_begin);
    }
}

LineReader LineReader::from_file(const char* filename) {
    FILE* f = fopen(filename, "rb");

    LineReader reader;
    reader.m_file            = f;
    reader.m_buffer_begin    = (char*)malloc(BUFSIZ);
    reader.m_buffer_begin[0] = '\n';  // A bit of a hack - pretend we start with an empty line in
                                      // the front of the buffer. Makes next() logic simpler.
    reader.m_line_begin = reader.m_line_end = reader.m_buffer_begin;

    // Try to fill in the rest of the buffer with a load.
    size_t bytes_read   = fread(reader.m_buffer_begin + 1, 1, BUFSIZ - 1, f);
    reader.m_buffer_end = reader.m_buffer_begin + 1 + bytes_read;

    return reader;
}

LineReader LineReader::from_mem(void* mem, size_t size) {
    LineReader reader;
    reader.m_buffer_begin = static_cast<char*>(mem);
    reader.m_buffer_end   = static_cast<char*>(mem) + size;
    reader.m_line_begin   = reader.m_buffer_begin;

    reader.m_line_end = reader.m_line_begin;
    while (reader.m_line_end != reader.m_buffer_end && *reader.m_line_end != '\n')
        ++reader.m_line_end;

    return reader;
}

// Advance the line reader, updating line_start and line_end and consuming the line from the stream.
ObjErrorCode LineReader::next() {
    // Try to advance the line cursor, re-filling the buffer as needed.

    // Invariant: At start, m_line_end points to the end (\n) of the previous line?
    m_line_begin = ++m_line_end;
    while (m_line_end != m_buffer_end && *m_line_end != '\n') { m_line_end++; }

    if (m_line_end == m_buffer_end) {
        if (m_file) {
            if (feof((FILE*)m_file)) { return ObjErrorCode::ReadPastEOF; }
            // Move the line to the start of the buffer, before refilling
            const size_t num_bytes = m_buffer_end - m_line_begin;
            memmove(m_buffer_begin, m_line_begin, num_bytes);
            const size_t bytes_read =
                fread(m_buffer_begin + num_bytes, 1, BUFSIZ - num_bytes, (FILE*)m_file);
            m_buffer_end = m_buffer_begin + num_bytes + bytes_read;

            m_line_begin = m_buffer_begin;
            m_line_end   = m_line_begin;
            while (m_line_end != m_buffer_end && *m_line_end != '\n') { m_line_end++; }

            if (m_line_end == m_buffer_end) { return ObjErrorCode::LineOverflow; }
        } else {
        }
        // Need to reload the buffer
    }


    return ObjErrorCode::Success;
}

StringView LineReader::line() const {
    return StringView(m_line_begin, m_line_end);
}

bool is_blank(char x) {
    return x == ' ' || x == '\t' || x == '\r';
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
static int parse_int(StringView* tok) {
    // No good native library solution, just roll our own naive one for now.
    // We know that since we're returning a 32-bit signed integer, we have a maximum of 10 digits
    // plus an optional hyphen. More commonly, we will have less than 8 digits, which would fit into
    // a uint64_t.
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

static void parse_face(StringView line, ObjMesh* mesh) {
    // Faces are the most complicated thing to parse.
    // The format may just be a list of numbers, but it may have slashes separating texture coords
    // and vertex normals. As well, the length of the list isn't known ahead of time. How to
    // best handle this? Simplest way is to just assume it has texture and normal,
    int8_t num_face_verts = 0;

    consume_leading_whitespace(&line);

    while (!line.is_empty()) {
        // Anything remaining is the normal index.
        int32_t pos_index    = parse_int(&line);
        int32_t tex_index    = 0;
        int32_t normal_index = 0;
        if (line.front() == '/') {
            line.remove_prefix(1);
            tex_index = parse_int(&line);
        }
        if (line.front() == '/') {
            line.remove_prefix(1);
            normal_index = parse_int(&line);
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

Box<ObjMesh> obj_parse(LineReader& stream) {
    Box<ObjMesh> mesh = make_box<ObjMesh>(ObjMesh{
        .positions           = {},
        .texcoords           = {},
        .normals             = {},
        .face_vertex_count   = {},
        .face_pos_indices    = {},
        .face_tex_indices    = {},
        .face_normal_indices = {},
    });

    // OBJ is a line-centric file format. The parsing of each line depends on the first token
    while (stream.next() == ObjErrorCode::Success) {
        StringView line = stream.line();

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
            case TokenType::FACE: parse_face(line, mesh.get()); break;
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
struct Index {
    uint32_t p, t, n;

    bool operator==(const Index& rhs) const { return p == rhs.p && t == rhs.t && n == rhs.n; }
};

template <>
struct std::hash<Index> {
    size_t operator()(const Index& i) const noexcept {
        auto h = std::hash<uint32_t>();
        return h(i.p) ^ (h(i.n) << 1) ^ (h(i.t) << 2);
    }
};

namespace loon {
Box<ReindexedMesh> cleanup_mesh(Box<ObjMesh> mesh) {
    // TODO: Triangulation

    const size_t num_indices = mesh->face_pos_indices.size();



    // Feels like there should be a way to do this without a hash table using some cleverness,
    // but maybe not.
    std::unordered_map<Index, uint32_t> remap_table;

    std::vector<geometry::float3> positions_out;
    std::vector<geometry::float2> texture_out;
    std::vector<geometry::float3> normal_out;
    std::vector<uint32_t>         index_buffer_out;

    for (uint32_t idx_in = 0; idx_in < num_indices; ++idx_in) {
        const auto p_idx = mesh->face_pos_indices[idx_in];
        const auto t_idx = mesh->face_tex_indices[idx_in];
        const auto n_idx = mesh->face_tex_indices[idx_in];

        auto [it, inserted] = remap_table.try_emplace({p_idx, t_idx, n_idx}, positions_out.size());
        if (inserted) {
            const auto& p = mesh->positions[p_idx];
            const auto& t = mesh->texcoords[t_idx];
            const auto& n = mesh->normals[n_idx];
            positions_out.emplace_back(p.x, p.y, p.z);
            texture_out.emplace_back(t.x, t.y);
            normal_out.emplace_back(n);
        } else {
            index_buffer_out.push_back(it->second);
        }
    }

    return make_box<ReindexedMesh>(ReindexedMesh{
        .positions = std::move(positions_out),
        .texcoords = std::move(texture_out),
        .normals   = std::move(normal_out),
        .indices   = std::move(index_buffer_out),
    });
}

}  // namespace loon
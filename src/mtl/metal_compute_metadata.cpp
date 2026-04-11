#include "metal_compute_metadata.h"

// For simplicity, I'm just going to use string view for its search functions
// I'm not going to go all-in on boyer-moore substring searching or anything like that.
#include <charconv>
#include <string_view>

namespace loon::gpu {

static Dimension3D parse_dim(std::string_view str) {
    uint32_t x, y = 1, z = 1;
    std::from_chars(str.begin(), str.end(), x);
    size_t comma = str.find(',');
    if (comma != str.npos) { std::from_chars(str.begin() + comma + 1, str.end(), y); }
    comma = str.find(',', comma + 1);
    if (comma != str.npos) { std::from_chars(str.begin() + comma + 1, str.end(), z); }

    return {x, y, z};
}

// TODO: This is super hacky, need better testing around it.
// Would like to not need it at all, but what can ya do.
ShaderMetadata parse_metadata(Arena            arena,
                              Span<const char> metal_source,
                              Span<const char> entry_point) {
    // First, look for a copy of "entry_point", then go backwards to see if it's preceeded by
    // "void" - this is the kernel definition.
    const std::string_view source(metal_source.begin(), metal_source.end());
    const std::string_view needle(entry_point.begin(), entry_point.end());

    size_t entry_point_idx = 0;

    while (entry_point_idx != source.npos) {
        entry_point_idx = source.find(needle, entry_point_idx);
        // TODO:
        // Move backwards, skipping whitespace, then check for "void".
        // If we see it, then look for our attribute.
        // This would be more robust, but really, kernel entry points shouldn't be called like other
        // functions anyways, so as long as no one names a variable the same as our entry point we
        // should be okay.


        constexpr std::string_view attribute     = "required_threads_per_threadgroup(";
        auto                       attribute_pos = source.rfind(attribute, entry_point_idx);
        if (attribute_pos != source.npos) {
            attribute_pos += attribute.size();
            const auto attribute_end = source.find(')', attribute_pos);

            const std::string_view triplet
                = source.substr(attribute_pos, attribute_end - attribute_pos);

            return {.required_threadgroup_size = parse_dim(triplet)};
        }
        ++entry_point_idx;
    }
    return {
        .required_threadgroup_size = {.x = 0, .y = 0, .z = 0},
    };
}


}  // namespace loon::gpu

#include "filesystem.h"

#include <cstring>

namespace loon::filesystem {

[[nodiscard]] static bool is_alpha(char c) {
    // Ascii upper case is between 0x41 and 0x5A
    // Ascii lower case is between 0x61 and 0x7A
    // We can just clear a single bit to go to_upper, then do the range check.
    c &= (char)0xDF;
    return c >= 'A' && c <= 'Z';
}

[[nodiscard]] static StringView clone(Arena* a, StringView s) {
    char* output = reinterpret_cast<char*>(a->alloc(s.size()));
    if (!output) { return {}; }
    memcpy(output, s.data(), s.size());
    return StringView(output, s.size());
}

[[nodiscard]] static StringView concat(Arena* a, StringView head, StringView tail) {
    if ((uintptr_t)head.end() != a->current_ptr()) { head = clone(a, head); }
    return StringView(head.data(), head.size() + clone(a, tail).size());
}

StringView root_name(StringView path) {
    // If we start with 2 backslashes, they're part of the root name - we're on a samba sedrver or
    // something.
    if (path.starts_with("\\\\")) {
        const char* it = path.begin() + 2;
        while (it != path.end() && *it != '\\' && *it != '/') ++it;

        return StringView(path.begin(), it);
    } else if (path.size() >= 2 && path[1] == ':' && is_alpha(path[0])) {
        // If we've got a letter followed by a colon, we've got a windows disk letter.
        return StringView(path.begin(), 2);
    }

    return StringView();
}

StringView parent_path(StringView path) {
    size_t sep_idx = path.find_last_of("\\/");
    return path.substr(0, sep_idx);
}

StringView normalize_path(Arena* arena, StringView path, char preferred_separator) {
    if (path.is_empty()) { return path; }

    // First, copy over the root name.
    // Root names, can look like "C:" or like "//myserver"
    const auto root = root_name(path);
    path.remove_prefix(root.size());

    const StringView sep(&preferred_separator, 1);

    StringView result = clone(arena, root);
    if (path.front() == '/') { result = concat(arena, result, "/"); }


    const auto remove_next_path_segment = [](StringView& path) -> StringView {
        size_t start = 0;
        while (start < path.size() && (path[start] == '/' || path[start] == '\\')) start++;

        size_t end = start;
        while (end < path.size() && (path[end] != '/' && path[end] != '\\')) end++;

        StringView result = path.substr(start, end - start);
        path.remove_prefix(end);
        return result;
    };

    while (!path.is_empty()) {
        // Get next segment of path - segment is the next range of non-separator characters.
        auto segment = remove_next_path_segment(path);
        if (segment == "..") {
            // Strip the result to the last directory separator. Should also free that memory from
            // the arena to prevent realloc in the concat operator.
            size_t pos = result.size() - 1;
            while (pos > 0 && result[pos] != preferred_separator) --pos;
            arena->free(result.data() + pos, result.size() - pos);
            result.remove_suffix(result.size() - pos);

        } else if (segment == ".") {
            // Do nothing, just skip the next directory separator.
        } else {
            if (!result.is_empty() && result != "/") { result = concat(arena, result, sep); }
            result = concat(arena, result, segment);
        }
    }

    return result;
}

}  // namespace loon::filesystem
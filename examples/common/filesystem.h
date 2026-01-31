#pragma once

#include <cstdint>
#include <iterator>

#include "arena.h"
#include "string_view.h"

namespace loon::filesystem {

StringView root_name(StringView path);
StringView normalize_path(Arena* arena, StringView path, char preferred_separator = '/');

};  // namespace loon::filesystem
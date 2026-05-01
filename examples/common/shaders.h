#pragma once

#include <string_view>
#include <vector>

#include "common/box.h"

namespace slang {
struct TypeLayoutReflection;
};

struct ShaderModuleImpl;
struct ShaderModuleDeleter {
    void operator()(ShaderModuleImpl* ptr);
};
using ShaderModule = loon::Box<ShaderModuleImpl, ShaderModuleDeleter>;

std::vector<uint8_t> get_spirv(ShaderModuleImpl* module, const char* entry_point);

class ShaderLoader {
   public:
    ShaderLoader(std::string_view search_path, bool use_metal = false);
    ~ShaderLoader();

    ShaderModule load_module(std::string_view module_name);

    void reset_cache();

   private:
    class Impl;
    loon::Box<Impl> m_impl;
};

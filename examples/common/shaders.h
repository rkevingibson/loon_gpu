#pragma once

#include <memory>
#include <string_view>
#include <vector>


namespace slang {
struct TypeLayoutReflection;
};

struct ShaderModuleImpl;
struct ShaderModuleDeleter {
    void operator()(ShaderModuleImpl* ptr);
};
using ShaderModule = std::unique_ptr<ShaderModuleImpl, ShaderModuleDeleter>;

std::vector<uint8_t> get_spirv(ShaderModuleImpl* module, const char* entry_point);

class ShaderLoader {
   public:
    ShaderLoader(std::string_view search_path, bool use_metal = false);
    ~ShaderLoader();

    ShaderModule load_module(std::string_view module_name);

    void reset_cache();

   private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

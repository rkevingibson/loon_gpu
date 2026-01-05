#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "webgpu/webgpu.h"

namespace slang {
struct TypeLayoutReflection;
};

struct ShaderModuleImpl;
struct ShaderModuleDeleter {
    void operator()(ShaderModuleImpl* ptr);
};
using ShaderModule = std::unique_ptr<ShaderModuleImpl, ShaderModuleDeleter>;

std::vector<uint32_t>               get_spirv(ShaderModuleImpl* module, const char* entry_point);
std::span<WGPUBindGroupLayoutEntry> get_bind_group_layout_info(ShaderModuleImpl*, const char* name);

class ShaderLoader {
   public:
    ShaderLoader(std::string_view search_path);
    ~ShaderLoader();

    ShaderModule load_module(std::string_view module_name);

    void reset_cache();

   private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

struct ShaderObjectLayoutBuilder {
   public:
    ShaderObjectLayoutBuilder& dynamic_offsets(bool enabled, const char* binding_name = nullptr);

    WGPUBindGroupLayout build(WGPUDevice device);

   private:
    std::vector<WGPUBindGroupLayoutEntry> m_entries;
    uint32_t                              m_binding_index = 0;
    uint32_t                              m_set_index     = 0;

    void add_bindings_for_parameter_block(slang::TypeLayoutReflection* type_layout);
    void add_bindings_from(slang::TypeLayoutReflection* type_layout, uint32_t element_count);

    friend ShaderObjectLayoutBuilder create_bind_group_layout_builder(
        ShaderModuleImpl* module,
        const char*       parameter_block_name);
};

ShaderObjectLayoutBuilder create_bind_group_layout_builder(ShaderModuleImpl* module,
                                                           const char*       parameter_block_name);

class ShaderObject {};

class ShaderCursor {
   public:
    ShaderCursor field(const char* name);
    ShaderCursor field(int index);
    ShaderCursor element(int index);

    void write(const void* data, size_t size);

   private:
    slang::TypeLayoutReflection* m_type_layout                  = nullptr;
    size_t                       m_byte_offset                  = 0;
    uint32_t                     m_binding_range_index          = 0;
    uint32_t                     m_array_index_in_binding_range = 0;
    ShaderObject                 m_object                       = {};
};

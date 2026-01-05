#include "shaders.h"

#include <filesystem>
#include <string>

#include "slang-com-ptr.h"
#include "slang.h"
#include "webgpu/webgpu.h"

using namespace slang;


struct ShaderModuleImpl {
    Slang::ComPtr<IModule> module;
};
void ShaderModuleDeleter::operator()(ShaderModuleImpl* p) {
    delete p;
};

std::vector<uint32_t> get_spirv(ShaderModuleImpl* module, const char* entry_point_name) {
    auto                       m = module->module;
    Slang::ComPtr<IEntryPoint> entry_point{};
    auto result = m->findEntryPointByName(entry_point_name, entry_point.writeRef());

    if (SLANG_FAILED(result)) {
        fprintf(stderr,
                "Failed to get entry point %s in shader module: 0x%x 0x%x",
                entry_point_name,
                SLANG_GET_RESULT_FACILITY(result),
                SLANG_GET_RESULT_CODE(result));
        return {};
    }

    Slang::ComPtr<ISlangBlob>     diagnostics{};
    slang::IComponentType*        components[] = {m, entry_point};
    Slang::ComPtr<IComponentType> program;

    result = m->getSession()->createCompositeComponentType(components,
                                                           2,
                                                           program.writeRef(),
                                                           diagnostics.writeRef());
    if (SLANG_FAILED(result)) {
        fprintf(stderr,
                "Failed to create composite component: 0x%x 0x%x",
                SLANG_GET_RESULT_FACILITY(result),
                SLANG_GET_RESULT_CODE(result));
        return {};
    }
    Slang::ComPtr<slang::IComponentType> component{};
    result = program->link(component.writeRef(), diagnostics.writeRef());
    if (SLANG_FAILED(result)) {
        fprintf(stderr,
                "Failed to retrieve entry point code: 0x%x 0x%x",
                SLANG_GET_RESULT_FACILITY(result),
                SLANG_GET_RESULT_CODE(result));
        return {};
    }
    Slang::ComPtr<ISlangBlob> code{};
    result = component->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef());
    if (SLANG_FAILED(result)) {
        fprintf(stderr,
                "Failed to retrieve entry point code: 0x%x 0x%x",
                SLANG_GET_RESULT_FACILITY(result),
                SLANG_GET_RESULT_CODE(result));
        return {};
    }

    const uint32_t* begin = reinterpret_cast<const uint32_t*>(code->getBufferPointer());
    const size_t    size  = code->getBufferSize() / sizeof(uint32_t);
    return std::vector<uint32_t>(begin, begin + size);
}

class ShaderLoader::Impl {
   public:
    Impl(std::string_view search_path) : m_search_path(search_path) {
        SlangGlobalSessionDesc desc = {};
        createGlobalSession(&desc, m_global_session.writeRef());

        reset_cache();
    }

    ShaderModule load_module(std::string_view module_name) {
        for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(m_search_path)) {
            if (dir_entry.is_regular_file() && dir_entry.path().filename() == module_name) {
                // Found the shader file
                Slang::ComPtr<IBlob> diagnostics;
                auto                 module = Slang::ComPtr<IModule>(
                    m_session->loadModule(dir_entry.path().string().c_str(),
                                          diagnostics.writeRef()));
                if (diagnostics) {
                    fprintf(stderr,
                            "Shader diagnostic: %s",
                            (char*)diagnostics->getBufferPointer());
                }
                return ShaderModule(new ShaderModuleImpl{.module = module});
            }
        }

        return nullptr;
    }

    void reset_cache() {
        SessionDesc session_desc{};
        TargetDesc  target_description{};
        target_description.structureSize = sizeof(TargetDesc);
        target_description.format        = SLANG_SPIRV;
        target_description.profile       = m_global_session->findProfile("spirv_1_6");
        target_description.flags         = 0;

        CompilerOptionEntry options[] = {
            {
                .name  = slang::CompilerOptionName::VulkanUseEntryPointName,
                .value = {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
            },
            {
                .name  = slang::CompilerOptionName::EmitSpirvDirectly,
                .value = {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
            },
        };
        target_description.compilerOptionEntries    = options;
        target_description.compilerOptionEntryCount = sizeof(options) / sizeof(options[0]);

        session_desc.targets     = &target_description;
        session_desc.targetCount = 1;

        // TODO: Search paths for imports should be hooked up.
        session_desc.searchPaths     = nullptr;
        session_desc.searchPathCount = 0;

        session_desc.preprocessorMacros     = nullptr;
        session_desc.preprocessorMacroCount = 0;

        session_desc.compilerOptionEntries    = options;
        session_desc.compilerOptionEntryCount = sizeof(options) / sizeof(options[0]);

        m_session = nullptr;
        m_global_session->createSession(session_desc, m_session.writeRef());
    }

   private:
    Slang::ComPtr<IGlobalSession> m_global_session;
    Slang::ComPtr<ISession>       m_session;
    std::string                   m_search_path;
};

ShaderLoader::ShaderLoader(std::string_view search_path) {
    m_impl = std::make_unique<Impl>(search_path);
}

ShaderLoader::~ShaderLoader() = default;

ShaderModule ShaderLoader::load_module(std::string_view module_name) {
    return m_impl->load_module(module_name);
}

void ShaderLoader::reset_cache() {
    return m_impl->reset_cache();
}

ShaderObjectLayoutBuilder create_bind_group_layout_builder(ShaderModuleImpl* module,
                                                           const char*       parameter_block_name) {
    auto                         m             = module->module;
    slang::ProgramLayout*        spirv_layout  = m->getLayout(0);
    slang::TypeLayoutReflection* global_params = spirv_layout->getGlobalParamsTypeLayout();

    VariableLayoutReflection* cameraDataField
        = global_params->getFieldByIndex(global_params->findFieldIndexByName(parameter_block_name));
    const auto set_index = cameraDataField->getOffset(slang::ParameterCategory::RegisterSpace);

    auto type_layout = cameraDataField->getTypeLayout()->getElementTypeLayout();
    ShaderObjectLayoutBuilder layout_builder{};
    layout_builder.add_bindings_for_parameter_block(type_layout);
    layout_builder.m_set_index = set_index;
    return layout_builder;
}

ShaderObjectLayoutBuilder& ShaderObjectLayoutBuilder::dynamic_offsets(bool        enabled,
                                                                      const char* binding_name) {
    //
    if (binding_name == nullptr) {
        m_entries[0].buffer.hasDynamicOffset = enabled;
    } else {
        // TODO: Find the appropriate binding based on the name.
    }
    return *this;
}

WGPUBindGroupLayout ShaderObjectLayoutBuilder::build(WGPUDevice device) {
    WGPUBindGroupLayoutDescriptor descriptor{
        .nextInChain = nullptr,
        .label       = WGPU_STRING_VIEW_INIT,
        .entryCount  = m_entries.size(),
        .entries     = m_entries.data(),
    };
    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

void ShaderObjectLayoutBuilder::add_bindings_for_parameter_block(
    slang::TypeLayoutReflection* type_layout) {
    if (auto size = type_layout->getSize()) {
        printf("Binding %d: uniform buffer with size %zu\n", m_binding_index, size);
        WGPUBindGroupLayoutEntry entry = {
            .nextInChain      = nullptr,
            .binding          = m_binding_index++,
            .visibility       = WGPUShaderStage_None,
            .bindingArraySize = 1,
            .buffer           = {.nextInChain      = nullptr,
                                 .type             = WGPUBufferBindingType_Uniform,
                                 .hasDynamicOffset = false,
                                 .minBindingSize   = size},
            .sampler          = {},
            .texture          = {},
            .storageTexture   = {},
        };
        m_entries.emplace_back(std::move(entry));
    }
    add_bindings_from(type_layout, 1);
}

void ShaderObjectLayoutBuilder::add_bindings_from(slang::TypeLayoutReflection* type_layout,
                                                  uint32_t                     element_count) {
    int binding_range_count = static_cast<int>(type_layout->getBindingRangeCount());
    printf("Adding %d bindings:\n", binding_range_count);
    // for (int i = 0; i < binding_range_count; ++i) {
    //     WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    //     entry.binding                  = m_binding_index++;
    //     auto type                      = type_layout->getBindingRangeType(i);
    //     auto count                     = type_layout->getBindingRangeBindingCount(i);
    //     switch (type) {
    //         case slang::BindingType::Sampler:
    //             entry.sampler = {
    //                 .nextInChain = nullptr,
    //                 .type        = WGPUSamplerBindingType_Filtering,
    //             };
    //             break;
    //         case slang::BindingType::Texture:
    //             // Use resource shape to get the texture viewDimension.
    //             type_layout->getResourceShape();  // Has a flag for multisample as well
    //             type_layout->getResourceAccess();
    //             type_layout->getResourceResultType();  // Can map to sampleType (float, int, etc)
    //             entry.texture = {
    //                 .nextInChain   = nullptr,
    //                 .sampleType    =,
    //                 .viewDimension =,
    //                 .multisampled  =,
    //             };
    //             break;
    //         case slang::BindingType::ConstantBuffer: <#code #> break;
    //         case slang::BindingType::ParameterBlock: <#code #> break;
    //         case slang::BindingType::TypedBuffer: <#code #> break;
    //         case slang::BindingType::RawBuffer: <#code #> break;
    //         case slang::BindingType::CombinedTextureSampler: <#code #> break;
    //         case slang::BindingType::InputRenderTarget: <#code #> break;
    //         case slang::BindingType::InlineUniformData: <#code #> break;
    //         case slang::BindingType::RayTracingAccelerationStructure: <#code #> break;
    //         case slang::BindingType::VaryingInput: <#code #> break;
    //         case slang::BindingType::VaryingOutput: <#code #> break;
    //         case slang::BindingType::ExistentialValue: <#code #> break;
    //         case slang::BindingType::PushConstant: <#code #> break;
    //         case slang::BindingType::MutableFlag: <#code #> break;
    //         case slang::BindingType::MutableTexture: <#code #> break;
    //         case slang::BindingType::MutableTypedBuffer: <#code #> break;
    //         case slang::BindingType::MutableRawBuffer: <#code #> break;
    //         default: fprintf(stderr, "Unknown binding type\n");
    //     }
    //     printf("Binding %d: type %d, count %lld\n", entry.binding, type, count);
    // }
}

// MARK: ShaderCursor
ShaderCursor ShaderCursor::field(const char* name) {
    const int idx = static_cast<int>(m_type_layout->findFieldIndexByName(name));
    return field(idx);
}
ShaderCursor ShaderCursor::field(int index) {
    slang::VariableLayoutReflection* field = m_type_layout->getFieldByIndex(index);

    ShaderCursor result  = *this;
    result.m_type_layout = field->getTypeLayout();
    result.m_byte_offset += field->getOffset();
    result.m_binding_range_index += m_type_layout->getFieldBindingRangeOffset(index);
    return result;
}

ShaderCursor ShaderCursor::element(int index) {
    slang::TypeLayoutReflection* elementTypeLayout = m_type_layout->getElementTypeLayout();

    ShaderCursor result  = *this;
    result.m_type_layout = elementTypeLayout;
    result.m_byte_offset += index * elementTypeLayout->getStride();

    result.m_array_index_in_binding_range *= m_type_layout->getElementCount();
    result.m_array_index_in_binding_range += index;

    return result;
}

void ShaderCursor::write(const void* data, size_t size) {}
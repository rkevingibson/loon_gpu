#include "shaders.h"

#include <fstream>
#include <iterator>
#include <string>

#include "slang-com-ptr.h"
#include "slang.h"

using namespace slang;


struct ShaderModuleImpl {
    Slang::ComPtr<IModule> module;
    std::vector<uint8_t>   metal_source;
};
void ShaderModuleDeleter::operator()(ShaderModuleImpl* p) {
    delete p;
};

std::vector<uint8_t> get_spirv(ShaderModuleImpl* module, const char* entry_point_name) {
    if (module->module) {
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
                    "Failed to create composite component: 0x%x 0x%x\n",
                    SLANG_GET_RESULT_FACILITY(result),
                    SLANG_GET_RESULT_CODE(result));
            fprintf(stderr, "Shader diagnostic: %s\n", (char*)diagnostics->getBufferPointer());
            return {};
        }
        Slang::ComPtr<slang::IComponentType> component{};
        result = program->link(component.writeRef(), diagnostics.writeRef());
        if (SLANG_FAILED(result)) {
            fprintf(stderr,
                    "Failed to link program: 0x%x 0x%x\n",
                    SLANG_GET_RESULT_FACILITY(result),
                    SLANG_GET_RESULT_CODE(result));
            fprintf(stderr, "Shader diagnostic: %s\n", (char*)diagnostics->getBufferPointer());
            return {};
        }
        Slang::ComPtr<ISlangBlob> code{};
        result = component->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef());
        if (SLANG_FAILED(result)) {
            fprintf(stderr,
                    "Failed to retrieve entry point code: 0x%x 0x%x\n",
                    SLANG_GET_RESULT_FACILITY(result),
                    SLANG_GET_RESULT_CODE(result));
            fprintf(stderr, "Shader diagnostic: %s\n", (char*)diagnostics->getBufferPointer());
            return {};
        }

        const uint8_t* begin = reinterpret_cast<const uint8_t*>(code->getBufferPointer());
        const size_t   size  = code->getBufferSize();
        return std::vector<uint8_t>(begin, begin + size);
    } else {
        return module->metal_source;
    }
}

std::vector<uint8_t> read_file_into_vec(const std::string& filepath) {
    std::ifstream                  fs(filepath);
    std::istreambuf_iterator<char> it(fs);
    std::vector<uint8_t>           result(it, std::istreambuf_iterator<char>());
    return result;
}

class ShaderLoader::Impl {
   public:
    Impl(std::string_view search_path, bool use_metal) :
        m_search_path(search_path), m_use_metal(use_metal) {
        if (!use_metal) {
            SlangGlobalSessionDesc desc = {};
            createGlobalSession(&desc, m_global_session.writeRef());
            reset_cache();
        }
    }

    ShaderModule load_module(std::string_view module_name) {
        if (m_use_metal) {
            std::string name = m_search_path + "/" + std::string(module_name) + ".metal";

            return ShaderModule(new ShaderModuleImpl{
                .module       = nullptr,
                .metal_source = read_file_into_vec(name),
            });
        } else {
            std::string          name = std::string(module_name);
            Slang::ComPtr<IBlob> diagnostics;
            auto                 module =
                Slang::ComPtr<IModule>(m_session->loadModule(name.c_str(), diagnostics.writeRef()));
            if (diagnostics) {
                fprintf(stderr, "Shader diagnostic: %s", (char*)diagnostics->getBufferPointer());
            }
            return ShaderModule(new ShaderModuleImpl{.module = module});
        }
    }

    void reset_cache() {
        if (m_use_metal) return;
        SessionDesc session_desc{};
        TargetDesc  target_description{};
        target_description.structureSize = sizeof(TargetDesc);
        if (m_use_metal) {
            target_description.format = SLANG_METAL;
        } else {
            target_description.format  = SLANG_SPIRV;
            target_description.profile = m_global_session->findProfile("spirv_1_6");
        }
        target_description.flags = 0;

        CompilerOptionEntry options[] = {
            {
                .name  = slang::CompilerOptionName::VulkanUseEntryPointName,
                .value = {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
            },
            {
                .name  = slang::CompilerOptionName::EmitSpirvDirectly,
                .value = {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
            },
            {
                .name  = slang::CompilerOptionName::GLSLForceScalarLayout,
                .value = {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
            },
            {.name  = slang::CompilerOptionName::DebugInformation,
             .value = {slang::CompilerOptionValueKind::Int,
                       SlangDebugInfoLevel::SLANG_DEBUG_INFO_LEVEL_STANDARD,
                       0,
                       nullptr,
                       nullptr}}};
        target_description.compilerOptionEntries    = options;
        target_description.compilerOptionEntryCount = sizeof(options) / sizeof(options[0]);

        session_desc.targets     = &target_description;
        session_desc.targetCount = 1;

        session_desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;

        // TODO: Search paths for imports should be hooked up.
        const char* searchPaths[] = {
            m_search_path.c_str(),
        };
        session_desc.searchPaths     = searchPaths;
        session_desc.searchPathCount = 1;

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
    bool                          m_use_metal;
};

ShaderLoader::ShaderLoader(std::string_view search_path, bool use_metal) {
    m_impl = loon::make_box<Impl>(search_path, use_metal);
}

ShaderLoader::~ShaderLoader() = default;

ShaderModule ShaderLoader::load_module(std::string_view module_name) {
    return m_impl->load_module(module_name);
}

void ShaderLoader::reset_cache() {
    return m_impl->reset_cache();
}

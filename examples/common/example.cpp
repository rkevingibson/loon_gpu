#include "example.h"

#define STB_IMAGE_IMPLEMENTATION

#if __APPLE__
#    include <mach-o/dyld.h>
#endif

#include <filesystem>
#include <memory>

#include "hello_cube/hello_cube.h"
#include "hello_triangle/hello_triangle.h"
#include "stb_image.h"
#include "textured_cube/textured_cube.h"

static std::filesystem::path get_executable_directory() {
#if __APPLE__
    uint32_t bufsize = 0;
    _NSGetExecutablePath(nullptr, &bufsize);
    std::vector<char> path(bufsize);
    _NSGetExecutablePath(path.data(), &bufsize);
    return std::filesystem::path(path.begin(), path.end()).parent_path();
#endif
}

FilePaths default_file_paths() {
    auto exec_dir = get_executable_directory();

    auto default_shader_dir = (exec_dir / "../../../examples").lexically_normal();
    auto default_asset_dir  = (exec_dir / "../../../assets").lexically_normal();

    return {
        .asset_directory  = default_asset_dir.string(),
        .shader_directory = default_shader_dir.string(),
    };
}

void log_callback(loon::gpu::LogLevel lvl, loon::gpu::Span<const char> message, void* userdata) {
    fprintf(stderr, "%.*s\n", static_cast<int>(message.size()), message.data());
}

std::unique_ptr<Example> create_example(ExampleName name, const WindowState& state) {
    switch (name) {
        case ExampleName::HelloTriangle: return std::make_unique<HelloTriangle>(state);
        case ExampleName::HelloCube: return std::make_unique<HelloCube>(state);
        case ExampleName::TexturedCube: return std::make_unique<TexturedCube>(state);
        default: return nullptr;
    }
}
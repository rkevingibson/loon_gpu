#include "example.h"

#include "filesystem.h"

#define STB_IMAGE_IMPLEMENTATION

#if __APPLE__
#    include <mach-o/dyld.h>
#elif _WIN32
extern "C" __declspec(dllimport) unsigned long __stdcall GetModuleFileNameA(void*,
                                                                            char*,
                                                                            unsigned long);
#else
#    include <unistd.h>
#endif

#include <memory>
#include <vector>

#include "hello_cube/hello_cube.h"
#include "hello_triangle/hello_triangle.h"
#include "many_cubes/many_cubes.h"
#include "particle_emitter/particle_emitter.h"
#include "stb_image.h"
#include "string_view.h"
#include "textured_cube/textured_cube.h"


static std::string get_executable_directory() {
#if __APPLE__
    uint32_t bufsize = 0;
    _NSGetExecutablePath(nullptr, &bufsize);
    std::vector<char> path(bufsize);
    _NSGetExecutablePath(path.data(), &bufsize);
#elif _WIN32
    std::vector<char> path(256);
    uint32_t          bufsize = GetModuleFileNameA(nullptr, path.data(), (uint32_t)path.size()) + 1;
#else
    uint32_t          bufsize = 256;
    std::vector<char> path(bufsize, '\0');
    ssize_t           result = 0;
    do {
        result = readlink("/proc/self/exe", path.data(), bufsize);
        if (result > 0 && static_cast<uint32_t>(result) == bufsize) {
            // Truncation possible
            bufsize *= 2;
            path.resize(bufsize, '\0');
        } else if (result > 0) {
            bufsize = static_cast<uint32_t>(result) + 1;
        }
    } while (result == bufsize || result == -1);

#endif
    auto parent = loon::filesystem::parent_path(loon::StringView(path.data(), bufsize - 1));
    return std::string(parent.begin(), parent.end());
}

FilePaths default_file_paths() {
    auto exec_dir = get_executable_directory();

    char        arena_buf[2048];
    loon::Arena arena(arena_buf, 2048);
    auto        default_shader_dir
        = loon::filesystem::normalize_path(&arena,
                                           (exec_dir + "/../../../examples/shaders/").c_str());
    auto default_asset_dir
        = loon::filesystem::normalize_path(&arena, (exec_dir + "/../../../assets/").c_str());

    return {
        .shader_directory = std::string(default_shader_dir.data(), default_shader_dir.size()),
        .asset_directory  = std::string(default_asset_dir.data(), default_asset_dir.size()),
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
        case ExampleName::ParticleEmitter: return std::make_unique<ParticleEmitter>(state);
        case ExampleName::ManyCubes: return std::make_unique<ManyCubes>(state);
        default: return nullptr;
    }
}
#pragma once

#include <gpu/loon_gpu.h>

#include <filesystem>
#include <memory>

class ShaderLoader;

struct FilePaths {
    std::filesystem::path shader_directory;
    std::filesystem::path asset_directory;
};

struct WindowState {
    uintptr_t                     native_window_handle;
    uintptr_t                     native_instance_handle;
    uint16_t                      width;
    uint16_t                      height;
    std::unique_ptr<ShaderLoader> shader_loader;
    FilePaths                     file_paths;
};

class Example {
   public:
    virtual ~Example() {};
    virtual void Update(const WindowState& window) = 0;
};

enum class ExampleName {
    HelloTriangle,
    HelloCube,
    TexturedCube,

    Count,
};

FilePaths default_file_paths();

void log_callback(loon::gpu::LogLevel lvl, loon::gpu::Span<const char> message, void* userdata);

std::unique_ptr<Example> create_example(ExampleName name, const WindowState& state);
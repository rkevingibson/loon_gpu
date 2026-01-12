#pragma once

#include <gpu/loon_gpu.h>

#include <memory>

class ShaderLoader;

struct WindowState {
    uintptr_t                     native_window_handle;
    uintptr_t                     native_instance_handle;
    uint16_t                      width;
    uint16_t                      height;
    std::unique_ptr<ShaderLoader> shader_loader;
};

class Example {
   public:
    virtual ~Example() {};
    virtual void Update(const WindowState& window) = 0;
};

enum class ExampleName {
    HelloTriangle,
    HelloCube,

    Count,
};

void log_callback(loon::gpu::LogLevel lvl, loon::gpu::Span<const char> message, void* userdata);

std::unique_ptr<Example> create_example(ExampleName name, const WindowState& state);
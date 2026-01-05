#pragma once

#include <memory>
#include <string>

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

std::unique_ptr<Example> create_example(ExampleName name, const WindowState& state);
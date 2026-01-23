#include "example.h"

#define STB_IMAGE_IMPLEMENTATION

#include <memory>

#include "hello_cube/hello_cube.h"
#include "hello_triangle/hello_triangle.h"
#include "stb_image.h"
#include "textured_cube/textured_cube.h"


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
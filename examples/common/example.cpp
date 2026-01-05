#include "example.h"

#include "hello_cube/hello_cube.h"
#include "hello_triangle/hello_triangle.h"

std::unique_ptr<Example> create_example(ExampleName name, const WindowState& state) {
    switch (name) {
        case ExampleName::HelloTriangle: return std::make_unique<HelloTriangle>(state);
        case ExampleName::HelloCube: return std::make_unique<HelloCube>(state);
        default: return nullptr;
    }
}
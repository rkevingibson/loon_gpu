# Split some of these into a separate library so we can write some unit tests for them
add_library(example_utilities OBJECT
    common/geometry.cpp
    common/filesystem.cpp
)

target_include_directories(example_utilities PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

set(EXAMPLE_SOURCES
    common/example.cpp
    common/shaders.cpp
    common/gpu_args.cpp
    hello_triangle/hello_triangle.cpp
    hello_cube/hello_cube.cpp
    textured_cube/textured_cube.cpp
    particle_emitter/particle_emitter.cpp
    many_cubes/many_cubes.cpp

    imgui/imgui_draw.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui.cpp
    imgui/imgui_impl_loon.cpp
 )

if (WIN32)
    list(APPEND EXAMPLE_SOURCES 
        common/example_win32.cpp
        imgui/imgui_impl_win32.cpp
        )
elseif(APPLE)
    enable_language(OBJCXX)
    list(APPEND EXAMPLE_SOURCES 
        common/example_macos.mm 
        imgui/imgui_impl_osx.mm
        )
elseif(LINUX)
    list(APPEND EXAMPLE_SOURCES
        common/example_linux.cpp
    )
endif()

add_executable(
    loon_gpu_examples
    ${EXAMPLE_SOURCES}
)

if (WIN32)
    target_link_options(loon_gpu_examples PRIVATE "/SUBSYSTEM:WINDOWS")
endif()

target_link_libraries(
    loon_gpu_examples
    PRIVATE
    loon_gpu
    slang
    example_utilities
)

if (APPLE)
    target_link_libraries(loon_gpu_examples
    PRIVATE
    "-framework Foundation"
    "-framework AppKit"
    "-framework Metal"
    "-framework MetalKit"
    "-framework QuartzCore"
    "-framework GameController"
    )
endif()

target_include_directories(loon_gpu_examples 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR})
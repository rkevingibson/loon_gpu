# If we're building examples, we need a copy of slang.
if(WIN32)
    set(SLANG_URL
        https://github.com/shader-slang/slang/releases/download/v2025.24.3/slang-2025.24.3-windows-x86_64.zip
    )
elseif(APPLE)
    set(SLANG_URL
        https://github.com/shader-slang/slang/releases/download/v2025.24.3/slang-2025.24.3-macos-aarch64.tar.gz)
elseif(UNIX)
set(SLANG_URL
    https://github.com/shader-slang/slang/releases/download/v2025.10/slang-2025.10-linux-x86_64.zip
)
endif()

FetchContent_Declare(slang URL ${SLANG_URL} DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

FetchContent_MakeAvailable(slang)

add_library(slang SHARED IMPORTED GLOBAL)
target_include_directories(slang INTERFACE ${slang_SOURCE_DIR}/include)

if(WIN32)
    set_target_properties(
        slang
        PROPERTIES
            IMPORTED_IMPLIB ${slang_SOURCE_DIR}/lib/slang.lib
            IMPORTED_LOCATION ${slang_SOURCE_DIR}/bin/slang.dll
    )
    # TODO: Copy the dlls to the install directory on install.

    set(SLANG_DLLS slang.dll slang-llvm.dll slang-rt.dll slang-glslang.dll slang-compiler.dll)

    list(
        TRANSFORM SLANG_DLLS
        PREPEND ${slang_SOURCE_DIR}/bin/
        OUTPUT_VARIABLE SLANG_SOURCE_DLLS
    )
    list(
        TRANSFORM SLANG_DLLS
        PREPEND ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/
        OUTPUT_VARIABLE SLANG_OUTPUT_DLLS
    )

    add_custom_command(
        OUTPUT ${SLANG_OUTPUT_DLLS}
        COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different ${SLANG_SOURCE_DLLS}
            "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/"
    )

    add_custom_target(slang_copy ALL DEPENDS ${SLANG_OUTPUT_DLLS})
elseif(APPLE)
    set_target_properties(
        slang
        PROPERTIES IMPORTED_IMPLIB ${slang_SOURCE_DIR}/lib/libslang.dylib
    )
    target_link_directories(slang INTERFACE ${slang_SOURCE_DIR}/lib)
else()
    set_target_properties(
        slang
        PROPERTIES IMPORTED_IMPLIB ${slang_SOURCE_DIR}/lib/libslang.so
    )
endif()

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
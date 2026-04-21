include(FetchContent)

### Vulkan Headers

FetchContent_Declare(
    vulkan
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers
    GIT_TAG vulkan-sdk-1.4.341.0
)
FetchContent_MakeAvailable(vulkan)

### Volk

FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG vulkan-sdk-1.4.341
)

if(WIN32)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
elseif(LINUX)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_XLIB_KHR)
elseif(APPLE)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_METAL_EXT)
endif()

set(VOLK_PULL_IN_VULKAN)
FetchContent_MakeAvailable(volk)

### VMA

FetchContent_Declare(
    vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.3.0
)

FetchContent_MakeAvailable(vma)

### Metal-cpp

if(APPLE)
    add_library(metal-cpp INTERFACE)
    target_include_directories(metal-cpp INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

endif()

### Slang

# Only needed if we're building examples or tests for Vulkan
if(LOON_GPU_BUILD_EXAMPLES OR LOON_GPU_BUILD_EXAMPLES)

if(WIN32)
    set(SLANG_URL
        https://github.com/shader-slang/slang/releases/download/v2025.24.3/slang-2025.24.3-windows-x86_64.zip
    )
elseif(APPLE)
    set(SLANG_URL
        https://github.com/shader-slang/slang/releases/download/v2025.24.3/slang-2025.24.3-macos-aarch64.tar.gz)
elseif(UNIX)
set(SLANG_URL
    https://github.com/shader-slang/slang/releases/download/v2025.24.3/slang-2025.24.3-linux-x86_64.zip
)
endif()

FetchContent_Declare(slang URL ${SLANG_URL} DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

FetchContent_MakeAvailable(slang)

add_library(slang SHARED IMPORTED GLOBAL)
target_include_directories(slang INTERFACE ${slang_SOURCE_DIR}/include)

if (WIN32)
    set(SLANGC_COMMAND "${slang_SOURCE_DIR}/bin/slangc.exe" PARENT_SCOPE )
else() 
    set(SLANGC_COMMAND "${slang_SOURCE_DIR}/bin/slangc" PARENT_SCOPE )
endif()

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


endif()
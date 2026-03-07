
cmake_minimum_required(VERSION 3.23)

project(loon_gpu VERSION 0.01 LANGUAGES CXX)

option(LOON_GPU_BUILD_EXAMPLES "Build examples" OFF)
option(LOON_GPU_BUILD_TESTS "Build tests" OFF)
option(LOON_GPU_BUILD_DOCS "Build documentation" OFF)
option(LOON_GPU_WERROR "Treat warnings as errors" OFF)
option(LOON_GPU_INSTALL "Install library" ON)
option(LOON_GPU_USE_KOSMICKRISP "Use the kosmickrisp vulkan driver on MacOS" OFF)

set(CLANG_TIDY_CONFIG clang-tidy --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(third_party)

add_library(
    loon_gpu STATIC
    src/containers.cpp
    src/platform_utils.cpp
)
set(PUBLIC_HEADERS
    include/gpu/loon_gpu.h
)
set_target_properties(
    loon_gpu
    PROPERTIES
    PUBLIC_HEADER "${PUBLIC_HEADERS}"
)
target_include_directories(
    loon_gpu
    PRIVATE ./src
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set(SOURCES
    src/vk/loon_gpu.cpp
    src/vk/vma_usage.cpp
    src/vk/gpu_to_vk.cpp
)

target_sources(
    loon_gpu
    PRIVATE
    ${SOURCES}
)

target_link_libraries(loon_gpu PRIVATE volk VulkanMemoryAllocator)

# We use device-specific loader, can define this to avoid accidentally accessing an unloaded function
target_compile_definitions(
    loon_gpu
    PRIVATE VOLK_NO_DEVICE_PROTOTYPES
)

set (TARGETS loon_gpu volk VulkanMemoryAllocator)

if (LOON_GPU_WERROR)
    if (MSVC)
        target_compile_options(loon_gpu PRIVATE /WX)
    else()
        target_compile_options(loon_gpu PRIVATE -Werror)
    endif()
endif()

if (LOON_GPU_BUILD_TESTS)
    message("Building Loon GPU unit tests")
    enable_testing()
    add_subdirectory(tests)
    list(APPEND TARGETS loon_gpu_tests)
endif()

if (LOON_GPU_BUILD_EXAMPLES)
    message("Building Loon GPU examples")
    add_subdirectory(examples)
    list(APPEND TARGETS loon_gpu_examples)
endif()

if(LOON_GPU_BUILD_DOCS)
    message("Building Loon GPU documentation")
    add_subdirectory(docs)
endif()

if (LOON_GPU_INSTALL)
    include(GNUInstallDirs)

    install(
        TARGETS ${TARGETS}
        EXPORT loonwebgpuTargets 
        COMPONENT loon_gpu
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/webgpu
    )

    install(EXPORT loonwebgpuTargets COMPONENT loon_gpu DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/loon_gpu NAMESPACE loon_gpu::)

    include(CMakePackageConfigHelpers)

    configure_package_config_file(
        config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/loon_gpuConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/loon_gpu NO_SET_AND_CHECK_MACRO
    )
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/loon_gpuConfigVersion.cmake COMPATIBILITY ExactVersion)
    install(FILES
	    ${CMAKE_CURRENT_BINARY_DIR}/loon_gpuConfig.cmake
	    ${CMAKE_CURRENT_BINARY_DIR}/loon_gpuConfigVersion.cmake
	    COMPONENT loon_gpu
	    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/loon_gpu
    )
endif()
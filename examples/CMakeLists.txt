include(FetchContent)

# We use Github Releases to store some larger example assets outside of git. We
# download them during configuration (about 90MB)

FetchContent_Declare(
  example_assets
  URL https://github.com/rkevingibson/loon_gpu/releases/download/v0.1.0/example_assets_v0_1_0.zip
  URL_HASH
    SHA256=0fb40a67d2af90631eed08fb3f2c7dbe3db5aa26b26dc8b05d6476f4e0009f55)
FetchContent_MakeAvailable(example_assets)

# Split some of these into a separate library so we can write some unit tests
# for them
add_library(example_utilities OBJECT common/geometry.cpp common/filesystem.cpp
                                     common/timer.cpp common/obj_parser.cpp)

target_include_directories(example_utilities
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

set(EXAMPLE_SOURCES
    common/example.cpp
    common/shaders.cpp
    common/gpu_args.cpp
    bunny/bunny.cpp
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
    imgui/imgui_impl_loon.cpp)

if(WIN32)
  list(APPEND EXAMPLE_SOURCES common/example_win32.cpp
       imgui/imgui_impl_win32.cpp)
elseif(APPLE)
  enable_language(OBJCXX)
  list(APPEND EXAMPLE_SOURCES common/example_macos.mm imgui/imgui_impl_osx.mm)
elseif(LINUX)
  list(APPEND EXAMPLE_SOURCES common/example_linux.cpp)
endif()

add_executable(loon_gpu_examples ${EXAMPLE_SOURCES})

if(WIN32)
  target_link_options(loon_gpu_examples PRIVATE "/SUBSYSTEM:WINDOWS")
endif()

target_link_libraries(loon_gpu_examples PRIVATE loon_gpu slang
                                                example_utilities)

if(APPLE)
  target_link_libraries(
    loon_gpu_examples
    PRIVATE "-framework Foundation" "-framework AppKit" "-framework Metal"
            "-framework MetalKit" "-framework QuartzCore"
            "-framework GameController")
endif()

target_include_directories(loon_gpu_examples
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Symlinks are used to let the executable see the assets and shaders from the
# build directory without making a copy. This lets us hot-reload shaders very
# easily without needing a fresh build. On install, we will make a copy of the
# shaders and assets.

set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_LINK_DIR "$<TARGET_FILE_DIR:loon_gpu_examples>/shaders")
set(ASSET_LINK_DIR "$<TARGET_FILE_DIR:loon_gpu_examples>/assets")
if(WIN32)
  # Convert paths to native Windows backslash layout
  file(TO_NATIVE_PATH "${SHADER_SOURCE_DIR}" NATIVE_SRC)
  file(TO_NATIVE_PATH "${SHADER_LINK_DIR}" NATIVE_LINK)
  file(TO_NATIVE_PATH "${example_assets_SOURCE_DIR}" NATIVE_ASSET_SRC)
  file(TO_NATIVE_PATH "${ASSET_LINK_DIR}" NATIVE_ASSET_LINK)

  # Windows: Use a Directory Junction (/J) via cmd.exe (No Admin Required)
  add_custom_command(
    TARGET loon_gpu_examples
    POST_BUILD
    COMMAND
      cmd.exe /c
      "if not exist \"${NATIVE_LINK}\" mklink /J \"${NATIVE_LINK}\" \"${NATIVE_SRC}\""
    COMMENT "Creating Windows Directory Junction for local shader iteration")
  add_custom_command(
    TARGET loon_gpu_examples
    POST_BUILD
    COMMAND
      cmd.exe /c
      "if not exist \"${NATIVE_ASSET_LINK}\" mklink /J \"${NATIVE_ASSET_LINK}\" \"${NATIVE_ASSET_SRC}\""
    COMMENT "Creating Windows Directory Junction for local assets")
else()
  # Unix-like (Linux/macOS): Standard Symbolic Link
  add_custom_command(
    TARGET loon_gpu_examples
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${SHADER_SOURCE_DIR}"
            "${SHADER_LINK_DIR}"
    COMMENT "Creating symbolic link for local shader iteration")

  add_custom_command(
    TARGET loon_gpu_examples
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${example_assets_SOURCE_DIR}"
            "${ASSET_LINK_DIR}"
    COMMENT "Creating symbolic link for local asset iteration")
endif()

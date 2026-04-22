# Compile shaders using slang to spirv

function(add_shader shader output_file_list)
  cmake_path(GET shader FILENAME filename)
  set(output_file ${filename}.inc.h)
  if(APPLE AND NOT LOON_GPU_USE_KOSMICKRISP)
    message("Using metal shaders")
    cmake_path(SET input_file NORMALIZE
               ${CMAKE_CURRENT_SOURCE_DIR}/${shader}.metal)
    add_custom_command(
      OUTPUT "${output_file}"
      COMMAND xxd -i -n ${filename}_bytes ${input_file} ${output_file}
      COMMAND_EXPAND_LISTS
      DEPENDS ${shader}.metal DEPENDS_EXPLICIT_ONLY)
  else()
    message("Using slang shaders")
    set(SLANGC_ARGS
        -target
        spirv
        -profile
        spirv_1_6
        -fvk-use-entrypoint-name
        -emit-spirv-directly
        -fvk-use-scalar-layout
        -source-embed-style
        u8)
    cmake_path(SET input_file NORMALIZE
               ${CMAKE_CURRENT_SOURCE_DIR}/${shader}.slang)
    add_custom_command(
      OUTPUT "${output_file}"
      COMMAND ${SLANGC_COMMAND} ${SLANGC_ARGS} ${input_file} -source-embed-name
              ${filename}_bytes -o ${output_file}
      COMMAND_EXPAND_LISTS
      DEPENDS ${shader}.slang DEPENDS_EXPLICIT_ONLY)
  endif()
  list(APPEND ${output_file_list} "${output_file}")
  set(${output_file_list}
      "${${output_file_list}}"
      PARENT_SCOPE)
endfunction()

set(COMPILED_SHADERS "")
add_shader("shaders/specialization" COMPILED_SHADERS)

# Create a custom target to track dependencies, allowing test recompilation on
# shader changes.
list(TRANSFORM COMPILED_SHADERS PREPEND "${CMAKE_CURRENT_BINARY_DIR}/"
                                        OUTPUT_VARIABLE SHADER_DEPENDENCIES)
add_custom_target(shaders DEPENDS ${SHADER_DEPENDENCIES})

# We have 3 lists that we substitute into the test_shaders cpp/header files.
# First, we generate a set of #include statements, then we define and
# extern-declare Span's that cover the shader data
list(TRANSFORM COMPILED_SHADERS REPLACE "(.+)" "#include \"\\1\""
                                        OUTPUT_VARIABLE SHADER_INCLUDES)
list(TRANSFORM COMPILED_SHADERS
     REPLACE "(.+).inc.h" "loon::gpu::Span<const unsigned char> \\1(\\1_bytes)"
             OUTPUT_VARIABLE SHADER_SPAN_DEFINES)
list(TRANSFORM COMPILED_SHADERS
     REPLACE "(.+).inc.h" "extern loon::gpu::Span<const unsigned char> \\1"
             OUTPUT_VARIABLE SHADER_EXTERN_DECLARATIONS)
string(JOIN "\n" SHADER_INCLUDES ${SHADER_INCLUDES})
string(JOIN ";\n" SHADER_EXTERN_DECLARATIONS ${SHADER_EXTERN_DECLARATIONS})
string(JOIN ";\n" SHADER_SPAN_DEFINES ${SHADER_SPAN_DEFINES})
configure_file("shaders/test_shaders.in.cpp"
               ${CMAKE_CURRENT_BINARY_DIR}/test_shaders.cpp @ONLY)
configure_file("shaders/test_shaders.in.h"
               ${CMAKE_CURRENT_BINARY_DIR}/test_shaders.h @ONLY)

add_executable(
  loon_gpu_tests
  main.cpp utilities_tests.cpp example_tests.cpp pipeline_tests.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/test_shaders.cpp)

add_dependencies(loon_gpu_tests shaders)

target_link_libraries(loon_gpu_tests PRIVATE loon_gpu example_utilities)

target_include_directories(
  loon_gpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src
                         ${CMAKE_CURRENT_BINARY_DIR})

if(LOON_GPU_WERROR)
  if(MSVC)
    target_compile_options(loon_gpu_tests PRIVATE /WX)
  else()
    target_compile_options(loon_gpu_tests PRIVATE -Werror)
  endif()
endif()

add_test(NAME loon_gpu COMMAND loon_gpu_tests)

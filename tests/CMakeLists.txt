# Compile shaders using slang to spirv
function(add_shader shader output_file_list)
    set(SLANGC_ARGS 
        -target spirv 
        -profile spirv_1_6 
        -fvk-use-entrypoint-name 
        -emit-spirv-directly 
        -fvk-use-scalar-layout 
        -source-embed-style u8
    )
    cmake_path(GET shader FILENAME filename)
    set(output_file ${filename}.inc)
    cmake_path (SET input_file NORMALIZE ${CMAKE_CURRENT_SOURCE_DIR}/${shader}.slang)   
    add_custom_command(
        OUTPUT "${output_file}.h"
        COMMAND ${SLANGC_COMMAND} ${SLANGC_ARGS} ${input_file} -source-embed-name ${filename} -o ${output_file}
        COMMAND_EXPAND_LISTS
        DEPENDS ${shader}.slang
        DEPENDS_EXPLICIT_ONLY
    )
    list(APPEND ${output_file_list} "${output_file}.h")
    set(${output_file_list} "${${output_file_list}}" PARENT_SCOPE)
    message("After appending ${${output_file_list}}")
endfunction()

set(COMPILED_SHADERS "")
add_shader("shaders/specialization" COMPILED_SHADERS)

message("Compiled ${COMPILED_SHADERS}")
# Create a custom target to 
list(TRANSFORM COMPILED_SHADERS PREPEND "${CMAKE_CURRENT_BINARY_DIR}/" OUTPUT_VARIABLE SHADER_DEPENDENCIES)
add_custom_target(shaders DEPENDS ${SHADER_DEPENDENCIES})

list(TRANSFORM COMPILED_SHADERS REPLACE "(.+)" "#include \"\\1\"" OUTPUT_VARIABLE SHADER_INCLUDES)
string(JOIN "\n" SHADER_INCLUDES ${SHADER_INCLUDES})

list(TRANSFORM COMPILED_SHADERS REPLACE "(.+).inc.h" "extern const unsigned char \\1[];extern const size_t \\1_sizeInBytes" OUTPUT_VARIABLE SHADER_EXTERN_DECLARATIONS)
string(JOIN ";\n" SHADER_EXTERN_DECLARATIONS ${SHADER_EXTERN_DECLARATIONS})

configure_file("shaders/test_shaders.in.cpp" ${CMAKE_CURRENT_BINARY_DIR}/test_shaders.cpp @ONLY)
configure_file("shaders/test_shaders.in.h" ${CMAKE_CURRENT_BINARY_DIR}/test_shaders.h @ONLY)

add_executable(
    loon_gpu_tests
    main.cpp
    utilities_tests.cpp
    example_tests.cpp
    pipeline_tests.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/test_shaders.cpp
)

add_dependencies(loon_gpu_tests shaders)

target_link_libraries(loon_gpu_tests PRIVATE loon_gpu example_utilities)

target_include_directories(loon_gpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_BINARY_DIR})

if (LOON_GPU_WERROR)
    if (MSVC)
        target_compile_options(loon_gpu_tests PRIVATE /WX)
    else()
        target_compile_options(loon_gpu_tests PRIVATE -Werror)
    endif()
endif()

add_test(NAME loon_gpu COMMAND loon_gpu_tests)

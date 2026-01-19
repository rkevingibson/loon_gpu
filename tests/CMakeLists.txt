
add_executable(
    loon_gpu_tests
    main.cpp
    utilities_tests.cpp
    example_tests.cpp
)

target_link_libraries(loon_gpu_tests PRIVATE GTest::gtest loon_gpu volk VulkanMemoryAllocator example_utilities)

target_include_directories(loon_gpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

if (LOON_GPU_WERROR)
    if (MSVC)
        target_compile_options(loon_gpu_tests PRIVATE /WX)
    else()
        target_compile_options(loon_gpu_tests PRIVATE -Werror)
    endif()
endif()

include(GoogleTest)
gtest_discover_tests(loon_gpu_tests)

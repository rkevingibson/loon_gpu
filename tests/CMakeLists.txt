
add_executable(
    loon_webgpu_tests
    main.cpp
    utilities_tests.cpp
)

target_link_libraries(loon_webgpu_tests PRIVATE GTest::gtest loon_webgpu volk VulkanMemoryAllocator)

target_include_directories(loon_webgpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

if (LOON_WEBGPU_WERROR)
    if (MSVC)
        target_compile_options(loon_webgpu_tests PRIVATE /WX)
    else()
        target_compile_options(loon_webgpu_tests PRIVATE -Werror)
    endif()
endif()

include(GoogleTest)
gtest_discover_tests(loon_webgpu_tests)

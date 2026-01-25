include(FetchContent)

### Volk

FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG vulkan-sdk-1.4.335
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

### MoltenVK

if(APPLE)

FetchContent_Declare(
    moltenvk
    URL https://github.com/KhronosGroup/MoltenVK/releases/download/v1.4.0/MoltenVK-macos.tar
    URL_HASH SHA256=f4feaf6a4988352de8e6d49874ccc1cd6a45021e3cb476e8531bef7ecc73e93a
)
FetchContent_MakeAvailable(moltenvk)

add_library(moltenvk SHARED IMPORTED GLOBAL)
target_include_directories(moltenvk INTERFACE ${moltenvk_SOURCE_DIR}/MoltenVK/include )
set_target_properties(
        moltenvk
        PROPERTIES
            IMPORTED_IMPLIB ${moltenvk_SOURCE_DIR}/MoltenVK/dylib/macOS/libMoltenVK.dylib
)
target_link_directories(moltenvk 
    INTERFACE ${moltenvk_SOURCE_DIR}/MoltenVK/dylib/macOS)
    
endif()


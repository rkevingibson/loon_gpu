#include "example.h"

#include <gpu/loon_gpu.h>

#include "filesystem.h"

#define STB_IMAGE_IMPLEMENTATION

#if __APPLE__
#    include <mach-o/dyld.h>
#elif _WIN32
extern "C" __declspec(dllimport) unsigned long __stdcall GetModuleFileNameA(void*,
                                                                            char*,
                                                                            unsigned long);
#else
#    include <unistd.h>
#endif

#include <vector>

#include "bunny/bunny.h"
#include "common/box.h"
#include "hello_cube/hello_cube.h"
#include "hello_triangle/hello_triangle.h"
#include "many_cubes/many_cubes.h"
#include "particle_emitter/particle_emitter.h"
#include "stb_image.h"
#include "string_view.h"
#include "textured_cube/textured_cube.h"


static std::string get_executable_directory() {
#if __APPLE__
    uint32_t bufsize = 0;
    _NSGetExecutablePath(nullptr, &bufsize);
    std::vector<char> path(bufsize);
    _NSGetExecutablePath(path.data(), &bufsize);
#elif _WIN32
    std::vector<char> path(256);
    uint32_t          bufsize = GetModuleFileNameA(nullptr, path.data(), (uint32_t)path.size()) + 1;
#else
    uint32_t          bufsize = 256;
    std::vector<char> path(bufsize, '\0');
    ssize_t           result = 0;
    do {
        result = readlink("/proc/self/exe", path.data(), bufsize);
        if (result > 0 && static_cast<uint32_t>(result) == bufsize) {
            // Truncation possible
            bufsize *= 2;
            path.resize(bufsize, '\0');
        } else if (result > 0) {
            bufsize = static_cast<uint32_t>(result) + 1;
        }
    } while (result == bufsize || result == -1);

#endif
    auto parent = loon::filesystem::parent_path(loon::StringView(path.data(), bufsize - 1));
    return std::string(parent.begin(), parent.end());
}

FilePaths default_file_paths() {
    auto exec_dir = get_executable_directory();

    char        arena_buf[2048];
    loon::Arena arena(arena_buf, 2048);
    auto        default_shader_dir =
        loon::filesystem::normalize_path(&arena, (exec_dir + "/shaders/").c_str());
    auto default_asset_dir =
        loon::filesystem::normalize_path(&arena, (exec_dir + "/assets/").c_str());

    return {
        .shader_directory = std::string(default_shader_dir.data(), default_shader_dir.size()),
        .asset_directory  = std::string(default_asset_dir.data(), default_asset_dir.size()),
    };
}

void log_callback(loon::gpu::LogLevel         lvl,
                  loon::gpu::Span<const char> message,
                  uint32_t                    line_number,
                  loon::gpu::Span<const char> filename,
                  void*                       userdata) {
    fprintf(stderr, "%.*s\n", static_cast<int>(message.size()), message.data());
}

loon::Box<Example> create_example(ExampleName name, const WindowState& state) {
    switch (name) {
        case ExampleName::HelloTriangle: return loon::make_box<HelloTriangle>(state);
        case ExampleName::HelloCube: return loon::make_box<HelloCube>(state);
        case ExampleName::TexturedCube: return loon::make_box<TexturedCube>(state);
        case ExampleName::ParticleEmitter: return loon::make_box<ParticleEmitter>(state);
        case ExampleName::ManyCubes: return loon::make_box<ManyCubes>(state);
        case ExampleName::Bunny: return loon::make_box<Bunny>(state);
        default: return nullptr;
    }
}

using namespace loon;

static Format select_surface_format(const loon::gpu::SurfaceCapabilities& surface_capabilities) {
    for (Format f : surface_capabilities.formats) {
        if (f == loon::gpu::Format::RGBA8UnormSrgb || f == loon::gpu::Format::BGRA8UnormSrgb) {
            // Choose 8 bit srgb if we have it
            return f;
        }
    }
    return surface_capabilities.formats[0];
}

Example::Example(const WindowState& window) {
    m_device = loon::gpu::create_device({
        .gpu_preference         = loon::gpu::GpuPreference::Discrete,
        .native_window_handle   = window.native_window_handle,
        .native_instance_handle = window.native_instance_handle,
        .log_callback           = log_callback,
        .log_userdata           = nullptr,
        .log_level              = LogLevel::Debug,
        .alloc_callback         = nullptr,
        .alloc_userdata         = nullptr,
    });

    auto surface_capabilities = gpu::get_surface_capabilities(m_device);
    m_swapchain_format        = select_surface_format(surface_capabilities);

    recreate_swapchain(window.width, window.height);

    m_queue = gpu::get_queue(m_device);
}

Example::~Example() {
    gpu::destroy_device(m_device);
}

void Example::tick(const WindowState& window) {
    auto surface_texture = gpu::get_current_texture(m_device);
    if (surface_texture.status == SurfaceStatus::OutOfDate ||
        surface_texture.status == SurfaceStatus::Suboptimal) {
        recreate_swapchain(window.width, window.height);
        return;
    } else if (surface_texture.status == SurfaceStatus::Error) {
        return;
    }

    this->update({
        .color_format  = m_swapchain_format,
        .depth_format  = m_depth_format,
        .color_texture = surface_texture.texture,
        .depth_texture = m_depth_texture,
        .texture_size =
            {
                .x = m_swapchain_width,
                .y = m_swapchain_height,
            },
    });

    const auto status = gpu::present(m_device, m_queue);
    if (status == SurfaceStatus::OutOfDate || status == SurfaceStatus::Suboptimal) {
        recreate_swapchain(window.width, window.height);
    }
}

void Example::recreate_swapchain(uint32_t width, uint32_t height) {
    gpu::unconfigure_surface(m_device);
    gpu::configure_surface(m_device,
                           {
                               .format       = m_swapchain_format,
                               .usages       = loon::gpu::UsageFlags::ColorAttachment,
                               .width        = width,
                               .height       = height,
                               .present_mode = PresentMode::Fifo,
                           });
    m_swapchain_width  = width;
    m_swapchain_height = height;

    // Recreate depth buffer as well
    if (m_depth_texture) { gpu::free(m_device, m_depth_texture); }
    m_depth_texture =
        gpu::create_texture(m_device,
                            {
                                .type       = TextureType::Tex2D,
                                .dimensions = {.x = width, .y = height, .z = 1},
                                .format     = m_depth_format,
                                .usage      = loon::gpu::UsageFlags::DepthStencilAttachment,
                            });
}
#pragma once

#include <gpu/loon_gpu.h>

#include <string>

#include "common/box.h"


class ShaderLoader;

struct FilePaths {
    std::string shader_directory;
    std::string asset_directory;
};

struct WindowState {
    uintptr_t               native_window_handle;
    uintptr_t               native_instance_handle;
    uint16_t                width;
    uint16_t                height;
    loon::Box<ShaderLoader> shader_loader;
    FilePaths               file_paths;
};

class Example {
   public:
    Example(const WindowState& window_state);
    virtual ~Example();
    void tick(const WindowState& window);

   protected:
    struct UpdateInfo {
        loon::gpu::Format                     color_format;
        loon::gpu::Format                     depth_format;
        loon::gpu::Handle<loon::gpu::Texture> color_texture;
        loon::gpu::Handle<loon::gpu::Texture> depth_texture;
        loon::gpu::Dimension2D                texture_size;
    };

    virtual bool update(const UpdateInfo& info) = 0;

    loon::gpu::Device m_device;
    loon::gpu::Queue  m_queue;
    loon::gpu::Format m_swapchain_format;

   private:
    void recreate_swapchain(uint32_t width, uint32_t height);

    loon::gpu::Format                     m_depth_format = loon::gpu::Format::Depth32Float;
    uint32_t                              m_swapchain_width;
    uint32_t                              m_swapchain_height;
    loon::gpu::Handle<loon::gpu::Texture> m_depth_texture;
};

enum class ExampleName {
    HelloTriangle,
    HelloCube,
    TexturedCube,
    ParticleEmitter,
    ManyCubes,

    Count,
};

FilePaths default_file_paths();

void log_callback(loon::gpu::LogLevel         lvl,
                  loon::gpu::Span<const char> message,
                  uint32_t                    line_number,
                  loon::gpu::Span<const char> filename,
                  void*                       userdata);

loon::Box<Example> create_example(ExampleName name, const WindowState& state);
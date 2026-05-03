#include "gpu/loon_gpu.h"
#include "test_shaders.h"
#include "utest.h"

using namespace loon;
using loon::gpu::operator""_sv;

UTEST(pipeline_tests, basic_compute_compilation) {
    auto device = gpu::create_device({});
    if (!device) { UTEST_SKIP("Unable to create device, likely running in CI without GPU."); }

    auto pipeline = gpu::create_compute_pipeline(device,
                                                 {
                                                     .source      = test::shaders::specialization,
                                                     .entry_point = "basic_compute"_sv,
                                                 });

    ASSERT_TRUE(pipeline);

    gpu::destroy_device(device);
}

struct ClearBufferArgs {
    gpu::GpuPtr buffer;
    uint32_t    num_elements;
};

UTEST(pipeline_tests, specialized_compute_compilation) {
    auto device = gpu::create_device({});
    if (!device) { UTEST_SKIP("Unable to create device, likely running in CI without GPU."); }

    const uint32_t kClearValue = 42;

    auto pipeline =
        gpu::create_compute_pipeline(device,
                                     {
                                         .source      = test::shaders::specialization,
                                         .entry_point = "clear_buffer"_sv,
                                     },
                                     gpu::SpecializationConstant{
                                         .constant_id = 0,
                                         .int_val     = 42,
                                         .type        = gpu::SpecializationConstantType::UInt32,
                                     });
    ASSERT_TRUE(pipeline);

    auto                    args        = gpu::malloc(device, sizeof(ClearBufferArgs));
    static constexpr size_t kBufferSize = 1024;
    auto buffer      = gpu::malloc(device, kBufferSize * sizeof(uint32_t), gpu::Memory::Readback);
    auto cpu_args    = reinterpret_cast<ClearBufferArgs*>(gpu::get_host_pointer(device, args));
    cpu_args->buffer = buffer;
    cpu_args->num_elements = kBufferSize;

    auto queue = gpu::get_queue(device);

    auto cmd = gpu::queue_start_command_recording(queue);

    gpu::cmd_set_pipeline(cmd, pipeline);
    gpu::cmd_dispatch(cmd, args, {.x = kBufferSize / 64, .y = 1, .z = 1});
    gpu::cmd_finalize(cmd);

    gpu::queue_submit(queue, cmd);

    gpu::device_wait_for_idle(device);

    auto cpu_buffer = reinterpret_cast<uint32_t*>(gpu::get_host_pointer(device, buffer));

    for (uint32_t i = 0; i < kBufferSize; ++i) { ASSERT_EQ(cpu_buffer[i], kClearValue); }

    gpu::destroy_device(device);
}

UTEST(api_tests, stencil_buffer) {
    auto device = gpu::create_device({});
    if (!device) { UTEST_SKIP("Unable to create device, likely running in CI without GPU."); }

    auto color_target = gpu::create_texture(
        device,
        gpu::TextureDesc{
            .dimensions =
                {
                    .x = 32,
                    .y = 32,
                    .z = 1,
                },
            .format = gpu::Format::RGBA8UnormSrgb,
            .usage  = gpu::UsageFlags::ColorAttachment | gpu::UsageFlags::TransferSrc,
        });
    auto stencil_texture = gpu::create_texture(device,
                                               gpu::TextureDesc{
                                                   .dimensions = {32, 32, 1},
                                                   .format     = gpu::Format::Stencil8,
                                                   .usage = gpu::UsageFlags::DepthStencilAttachment,
                                               });

    auto stencil_pipeline =
        gpu::create_graphics_pipeline(device,
                                      gpu::ShaderSource{
                                          .source      = test::shaders::specialization,
                                          .entry_point = "fullscreen_quad_vert"_sv,
                                      },
                                      gpu::ShaderSource{
                                          .source      = test::shaders::specialization,
                                          .entry_point = "fullscreen_quad_frag"_sv,
                                      },
                                      gpu::RasterDesc{
                                          .stencil_format = gpu::Format::Stencil8,
                                          .color_targets =
                                              gpu::ColorTarget{
                                                  .format     = gpu::Format::RGBA8UnormSrgb,
                                                  .blendstate = gpu::BlendDesc{},
                                              },
                                      });

    auto depth_stencil_state =
        gpu::create_depth_stencil_state(device,
                                        gpu::DepthStencilDesc{
                                            .depth_mode = gpu::DepthFlags::None,
                                            .stencil_front =
                                                gpu::Stencil{
                                                    .test          = gpu::Op::Always,
                                                    .fail_op       = gpu::StencilOp::Keep,
                                                    .pass_op       = gpu::StencilOp::IncrementClamp,
                                                    .depth_fail_op = gpu::StencilOp::Keep,
                                                    .reference     = 0,
                                                },
                                        });

    auto queue = gpu::get_queue(device);

    auto cmd = gpu::queue_start_command_recording(queue);
    gpu::cmd_begin_render_pass(cmd,
                               gpu::RenderPassDesc{
                                   .color_attachments =
                                       gpu::RenderAttachment{
                                           .texture  = color_target,
                                           .load_op  = loon::gpu::LoadOp::Clear,
                                           .store_op = loon::gpu::StoreOp::Store,
                                           .clear_color =
                                               gpu::Color{
                                                   0,
                                                   0,
                                                   0,
                                                   0,
                                               },
                                       },
                                   .depth_attachment = {},
                                   .stencil_attachment =
                                       {
                                           .texture     = stencil_texture,
                                           .load_op     = loon::gpu::LoadOp::Clear,
                                           .store_op    = loon::gpu::StoreOp::Discard,
                                           .clear_color = gpu::Color{0, 0, 0, 0},
                                       },
                                   .render_area{
                                       .offset_x = 0,
                                       .offset_y = 0,
                                       .width    = 32,
                                       .height   = 32,
                                   },
                               });

    gpu::cmd_set_pipeline(cmd, stencil_pipeline);
    gpu::cmd_set_depth_stencil_state(cmd, depth_stencil_state);
    gpu::cmd_draw(cmd, 0, 0, 3, 1);
    gpu::cmd_end_render_pass(cmd);
    gpu::cmd_finalize(cmd);
    gpu::destroy_device(device);
}
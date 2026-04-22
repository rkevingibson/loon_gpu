#include "test_shaders.h"
#include "utest.h"

using namespace loon;
using loon::gpu::operator""_sv;

UTEST(pipeline_tests, basic_compute_compilation) {
    auto device = gpu::create_device({});
    ASSERT_NE(device, nullptr);

    auto pipeline = gpu::create_compute_pipeline(device,
                                                 {
                                                     .spirv       = test::shaders::specialization,
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
    ASSERT_NE(device, nullptr);

    const uint32_t kClearValue = 42;

    auto pipeline
        = gpu::create_compute_pipeline(device,
                                       {
                                           .spirv       = test::shaders::specialization,
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
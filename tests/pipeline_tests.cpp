#include "test_shaders.h"
#include "utest.h"

using namespace loon;
using loon::gpu::operator""_sv;

UTEST(pipeline_tests, basic_compute_compilation) {
    auto device = gpu::create_device({});

    auto pipeline = gpu::create_compute_pipeline(
        device,
        {
            .spirv
            = gpu::Span(test::shaders::specialization, test::shaders::specialization_sizeInBytes),
            .entry_point = "basic_compute"_sv,
        });

    ASSERT_TRUE(pipeline);


    gpu::destroy_device(device);
}
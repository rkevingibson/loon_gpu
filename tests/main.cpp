#include <gtest/gtest.h>
#include "webgpu/webgpu_loon.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(wgpu_tests, function_table) {
    WGPULoonFunctionTable gpu{};
    wgpuLoonGetFunctionTable(&gpu);

    WGPUInstanceDescriptor instance_descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    WGPUInstance           instance            = gpu.create_instance(&instance_descriptor);

    // Note: curently, we only support creating a single instance
    // WGPUInstance instance2 = gpu.create_instance(&instance_descriptor);
    // gpu.instance_release(instance2);

    // Need an HWND to create a surface...
    // gpu.instance_create_surface(instance, )

    WGPURequestAdapterOptions adapter_options{
        .nextInChain          = nullptr,
        .featureLevel         = WGPUFeatureLevel_Core,
        .powerPreference      = WGPUPowerPreference_HighPerformance,
        .forceFallbackAdapter = false,
        .backendType          = WGPUBackendType_Undefined,
        .compatibleSurface    = nullptr,
    };


    WGPUAdapter adapter = nullptr;
    auto        future  = gpu.instance_request_adapter(
        instance,
        &adapter_options,
        {
                    .nextInChain = nullptr,
                    .mode        = WGPUCallbackMode_AllowSpontaneous,
                    .callback    = [](WGPURequestAdapterStatus,
                           WGPUAdapter adapter,
                           WGPUStringView,
                           void* userdata1,
                           void*) { *reinterpret_cast<WGPUAdapter*>(userdata1) = adapter; },
                    .userdata1   = &adapter,
                    .userdata2   = nullptr,
        });

    WGPUFutureWaitInfo wait_info{
        .future    = future,
        .completed = false,
    };
    ASSERT_EQ(gpu.instance_wait_any(instance, 1, &wait_info, UINT64_MAX), WGPUWaitStatus_Success);
    ASSERT_EQ(gpu.instance_wait_any(instance, 1, &wait_info, UINT64_MAX), WGPUWaitStatus_Success);
    ASSERT_TRUE(wait_info.completed);
    ASSERT_NE(adapter, nullptr);

    WGPUAdapterInfo adapter_info{};
    gpu.adapter_get_info(adapter, &adapter_info);
    ASSERT_NE(adapter_info.subgroupMaxSize, 0);
    ASSERT_NE(adapter_info.subgroupMinSize, 0);

    WGPULimits limits{};
    gpu.adapter_get_limits(adapter, &limits);
    // TODO: Check these limits against the spec, make sure they meet the minimums.


    WGPUDeviceDescriptor device_descriptor{
        .nextInChain                 = nullptr,
        .label                       = WGPU_STRING_VIEW_INIT,
        .requiredFeatureCount        = 0,
        .requiredFeatures            = nullptr,
        .requiredLimits              = nullptr,
        .defaultQueue                = WGPU_QUEUE_DESCRIPTOR_INIT,
        .deviceLostCallbackInfo      = WGPU_DEVICE_LOST_CALLBACK_INFO_INIT,
        .uncapturedErrorCallbackInfo = WGPU_UNCAPTURED_ERROR_CALLBACK_INFO_INIT,
    };

    WGPUDevice device = nullptr;
    gpu.adapter_request_device(
        adapter,
        &device_descriptor,
        {
            .nextInChain = nullptr,
            .mode        = WGPUCallbackMode_AllowProcessEvents,
            .callback    = [](WGPURequestDeviceStatus,
                           WGPUDevice device,
                           WGPUStringView,
                           void* userdata1,
                           void*) { *reinterpret_cast<WGPUDevice*>(userdata1) = device; },
            .userdata1   = &device,
            .userdata2   = nullptr,
        });

    gpu.instance_process_events(instance);
    ASSERT_NE(device, nullptr);

    gpu.device_destroy(device);
    gpu.adapter_release(adapter);
    gpu.instance_release(instance);
}

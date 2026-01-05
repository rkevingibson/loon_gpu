#include "initialization.h"

#include <cassert>

#include "webgpu/webgpu_loon.h"

static void WGPUErrorCallback(WGPUDevice const*,
                              WGPUErrorType  type,
                              WGPUStringView message,
                              WGPU_NULLABLE void*,
                              WGPU_NULLABLE void*) {
    switch (type) {
        case WGPUErrorType_Validation:
            fprintf(stderr,
                    "Validation Error: %.*s",
                    static_cast<int>(message.length),
                    message.data);
            break;
        case WGPUErrorType_Internal:
            fprintf(stderr,
                    "Internal webgpu Error: %.*s",
                    static_cast<int>(message.length),
                    message.data);
            break;
        case WGPUErrorType_OutOfMemory:
            fprintf(stderr,
                    "Webgpu out of memory error: %.*s",
                    static_cast<int>(message.length),
                    message.data);
            break;
        default: break;
    }
}

static void LoonLogCallback(WGPULoonLogLevel lvl, WGPUStringView message, void* userdata) {
    fprintf(stderr, "%.*s", static_cast<int>(message.length), message.data);
}

WGPUInstance create_instance() {
    WGPULoonInstanceConfiguration loon_config {
        .chain = { .next = nullptr, .sType = static_cast<WGPUSType>(WGPUSType_LoonInstanceConfiguration),
           },
        .alloc = nullptr,
        .alloc_userdata = nullptr,
        .log_level = WGPULoonLogLevel_Debug,
        .log = LoonLogCallback,
        .log_userdata = nullptr,
    };

    WGPUInstanceDescriptor instance_descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    instance_descriptor.nextInChain            = &loon_config.chain;
    WGPUInstance instance                      = wgpuCreateInstance(&instance_descriptor);
    return instance;
}

WGPUAdapter get_default_adapter(WGPUInstance instance) {
    WGPURequestAdapterOptions adapter_options{
        .nextInChain          = nullptr,
        .featureLevel         = WGPUFeatureLevel_Core,
        .powerPreference      = WGPUPowerPreference_HighPerformance,
        .forceFallbackAdapter = false,
        .backendType          = WGPUBackendType_Undefined,
        .compatibleSurface    = nullptr,
    };

    WGPUAdapter adapter = nullptr;
    auto        future  = wgpuInstanceRequestAdapter(
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
                    .userdata1   = (void*)&adapter,
                    .userdata2   = nullptr,
        });

    WGPUFutureWaitInfo wait_info{
        .future    = future,
        .completed = false,
    };
    const WGPUWaitStatus wait_status = wgpuInstanceWaitAny(instance, 1, &wait_info, UINT64_MAX);
    assert(wait_status == WGPUWaitStatus_Success);
    assert(wait_info.completed);

    return adapter;
}

WGPUSurface create_surface(WGPUInstance instance, const WindowState& window_state) {
#if _WIN32

    WGPUSurfaceSourceWindowsHWND surface_source = WGPU_SURFACE_SOURCE_WINDOWS_HWND_INIT;
    surface_source.hwnd      = reinterpret_cast<void*>(window_state.native_window_handle);
    surface_source.hinstance = reinterpret_cast<void*>(window_state.native_instance_handle);
#elif __linux__
#    error "Unsupported platform currently"
#elif __APPLE__
    WGPUSurfaceSourceMetalLayer surface_source = WGPU_SURFACE_SOURCE_METAL_LAYER_INIT;
    surface_source.layer = reinterpret_cast<void*>(window_state.native_window_handle);
#endif

    WGPUSurfaceDescriptor surface_descriptor{
        .nextInChain = &surface_source.chain,
        .label       = "Hello Triangle Surface"_wsv,
    };

    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surface_descriptor);
    return surface;
}

WGPUDevice get_default_device(WGPUInstance instance, WGPUAdapter adapter) {
    WGPUDevice device = nullptr;
    WGPUDeviceDescriptor device_descriptor{
        .nextInChain                 = nullptr,
        .label                       = WGPU_STRING_VIEW_INIT,
        .requiredFeatureCount        = 0,
        .requiredFeatures            = nullptr,
        .requiredLimits              = nullptr,
        .defaultQueue                = WGPU_QUEUE_DESCRIPTOR_INIT,
        .deviceLostCallbackInfo      = WGPU_DEVICE_LOST_CALLBACK_INFO_INIT,
        .uncapturedErrorCallbackInfo = {
            .nextInChain = nullptr,
            .callback = WGPUErrorCallback,
            .userdata1 = nullptr,
            .userdata2 = nullptr,
        },
    };

    const auto device_future = wgpuAdapterRequestDevice(
        adapter,
        &device_descriptor,
        {
            .nextInChain = nullptr,
            .mode        = WGPUCallbackMode_AllowSpontaneous,
            .callback    = [](WGPURequestDeviceStatus,
                           WGPUDevice device,
                           WGPUStringView,
                           void* userdata1,
                           void*) { *reinterpret_cast<WGPUDevice*>(userdata1) = device; },
            .userdata1   = (void*)&device,
            .userdata2   = nullptr,
        });

    WGPUFutureWaitInfo wait_info{
        .future    = device_future,
        .completed = false,
    };

    const WGPUWaitStatus wait_status = wgpuInstanceWaitAny(instance, 1, &wait_info, UINT64_MAX);
    assert(wait_status == WGPUWaitStatus_Success);
    assert(wait_info.completed);
    return device;
}

WGPUTextureFormat select_surface_format(const WGPUSurfaceCapabilities& surface_capabilities) {
    for (size_t format_idx = 0; format_idx < surface_capabilities.formatCount; ++format_idx) {
        if (surface_capabilities.formats[format_idx] == WGPUTextureFormat_BGRA8UnormSrgb) {
            // Choose 8 bit srgb if we have it
            return surface_capabilities.formats[format_idx];
        }
    }
    return surface_capabilities.formats[0];
}
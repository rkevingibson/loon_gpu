#pragma
#include "example.h"
#include "webgpu/webgpu.h"

WGPUInstance      create_instance();
WGPUAdapter       get_default_adapter(WGPUInstance instance);
WGPUSurface       create_surface(WGPUInstance instance, const WindowState& window_state);
WGPUDevice        get_default_device(WGPUInstance instance, WGPUAdapter adapter);
WGPUTextureFormat select_surface_format(const WGPUSurfaceCapabilities& surface_capabilities);
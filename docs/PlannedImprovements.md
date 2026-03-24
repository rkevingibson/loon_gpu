# Planned Improvements

## API additions
- Timer queries
- Mesh shader support
- Hardware ray tracing support
- 

## Vulkan Backend

- Support for VK_EXT_descriptor_heap should simplify shader writing and reduce some CPU overhead of creating texture views/samplers
- Support for VK_KHR_device_address_commands. This should eliminate most of the need for lookups from GPUPtrs to Buffer + offset pairs internally, reducing CPU overhead for draw calls.
- Support for VK_KHR_maintenance5, specifically avoiding explicit shader module creation and passing them to VkPipelineShaderStageCreateInfo.
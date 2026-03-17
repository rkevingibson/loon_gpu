# Shader Arguments

Loon GPU prioritizes a bindless approach, which simplifies GPU resource binding. Here we detail our bindless approach which is backend-specific.

## Vulkan

In vulkan, we use push constants to pass gpu buffer addresses to shaders. This relies on the [Buffer Device Address feature](https://docs.vulkan.org/guide/latest/buffer_device_address.html). Basically, shaders should expect a pointer to a struct containing all the arguments. 

For compute pipeline, we have a single push constant at offset 0 of size 8, containing a single pointer. In Slang, this can be expressed by taking a single `uniform Args* args` argument in your compute entry point. 

For graphics pipelines, we have two push constants - one at offset 0 for the vertex args, and one at offset 8 for the fragment args. 
Note that [due to an limitation in slang](https://github.com/shader-slang/slang/issues/9643), the uniform pointer approach used for compute shaders doesn't quite work for vertex/fragment shaders. As a result, the current behavior makes the entire push constant range visible to vertex and fragment stages. See the examples for a workaround, but I'm hopeful this will get fixed in the future and the simpler uniform parameter syntax will work.


For textures, shaders can access them through their `TextureView` handles. 


## 
# Writing shaders

While examples use [Slang](https://shader-slang.org/), Loon is agnostic to the specific shader language used. However, Loon is opinionated about how these shaders should look - particularly how resource binding is done. We use a bindless approach taking advantage of pointers to gpu memory, which simplifies GPU resource binding. At a high level, the idea is that every draw and dispatch gets passed a `GPUPtr` containing arguments. For draw calls we pass 2 pointers - one for vertex, and one for fragment shaders. These pointers should be to a struct in GPU memory containing any GPU arguments you like - including other pointers, textures, or other draw data. 

For textures, we use a lightweight textureview object, which is an index into a larget list of textures. The list itself is represented by the `TextureHeap` object, and most applications could get by with having a single texture heap that is set at the start of a command buffer and never changed. 

For samplers, currently we only support a fixed set of immutable samplers, created at device initialization time. Ideally we could define all our samplers in shader code, like Metal supports with constexpr samplers, but that isn't currently possible. I'm still iterating on this so we may end up with a separate sampler heap similar to what is in DX12 and the new VK_EXT_Descriptor_heap extension.

## Vulkan

In vulkan, we use push constants to pass gpu buffer addresses to shaders. 

For compute pipeline, we have a single push constant at offset 0 of size 8, containing a single pointer. In Slang, this can be expressed by taking a single `uniform Args* args` argument in your compute entry point. 

For graphics pipelines, we have two push constants - one at offset 0 for the vertex args, and one at offset 8 for the fragment args. Currently, both push constants are made visible to both stages.
Note that [due to an limitation in slang](https://github.com/shader-slang/slang/issues/9643), the uniform pointer approach used for compute shaders doesn't quite work for vertex/fragment shaders. See the examples for a workaround, but I'm hopeful this will get fixed in the future and the simpler uniform parameter syntax will work, at which point we'll likely switch the logic to make the push constants only visible to the appropriate stages.

For textures, shaders can access them through their `TextureView` handles. Textures are placed in an array of Sampled images at set 0, binding slot 2. Samplers are in set 0 at slot 0, and both are visible to all stages. Set `create_descriptor_layout()` in vk/loon_gpu.cpp for details. The textureView that is obtained by a call to `add_texture_view_to_heap()` is the textures index into that array.

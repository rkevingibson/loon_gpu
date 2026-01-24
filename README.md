# Loon GPU

A bindless wrapper for Vulkan, targetting modern GPUs.

The API design is inspired by Sebastian Aaltonen's blog post [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api), adapted to the realities of what is possible in Vulkan today. Currently it targets a roughly Vulkan 1.3 feature set, with a couple of required features:
- Dynamic Rendering
- Buffer Device Addresses
- Timeline Semaphores
- Descriptor Indexing

This should support most modern desktop GPUs, and I've also been able to run it on my M5 Macbook Pro via [KosmicKrisp](https://www.lunarg.com/a-vulkan-on-metal-mesa-3d-graphics-driver/) with no issues.


The project is designed to be built with cmake, and should be self-contained (all dependencies are either vendored or downloaded by cmake via FetchContent). 
Fast compile times are a goal, and so minimal use of C++ STL headers are used whenever possible - internally we use custom containers and the API uses a custom Span replacement.
Note that this doesn't apply to the examples, which use things like std::filesystem and std::vector for simplicity.


#include <metal_stdlib>

[[kernel, required_threads_per_threadgroup(64,1,1)]]
void basic_compute()
{
    
}


constant uint kClearValue [[function_constant(0)]];

struct ClearBufferArgs {
    device uint* buffer;
    uint num_elements;
};

[[kernel, required_threads_per_threadgroup(64,1,1)]]
void clear_buffer(device ClearBufferArgs *args [[buffer(0)]], uint gid [[thread_position_in_grid]])
{
    if (gid < args->num_elements) {
        args->buffer[gid] = kClearValue;
    }
}
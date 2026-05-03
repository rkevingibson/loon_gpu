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


[[vertex]]
float4 fullscreen_quad_vert(uint vertexIdx [[vertex_id]])
{
   const float2 verts[3] =  {float2(-1,-1), float2(3,-1), float2(-1,3)};

   return float4(verts[vertexIdx], 0.5, 1.0);
}

[[fragment]]
float4 fullscreen_quad_frag()
{
    return float4(1.0, 0.0, 0.0, 1.0);
}
[shader("compute")]
[numthreads(64,1,1)]
void basic_compute()
{
    
}

[vk::constant_id(0)]
const uint kClearValue = 0; // Maps to a SPIR-V specialization constant

struct ClearBufferArgs {
    uint* buffer;
    uint num_elements;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffer(uniform ClearBufferArgs *args, uint gid: SV_DispatchThreadID)
{
    if (gid < args->num_elements) {
        args->buffer[gid] = kClearValue;
    }
}

[shader("vertex")]
float4 fullscreen_quad_vert(uint32_t vertexIdx : SV_VertexID)
{
   const float2 verts[3] =  {float2(-1,-1), float2(3,-1), float2(-1,3)};

   return float4(verts[vertexIdx], 0.5, 1.0);
}

[shader("fragment")]
float4 fullscreen_quad_frag() : SV_Target
{
    return float4(1.0, 0.0, 0.0, 1.0);
}
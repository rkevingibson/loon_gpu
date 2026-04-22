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
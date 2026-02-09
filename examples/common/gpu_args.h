#pragma once

#include "gpu/loon_gpu.h"

namespace loon {

class RingBuffer {
   public:
    RingBuffer(gpu::Device* device);


    template <class T>
    gpu::GpuPtr append(T&& t);

   private:
    gpu::Handle<gpu::Buffer> m_buffer;
};
}  // namespace loon
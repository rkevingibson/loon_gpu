#pragma once

#include <gpu/loon_gpu.h>

#include "common/example.h"
using namespace loon::gpu;

class HelloTriangle : public Example {
   public:
    HelloTriangle(const WindowState& window_state);
    ~HelloTriangle() override = default;

   protected:
    bool Update(const UpdateInfo& window) override;

   private:
    Handle<Pipeline> m_render_pipeline;
};

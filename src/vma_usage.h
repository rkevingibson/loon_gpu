#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#include "volk.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include "vk_mem_alloc.h"

#pragma clang diagnostic pop
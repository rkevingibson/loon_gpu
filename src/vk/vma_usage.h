#pragma once
#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wnullability-completeness"
#    pragma clang diagnostic ignored "-Wunused-variable"
#    pragma clang diagnostic ignored "-Wunused-private-field"
#    pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include "volk.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include "vk_mem_alloc.h"

#ifdef __clang__
#    pragma clang diagnostic pop
#endif
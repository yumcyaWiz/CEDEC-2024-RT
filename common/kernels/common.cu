#pragma once
#include "common/core.hpp"

KERNEL void clear(TypedBuffer<float4> buffer, int width, int height)
{
    // compute thread id
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    // compute pixel id
    const int xi = tid % width;
    const int yi = tid / width;
    const int pixel_idx = xi + (height - yi - 1) * width;

    // clear buffer
    buffer[pixel_idx] = {0.0f, 0.0f, 0.0f, 0.0f};
}

INLINE DEVICE float aces_tone_mapping(float x)
{
    // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

KERNEL void tone_mapping(TypedBuffer<uint8_t> pixels,
                         TypedBuffer<float4> accumulation, int width,
                         int height)
{
    // compute thread id
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    // compute pixel id
    const int xi = tid % width;
    const int yi = tid / width;
    const int pixel_idx = xi + (height - yi - 1) * width;

    // read radiance and sample count
    const float4 radiance_and_spp = accumulation[pixel_idx];

    // average
    float3 radiance = {radiance_and_spp.x / radiance_and_spp.w,
                       radiance_and_spp.y / radiance_and_spp.w,
                       radiance_and_spp.z / radiance_and_spp.w};

    // multiply exposure
    constexpr float exposure = 1.0f;
    radiance *= exposure;

    // tone mapping
    radiance.x = aces_tone_mapping(radiance.x);
    radiance.y = aces_tone_mapping(radiance.y);
    radiance.z = aces_tone_mapping(radiance.z);

    // gamma correction
    constexpr float gamma = 1.0f / 2.2f;
    radiance.x = powf(radiance.x, gamma);
    radiance.y = powf(radiance.y, gamma);
    radiance.z = powf(radiance.z, gamma);

    // write to pixel buffer
    pixels[4 * pixel_idx + 0] =
        (uint8_t)(clamp(radiance.x * 255.0f, 0.0f, 255.0f));
    pixels[4 * pixel_idx + 1] =
        (uint8_t)(clamp(radiance.y * 255.0f, 0.0f, 255.0f));
    pixels[4 * pixel_idx + 2] =
        (uint8_t)(clamp(radiance.z * 255.0f, 0.0f, 255.0f));
    pixels[4 * pixel_idx + 3] = 255;
}

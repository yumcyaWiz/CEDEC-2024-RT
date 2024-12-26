#pragma once
#include "common/core.hpp"
#include "common/raytrace.hpp"

struct ReservoirSample
{
    float3 origin_position = {0.0f, 0.0f, 0.0f};  // origin position
    float3 origin_normal = {0.0f, 0.0f, 0.0f};    // origin normal
    float3 hit_position = {0.0f, 0.0f, 0.0f};     // hit position
    float3 hit_normal = {0.0f, 0.0f, 0.0f};       // hit normal
    float3 radiance = {0.0f, 0.0f, 0.0f};         // radiance at hit position
    bool visibility = false;                      // visibility
};

struct Reservoir
{
    ReservoirSample sample;  // sample
    float w_sum = 0.0f;      // weight sum
    float ucw = 0.0f;        // unbiased contribution weight
    int M = 0;               // sample count

    INLINE DEVICE void update(const ReservoirSample& next_sample, float weight,
                              float u)
    {
        // resampling
        w_sum += weight;
        M += 1;
        if (u < weight / w_sum) { sample = next_sample; }
    }

    INLINE DEVICE void merge(const Reservoir& reservoir, float weight, float u)
    {
        // resampling
        w_sum += weight;
        M += reservoir.M;
        if (u < weight / w_sum) { sample = reservoir.sample; }
    }
};

#if defined(__CUDACC__) || defined(__HIPCC__)

INLINE DEVICE float evaluate_target_function(
    const float3& origin_position, const float3& origin_normal,
    const float3& hit_position, const float3& hit_normal,
    const float3& radiance, const hiprtGeometry& hiprt_geometry,
    const TypedBuffer<Triangle>& triangles, const bool is_shadowed)
{
    const float brdf = 1.0f / PI;
    const float G =
        geometry_term(origin_position, origin_normal, hit_position, hit_normal);

    if (is_shadowed)
    {
        const float V = check_visibility(origin_position, origin_normal,
                                         hit_position, hiprt_geometry);
        return brdf * G * V * luminance(radiance);
    }
    else { return brdf * G * luminance(radiance); }
}

INLINE DEVICE float normal_rejection_heuristics(const float3& n0,
                                                const float3& n1)
{
    return powf(max(dot(n0, n1), 0.0f), 8.0f);
}

INLINE DEVICE float depth_rejection_heuristics(const float3& p0,
                                               const float3& p1,
                                               const float3& eye)
{
    const float depth0 = length(p0 - eye);
    const float depth1 = length(p1 - eye);
    const float diff = (depth1 - depth0) * (depth1 - depth0) / depth0;
    return expf(-32.0f * diff);
}

INLINE DEVICE float rejection_heuristics(const Reservoir& r0,
                                         const Reservoir& r1, const float3& eye)
{
    float w = 1.0f;
    w *= depth_rejection_heuristics(r0.sample.origin_position,
                                    r1.sample.origin_position, eye);
    w *= normal_rejection_heuristics(r0.sample.origin_normal,
                                     r1.sample.origin_normal);

    return w;
}

INLINE DEVICE float2 sample_2d_gaussian(const float rv0, const float rv1)
{
    // 2D gaussian sampling using Box-Muller's method
    const float radius = sqrt(max(-2.0f * log(rv0), 0.0f));
    const float phi = 2.0f * PI * rv1;
    return radius * float2{cos(phi), sin(phi)};
}

#endif
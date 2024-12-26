#pragma once
#include "common/core.hpp"

struct Options
{
    bool accumulate = false;
    int max_depth = 6;
    float3 sky_color = {0.0f, 0.0f, 0.0f};

    int ris_sample_count = 32;

    float rejection_heuristics_threshold = 0.2f;

    bool use_temporal_resampling = false;

    bool use_spatial_resampling = false;
    int spatial_resampling_sample_count = 5;
    float spatial_resampling_radius = 30.0f;
    int spatial_resampling_passes = 3;

    bool use_shadowed_target_function = false;
    bool use_visibility_reuse = true;
};
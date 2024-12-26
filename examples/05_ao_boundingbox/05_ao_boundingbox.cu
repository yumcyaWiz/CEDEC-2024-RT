#include "common/camera.hpp"
#include "common/core.hpp"
#include "common/math.hpp"
#include "common/rng.hpp"
#include "common/typedbuffer.hpp"

#define AO_BOUNDING_BOX_BLOCK_SIZE 128

// 1: AABB culling before triangle intersection, 0: intersect with all triangles
#define ENABLE_AABB_CULLING 1

// 1: do an extra AABB culling before every 32 AABBs
#define ENABLE_AABB_WARP_LEVEL_CULLING 0

template <class T>
INLINE DEVICE T warpMin(T val)
{
    for (int i = 1; i < 32; i *= 2) { val = min(val, __shfl_xor(val, i)); }
    return val;
}
template <class T>
INLINE DEVICE T warpMax(T val)
{
    for (int i = 1; i < 32; i *= 2) { val = max(val, __shfl_xor(val, i)); }
    return val;
}

struct AABB
{
    float3 lower;
    float3 upper;
};

INLINE DEVICE float3 max(float3 a, float3 b)
{
    return {
        max(a.x, b.x),
        max(a.y, b.y),
        max(a.z, b.z),
    };
}
INLINE DEVICE float3 min(float3 a, float3 b)
{
    return {
        min(a.x, b.x),
        min(a.y, b.y),
        min(a.z, b.z),
    };
}
INLINE DEVICE float3 clamp(float3 x, float3 a, float3 b)
{
    return min(max(x, a), b);
}

INLINE DEVICE float compMin(float3 v) { return min(min(v.x, v.y), v.z); }
INLINE DEVICE float compMax(float3 v) { return max(max(v.x, v.y), v.z); }
INLINE DEVICE float2 slabs(float3 p0, float3 p1, float3 ro, float3 one_over_rd)
{
    float3 t0 = (p0 - ro) * one_over_rd;
    float3 t1 = (p1 - ro) * one_over_rd;

    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float region_min = compMax(tmin);
    float region_max = compMin(tmax);

    region_min = max(region_min, 0.0f);

    return {region_min, region_max};
}

INLINE DEVICE float3 safe_inv_rd(float3 rd)
{
    return clamp(float3{1.0f, 1.0f, 1.0f} / rd,
                 float3{-FLT_MAX, -FLT_MAX, -FLT_MAX},
                 float3{FLT_MAX, FLT_MAX, FLT_MAX});
}

// TODO (keto): remove duplication in core.hpp
INLINE DEVICE bool closeset_hit(Intersection* intersection, float3 ro,
                                float3 rd,
                                const TypedBuffer<Triangle>& triangles)
{
#if ENABLE_AABB_CULLING
    float3 one_over_rd = safe_inv_rd(rd);

    __shared__ AABB s_AABBs[AO_BOUNDING_BOX_BLOCK_SIZE];

    float t = FLT_MAX;
    int index = -1;

    for (int i = 0;
         i < next_multiple(triangles.size(), AO_BOUNDING_BOX_BLOCK_SIZE);
         i += AO_BOUNDING_BOX_BLOCK_SIZE)
    {
        {
            int triIndex = i + threadIdx.x;
            float3 lower = {FLT_MAX, FLT_MAX, FLT_MAX};
            float3 upper = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
            if (triIndex < triangles.size())
            {
                Triangle tri = triangles[triIndex];
                lower =
                    min(min(tri.vertices[0], tri.vertices[1]), tri.vertices[2]);
                upper =
                    max(max(tri.vertices[0], tri.vertices[1]), tri.vertices[2]);
            }

            s_AABBs[threadIdx.x] = {lower, upper};
        }

        __syncthreads();

#if ENABLE_AABB_WARP_LEVEL_CULLING
        for (int j = 0; j < AO_BOUNDING_BOX_BLOCK_SIZE; j += 32)
        {
            int lane = threadIdx.x % 32;
            float3 lower = s_AABBs[j + lane].lower;
            float3 upper = s_AABBs[j + lane].upper;
            lower.x = warpMin(lower.x);
            lower.y = warpMin(lower.y);
            lower.z = warpMin(lower.z);
            upper.x = warpMax(upper.x);
            upper.y = warpMax(upper.y);
            upper.z = warpMax(upper.z);
            float2 rangeWarp = slabs(lower, upper, ro, one_over_rd);
            if (rangeWarp.y < rangeWarp.x) { continue; }
            for (int k = 0; k < 32; k++)
            {
                float2 range = slabs(s_AABBs[j + k].lower, s_AABBs[j + k].upper,
                                     ro, one_over_rd);
                int triIndex = i + j + k;
                if (range.x <= range.y && triIndex < triangles.size())
                {
                    Triangle tri = triangles[triIndex];
                    float u, v;
                    if (intersect_ray_triangle(&t, &u, &v, ro, rd, 0.0f, t,
                                               tri.vertices[0], tri.vertices[1],
                                               tri.vertices[2]))
                    {
                        index = triIndex;
                    }
                }
            }
        }
#else
        for (int j = 0; j < AO_BOUNDING_BOX_BLOCK_SIZE; j++)
        {
            int triIndex = i + j;
            float2 range =
                slabs(s_AABBs[j].lower, s_AABBs[j].upper, ro, one_over_rd);

            if (range.x <= range.y && triIndex < triangles.size())
            {
                Triangle tri = triangles[triIndex];
                float u, v;
                if (intersect_ray_triangle(&t, &u, &v, ro, rd, 0.0f, t,
                                           tri.vertices[0], tri.vertices[1],
                                           tri.vertices[2]))
                {
                    index = triIndex;
                }
            }
        }
#endif

        __syncthreads();
    }

#else
    float t = 3.402823466e+38f;
    int index = -1;
    for (int i = 0; i < triangles.size(); i++)
    {
        Triangle tri = triangles[i];

        float u, v;
        if (intersect_ray_triangle(&t, &u, &v, ro, rd, 0.0f, t, tri.vertices[0],
                                   tri.vertices[1], tri.vertices[2]))
        {
            index = i;
        }
    }
#endif

    if (index < 0) { return false; }
    intersection->t = t;
    intersection->index = index;
    return true;
}

KERNEL void kernelMain(TypedBuffer<uint8_t> pixels, RayGenerator rayGen,
                       int width, int height,
                       const TypedBuffer<Triangle> triangles)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    int xi = tid % width;
    int yi = tid / width;
    int pixelIdx = xi + (height - yi - 1) * width;

    PCG random(0, hashPCG3(xi, yi, 42));

    float3 ro, rd;
    rayGen.shoot(&ro, &rd, (float)xi / width, (float)yi / height);

    Intersection isect;

    bool primary_hit = false;
    float3 n = {1.0f, 0.0f, 0.0f};
    float3 tangent0 = {0.0f, 1.0f, 0.0f};
    float3 tangent1 = {0.0f, 0.0f, 1.0f};
    float3 p_hit = {100.0f, 0.0f, 0.0f};

    __shared__ uint32_t s_nPrimaryHit;

    if (threadIdx.x == 0) { s_nPrimaryHit = 0; }
    __syncthreads();

    if (closeset_hit(&isect, ro, rd, triangles))
    {
        primary_hit = true;

        n = normal_of(triangles[isect.index]);

        if (0.0f < dot(n, rd)) { n = -n; }

        tangent0 = a_tangent_of(triangles[isect.index]);
        tangent1 = cross(tangent0, n);

        p_hit = ro + rd * isect.t;

        atomicInc(&s_nPrimaryHit, 0xFFFFFFFF);
    }

    __syncthreads();

    float3 ao_ro = p_hit + n * 0.0001f;

    const int N_Rays = 64;
    int n_visible = 0;
    if (s_nPrimaryHit)
        for (int i = 0; i < N_Rays; i++)
        {
            float3 s = sample_hemisphere(random.uniformf(), random.uniformf(),
                                         random.uniformf());
            float3 ao_rd = tangent0 * s.x + tangent1 * s.z + n * s.y;

            Intersection ao_isect;
            if (closeset_hit(&ao_isect, ao_ro, ao_rd, triangles) == false)
            {
                n_visible++;
            }
        }
    float ao = (float)n_visible / N_Rays;

    if (tid < width * height)
    {
        if (primary_hit)
        {
            pixels[pixelIdx * 4 + 0] =
                (uint8_t)(powf(ao, 1.0f / 2.2f) * 255.0f);
            pixels[pixelIdx * 4 + 1] =
                (uint8_t)(powf(ao, 1.0f / 2.2f) * 255.0f);
            pixels[pixelIdx * 4 + 2] =
                (uint8_t)(powf(ao, 1.0f / 2.2f) * 255.0f);
            pixels[pixelIdx * 4 + 3] = 255;
        }
        else
        {
            pixels[pixelIdx * 4 + 0] = 32;
            pixels[pixelIdx * 4 + 1] = 32;
            pixels[pixelIdx * 4 + 2] = 32;
            pixels[pixelIdx * 4 + 3] = 255;
        }
    }
}

#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>
#else
#include <hiprt/hiprt.h>
#endif

#include "common/core.hpp"
#include "common/types.hpp"

#define SHARED_STACK_SIZE 32
#define BLOCK_SIZE 256

#if defined(__CUDACC__) || defined(__HIPCC__)

INLINE DEVICE bool raytrace(const Ray& r, const hiprtGeometry& geom,
                            Intersection& isect)
{
    // Stack memory for BVH traversal. For simplicity, no use of global
    // memory for stack. more details:
    // https://gpuopen.com/learn/hiprt_2_1_batch_construction_transformation_functions/
    __shared__ int s_mem[SHARED_STACK_SIZE * BLOCK_SIZE];
    hiprtGlobalStack stack(
        {}, hiprtSharedStackBuffer{SHARED_STACK_SIZE, (void*)s_mem});

    hiprtRay ray;
    ray.origin = hiprtFloat3{r.origin.x, r.origin.y, r.origin.z};
    ray.direction = hiprtFloat3{r.direction.x, r.direction.y, r.direction.z};
    ray.minT = r.tmin;
    ray.maxT = r.tmax;

    hiprtGeomTraversalClosestCustomStack<hiprtGlobalStack> tr(geom, ray, stack);
    hiprtHit hit = tr.getNextHit();
    if (hit.hasHit() == false) { return false; }

    isect.t = hit.t;
    isect.uv = hit.uv;
    isect.index = hit.primID;

    return true;
}

INLINE DEVICE float check_visibility(const float3& p0, const float3& n0,
                                     const float3& p1,
                                     const hiprtGeometry& geom)
{
    const Ray ray = make_ray(offset_ray_position(p0, n0), p1 - p0, 0.0f, 0.99f);
    Intersection isect;
    return raytrace(ray, geom, isect) ? 0.0f : 1.0f;
}

#endif
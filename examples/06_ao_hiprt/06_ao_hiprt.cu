#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

#include "common/camera.hpp"
#include "common/core.hpp"
#include "common/math.hpp"
#include "common/rng.hpp"
#include "common/typedbuffer.hpp"

#define SHARED_STACK_SIZE 32
#define BLOCK_SIZE 256

INLINE DEVICE bool hiprt_closeset_hit(Intersection* intersection, float3 ro,
                                      float3 rd, hiprtGeometry geom)
{
    // Stack memory for BVH traverssal. For simplicity, no use of global memory
    // for stack. more details:
    // https://gpuopen.com/learn/hiprt_2_1_batch_construction_transformation_functions/
    __shared__ int s_mem[SHARED_STACK_SIZE * BLOCK_SIZE];
    hiprtGlobalStack stack(
        {}, hiprtSharedStackBuffer{SHARED_STACK_SIZE, (void*)s_mem});

    hiprtRay ray;
    ray.origin = hiprtFloat3{ro.x, ro.y, ro.z};
    ray.direction = hiprtFloat3{rd.x, rd.y, rd.z};

    hiprtGeomTraversalClosestCustomStack<hiprtGlobalStack> tr(geom, ray, stack);
    hiprtHit hit = tr.getNextHit();
    if (hit.hasHit() == false) { return false; }
    intersection->t = hit.t;
    intersection->index = hit.primID;
    return true;
}

KERNEL void kernelMain(TypedBuffer<uint8_t> pixels, RayGenerator rayGen,
                       int width, int height, hiprtGeometry geom,
                       TypedBuffer<Triangle> triangles)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    int xi = tid % width;
    int yi = tid / width;
    int pixelIdx = xi + (height - yi - 1) * width;

    PCG random(0, hashPCG3(xi, yi, 42));

    float3 ro, rd;
    rayGen.shoot(&ro, &rd, (float)xi / width, (float)yi / height);

    Intersection isect;
    if (hiprt_closeset_hit(&isect, ro, rd, geom) == false)
    {
        pixels[pixelIdx * 4 + 0] = 32;
        pixels[pixelIdx * 4 + 1] = 32;
        pixels[pixelIdx * 4 + 2] = 32;
        pixels[pixelIdx * 4 + 3] = 255;
        return;
    }

    float3 n = normal_of(triangles[isect.index]);

    if (0.0f < dot(n, rd)) { n = -n; }

    float3 tangent0 = a_tangent_of(triangles[isect.index]);
    float3 tangent1 = cross(tangent0, n);

    float3 p_hit = ro + rd * isect.t;
    float3 ao_ro = p_hit + n * 0.0001f;

    const int N_Rays = 64;
    int n_visible = 0;
    for (int i = 0; i < N_Rays; i++)
    {
        float3 s = sample_hemisphere(random.uniformf(), random.uniformf(),
                                     random.uniformf());
        float3 ao_rd = tangent0 * s.x + tangent1 * s.z + n * s.y;

        Intersection ao_isect;
        if (hiprt_closeset_hit(&ao_isect, ao_ro, ao_rd, geom) == false)
        {
            n_visible++;
        }
    }
    float ao = (float)n_visible / N_Rays;

    pixels[pixelIdx * 4 + 0] = (uint8_t)(powf(ao, 1.0f / 2.2f) * 255.0f);
    pixels[pixelIdx * 4 + 1] = (uint8_t)(powf(ao, 1.0f / 2.2f) * 255.0f);
    pixels[pixelIdx * 4 + 2] = (uint8_t)(powf(ao, 1.0f / 2.2f) * 255.0f);
    pixels[pixelIdx * 4 + 3] = 255;
}

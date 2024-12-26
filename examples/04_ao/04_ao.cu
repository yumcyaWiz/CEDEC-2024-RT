#include "common/camera.hpp"
#include "common/core.hpp"
#include "common/math.hpp"
#include "common/rng.hpp"
#include "common/typedbuffer.hpp"

// TODO (keto): remove duplication in core.hpp
INLINE DEVICE bool closeset_hit(Intersection* intersection, float3 ro,
                                float3 rd,
                                const TypedBuffer<Triangle>& triangles)
{
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
    if (width * height <= tid) { return; }

    int xi = tid % width;
    int yi = tid / width;
    int pixelIdx = xi + (height - yi - 1) * width;

    PCG random(0, hashPCG3(xi, yi, 42));

    float3 ro, rd;
    rayGen.shoot(&ro, &rd, (float)xi / width, (float)yi / height);

    Intersection isect;
    if (closeset_hit(&isect, ro, rd, triangles))
    {
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
            if (closeset_hit(&ao_isect, ao_ro, ao_rd, triangles) == false)
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
    else
    {
        pixels[pixelIdx * 4 + 0] = 32;
        pixels[pixelIdx * 4 + 1] = 32;
        pixels[pixelIdx * 4 + 2] = 32;
        pixels[pixelIdx * 4 + 3] = 255;
    }
}

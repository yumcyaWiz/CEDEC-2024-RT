#include "common/camera.hpp"
#include "common/core.hpp"
#include "common/math.hpp"
#include "common/typedbuffer.hpp"

KERNEL void kernelMain(TypedBuffer<uint8_t> pixels, RayGenerator rayGen,
                       int width, int height,
                       const TypedBuffer<Triangle> triangles)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    int xi = tid % width;
    int yi = tid / width;

    float3 ro, rd;
    rayGen.shoot(&ro, &rd, (float)xi / width, (float)yi / height);

    float t = 3.402823466e+38f;
    int triIdx = -1;
    for (int i = 0; i < triangles.size(); i++)
    {
        Triangle tri = triangles[i];

        float u, v;
        if (intersect_ray_triangle(&t, &u, &v, ro, rd, 0.0f, t, tri.vertices[0],
                                   tri.vertices[1], tri.vertices[2]))
        {
            triIdx = i;
        }
    }

    int pixelIdx = xi + (height - yi - 1) * width;

    if (0 <= triIdx)
    {
        float3 color = triangles[triIdx].color;

        pixels[pixelIdx * 4 + 0] = (uint8_t)(color.x * 255.0f);
        pixels[pixelIdx * 4 + 1] = (uint8_t)(color.y * 255.0f);
        pixels[pixelIdx * 4 + 2] = (uint8_t)(color.z * 255.0f);
        pixels[pixelIdx * 4 + 3] = 255;
    }
    else
    {
        pixels[pixelIdx * 4 + 0] = 32;
        pixels[pixelIdx * 4 + 1] = 32;
        pixels[pixelIdx * 4 + 2] = 32;
        pixels[pixelIdx * 4 + 3] = 255;
    }

    // float t, u, v;

    // if(intersect_ray_triangle(&t, &u, &v, ro, rd, v0, v1, v2))
    //{
    //	float U = 1.0f - u - v;
    //	float V = u;
    //	float W = v;

    //	pixels[pixelIdx * 4 + 0] = U * 255.0f;
    //	pixels[pixelIdx * 4 + 1] = V * 255.0f;
    //	pixels[pixelIdx * 4 + 2] = W * 255.0f;
    //	pixels[pixelIdx * 4 + 3] = 255;
    //}
    // else
    //{
    //	pixels[pixelIdx * 4 + 0] = 32; // (sinf(xi * 0.01f + time * 3.0f ) *
    // 0.5f + 0.5f) * 255; 	pixels[pixelIdx * 4 + 1] = 32; // (sinf(yi * 0.02f +
    // time * 3.0f) * 0.5f + 0.5f) * 255; 	pixels[pixelIdx * 4 + 2] = 32;
    //	pixels[pixelIdx * 4 + 3] = 255;
    //}
}

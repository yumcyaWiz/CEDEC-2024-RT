#include "common/camera.hpp"
#include "common/core.hpp"
#include "common/math.hpp"
#include "common/typedbuffer.hpp"

INLINE DEVICE bool mollar(float* tOut, float* uOut, float* vOut, float3 ro,
                          float3 rd, float3 v0, float3 v1, float3 v2)
{
    const float kEpsilon = 1.0e-8f;

    float3 v0v1 = v1 - v0;
    float3 v0v2 = v2 - v0;
    float3 pvec = cross(rd, v0v2);
    float det = dot(v0v1, pvec);

    if (fabs(det) < kEpsilon) { return false; }

    float invDet = 1.0f / det;

    float3 tvec = ro - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) { return false; }

    float3 qvec = cross(tvec, v0v1);
    float v = dot(rd, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) { return false; }

    float t = dot(v0v2, qvec) * invDet;

    if (t < 0.0f) { return false; }

    *tOut = t;
    *uOut = u;
    *vOut = v;
    return true;
}

INLINE DEVICE bool intersect_ray_triangle(float* tOut, float* uOut, float* vOut,
                                          float3 ro, float3 rd, float3 v0,
                                          float3 v1, float3 v2)
{
    float3 e0 = v1 - v0;
    float3 e1 = v2 - v1;
    float3 e2 = v0 - v2;

    float3 n = cross(e0, e1);
    float t = dot(v0 - ro, n) / dot(n, rd);
    if (0.0f <= t && t < 3.402823466e+38f)
    {
        float3 p = ro + rd * t;

        float n2TriArea0 =
            dot(n, cross(e0, p - v0));  // |n| * 2 * tri_area( p, v0, v1 )
        float n2TriArea1 =
            dot(n, cross(e1, p - v1));  // |n| * 2 * tri_area( p, v1, v2 )
        float n2TriArea2 =
            dot(n, cross(e2, p - v2));  // |n| * 2 * tri_area( p, v2, v0 )
        if (n2TriArea0 < 0.0f || n2TriArea1 < 0.0f || n2TriArea2 < 0.0f)
        {
            return false;
        }
        float n2TriArea = n2TriArea0 + n2TriArea1 +
                          n2TriArea2;  // |n| * 2 * tri_area( v0, v1, v2 )

        // Barycentric Coordinates
        float bW = n2TriArea0 /
                   n2TriArea;  // tri_area( p, v0, v1 ) / tri_area( v0, v1, v2 )
        float bU = n2TriArea1 /
                   n2TriArea;  // tri_area( p, v1, v2 ) / tri_area( v0, v1, v2 )
        float bV = n2TriArea2 /
                   n2TriArea;  // tri_area( p, v2, v0 ) / tri_area( v0, v1, v2 )

        *tOut = t;
        *uOut = bV;
        *vOut = bW;
        return true;
    }

    return false;
}

KERNEL void kernelMain(TypedBuffer<uint8_t> pixels, RayGenerator rayGen,
                       int width, int height, float3 v0, float3 v1, float3 v2)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    int xi = tid % width;
    int yi = tid / width;

    float3 ro, rd;
    rayGen.shoot(&ro, &rd, (float)xi / width, (float)yi / height);

    int pixelIdx = xi + (height - yi - 1) * width;
    float t, u, v;

    if (intersect_ray_triangle(&t, &u, &v, ro, rd, v0, v1, v2))
    {
        float U = 1.0f - u - v;
        float V = u;
        float W = v;

        pixels[pixelIdx * 4 + 0] = U * 255.0f;
        pixels[pixelIdx * 4 + 1] = V * 255.0f;
        pixels[pixelIdx * 4 + 2] = W * 255.0f;
        pixels[pixelIdx * 4 + 3] = 255;
    }
    else
    {
        pixels[pixelIdx * 4 + 0] =
            32;  // (sinf(xi * 0.01f + time * 3.0f ) * 0.5f + 0.5f) * 255;
        pixels[pixelIdx * 4 + 1] =
            32;  // (sinf(yi * 0.02f + time * 3.0f) * 0.5f + 0.5f) * 255;
        pixels[pixelIdx * 4 + 2] = 32;
        pixels[pixelIdx * 4 + 3] = 255;
    }
}

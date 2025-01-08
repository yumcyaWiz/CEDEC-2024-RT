#pragma once

#include "common/math.hpp"
#include "common/typedbuffer.hpp"
#include "common/types.hpp"

struct Ray
{
    float3 origin = {0.0f, 0.0f, 0.0f};
    float3 direction = {0.0f, 0.0f, 0.0f};
    float tmin = 0.0f;
    float tmax = FLT_MAX;

    INLINE DEVICE float3 operator()(float t) const
    {
        return origin + t * direction;
    }
};

INLINE DEVICE Ray make_ray(const float3& ro, const float3& rd,
                           float tmin = 0.0f, float tmax = FLT_MAX)
{
    Ray ret;
    ret.origin = ro;
    ret.direction = rd;
    ret.tmin = tmin;
    ret.tmax = tmax;

    return ret;
}

INLINE DEVICE float3 offset_ray_position(const float3& p, const float3& n)
{
    constexpr float RAY_EPS = 0.001f;
    return p + RAY_EPS * n;
}

struct Triangle
{
    float3 vertices[3];
    float3 color;
    float3 emissive;
};

INLINE DEVICE float3 a_tangent_of(const Triangle& triangle)
{
    return normalize(triangle.vertices[1] - triangle.vertices[0]);
}

INLINE DEVICE float3 normal_of(const Triangle& triangle)
{
    float3 e0 = triangle.vertices[1] - triangle.vertices[0];
    float3 e1 = triangle.vertices[2] - triangle.vertices[0];
    return normalize(cross(e0, e1));
}

INLINE DEVICE float area_of(const Triangle& triangle)
{
    const float3 e0 = triangle.vertices[1] - triangle.vertices[0];
    const float3 e1 = triangle.vertices[2] - triangle.vertices[0];
    return 0.5f * length(cross(e0, e1));
}

INLINE DEVICE bool has_emission(const Triangle& triangle)
{
    return triangle.emissive.x > 0.0f || triangle.emissive.y > 0.0f ||
           triangle.emissive.z > 0.0f;
}

template <class T>
INLINE DEVICE T max_of(T x, T y)
{
    return (x < y) ? y : x;
}

INLINE DEVICE float3 sample_hemisphere(float r0, float r1, float r2)
{
    float theta = r0 * 2.0f * PI;
    float radius = r1 + r2;
    if (1.0f < radius) { radius = 2.0f - radius; }

    // uniform in a circle
    float x = cos(theta) * radius;
    float z = sin(theta) * radius;

    // cos-weighted
    float y = sqrtf(max_of(1.0f - radius * radius, 0.0f));
    return {x, y, z};
}

INLINE DEVICE bool intersect_ray_triangle(float* tOut, float* uOut, float* vOut,
                                          const float3& ro, const float3& rd,
                                          float tmin, float tmax,
                                          const float3& v0, const float3& v1,
                                          const float3& v2)
{
    const float3 e0 = v1 - v0;
    const float3 e1 = v2 - v1;
    const float3 e2 = v0 - v2;

    const float3 n = cross(e0, e1);
    const float t = dot(v0 - ro, n) / dot(n, rd);

    if (tmin <= t && t <= tmax)  // This condition is not met also when t is nan
    {
        const float3 p = ro + rd * t;
        const float n2TriArea0 =
            dot(n, cross(e0, p - v0));  // |n| * 2 * tri_area( p, v0, v1 )
        const float n2TriArea1 =
            dot(n, cross(e1, p - v1));  // |n| * 2 * tri_area( p, v1, v2 )
        const float n2TriArea2 =
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

struct Intersection
{
    float t = 0.0f;
    float2 uv = {0.0f, 0.0f};
    int index = -1;
};

struct SurfaceInfo
{
    float3 p = {0.0f, 0.0f, 0.0f};       // hit position
    float3 n = {0.0f, 1.0f, 0.0f};       // hit normal
    float2 barycentrics = {0.0f, 0.0f};  // barycentrics
};

INLINE DEVICE SurfaceInfo
make_surface_info(const Ray& ray, const Intersection& isect,
                  const TypedBuffer<Triangle>& triangles)
{
    SurfaceInfo ret;
    ret.p = ray(isect.t);
    ret.n = normal_of(triangles[isect.index]);
    ret.barycentrics = isect.uv;

    // flip normal
    if (dot(-ray.direction, ret.n) < 0.0f) { ret.n = -ret.n; }

    return ret;
}

struct Visibility
{
    float2 uv = {0.0f, 0.0f};  // barycentrics
    int index = -1;            // triangle index
    int _pad;                  // padding
};

INLINE DEVICE SurfaceInfo make_surface_info(
    const Visibility& visibility, const TypedBuffer<Triangle>& triangles)
{
    const Triangle& triangle = triangles[visibility.index];

    SurfaceInfo ret;
    ret.p = (1.0f - visibility.uv.x - visibility.uv.y) * triangle.vertices[0] +
            visibility.uv.x * triangle.vertices[1] +
            visibility.uv.y * triangle.vertices[2];
    ret.n = normal_of(triangle);
    ret.barycentrics = visibility.uv;

    return ret;
}

INLINE DEVICE SurfaceInfo
make_surface_info(const Visibility& visibility,
                  const TypedBuffer<Triangle>& triangles, const float3& eye)
{
    const Triangle& triangle = triangles[visibility.index];

    SurfaceInfo ret;
    ret.p = (1.0f - visibility.uv.x - visibility.uv.y) * triangle.vertices[0] +
            visibility.uv.x * triangle.vertices[1] +
            visibility.uv.y * triangle.vertices[2];
    ret.n = normal_of(triangle);
    ret.barycentrics = visibility.uv;

    // flip normal
    const float3 view = normalize(eye - ret.p);
    if (dot(view, ret.n) < 0.0f) { ret.n = -ret.n; }

    return ret;
}

struct TangentBasis
{
    float3 t = {1.0f, 0.0f, 0.0f};  // tangent vector
    float3 n = {0.0f, 1.0f, 0.0f};  // normal vector
    float3 b = {0.0f, 0.0f, 1.0f};  // bitangent vector
};

INLINE DEVICE float3 world_to_local(const float3& v, const TangentBasis& basis)
{
    return {dot(v, basis.t), dot(v, basis.n), dot(v, basis.b)};
}

INLINE DEVICE float3 local_to_world(const float3& v, const TangentBasis& basis)
{
    return v.x * basis.t + v.y * basis.n + v.z * basis.b;
}

INLINE DEVICE TangentBasis make_tangent_basis(
    const float3& n, int index, const TypedBuffer<Triangle>& triangles)
{
    TangentBasis ret;
    ret.t = a_tangent_of(triangles[index]);
    ret.n = n;
    ret.b = normalize(cross(ret.t, n));

    return ret;
}

INLINE DEVICE float2 warp_unit_triangle(float x, float y)
{
    // Heitz, Eric. "A Low-Distortion Map Between Triangle and Square." (2019).
    // https://hal.science/hal-02073696v2/document
    if (y > x)
    {
        x *= 0.5f;
        y -= x;
    }
    else
    {
        y *= 0.5f;
        x -= y;
    }
    return {x, y};
}

struct LightSample
{
    float3 p = {0.0f, 0.0f, 0.0f};  // position
    float3 n = {0.0f, 0.0f, 0.0f};  // normal
    int index = -1;                 // triangle index
};

INLINE DEVICE LightSample sample_light(const TypedBuffer<Triangle>& triangles,
                                       const TypedBuffer<uint32_t>& lights,
                                       float rv0, float rv1, float rv2)
{
    // uniform light sampling
    uint32_t nth_light = rv0 * lights.size();
    if (nth_light == lights.size()) { nth_light = lights.size() - 1; }

    LightSample ret;
    ret.index = lights[nth_light];

    const Triangle& light_triangle = triangles[ret.index];

    // uniform sampling on the triangle
    const float2 barycentrics = warp_unit_triangle(rv1, rv2);
    ret.p =
        (1.0f - barycentrics.x - barycentrics.y) * light_triangle.vertices[0] +
        barycentrics.x * light_triangle.vertices[1] +
        barycentrics.y * light_triangle.vertices[2];

    // use geometry normal
    ret.n = normal_of(light_triangle);

    return ret;
}

INLINE DEVICE float geometry_term(const float3& p0, const float3& n0,
                                  const float3& p1, const float3& n1)
{
    float3 v = p1 - p0;
    const float sqr_dist = dot(v, v);
    v = normalize(v);

    return abs(dot(v, n0)) * abs(dot(-v, n1)) / sqr_dist;
}
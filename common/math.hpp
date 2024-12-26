#pragma once

#include "common/types.hpp"

#if !defined(__KERNELCC__)
#include <math.h>
struct float2
{
    float x, y;
};

struct float3
{
    float x, y, z;
};

struct float4
{
    float x, y, z, w;
};
#endif

constexpr float PI = 3.14159265358979323846f;

#define FLT_MAX 3.402823466e+38f

#if !defined(__HIPCC__) && !defined(NO_VECTOR_OP_OVERLOAD)
INLINE DEVICE float3 operator/(const float3& a, const float3& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}
INLINE DEVICE float3 operator/(const float3& a, float s)
{
    return {a.x / s, a.y / s, a.z / s};
}

INLINE DEVICE float3 operator/=(const float3& a, float s)
{
    return {a.x / s, a.y / s, a.z / s};
}

INLINE DEVICE float3 operator*(const float3& a, const float3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
INLINE DEVICE float3 operator*(const float3& a, float s)
{
    return {a.x * s, a.y * s, a.z * s};
}
INLINE DEVICE float3 operator*(float s, const float3& a)
{
    return {a.x * s, a.y * s, a.z * s};
}

INLINE DEVICE float3 operator*=(const float3& a, const float3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
INLINE DEVICE float3 operator*=(const float3& a, float s)
{
    return {a.x * s, a.y * s, a.z * s};
}

INLINE DEVICE float3 operator+(const float3& a, const float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
INLINE DEVICE float3 operator+(const float3& a, float s)
{
    return {a.x + s, a.y + s, a.z + s};
}
INLINE DEVICE float3 operator+(float s, const float3& a)
{
    return {a.x + s, a.y + s, a.z + s};
}

INLINE DEVICE float3 operator+=(const float3& a, const float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

INLINE DEVICE float3 operator-(const float3& a, const float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

INLINE DEVICE float3 operator-(const float3& a) { return {-a.x, -a.y, -a.z}; }

INLINE DEVICE float4 operator+(const float4& a, const float4& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
INLINE DEVICE float4 operator+(const float4& a, float s)
{
    return {a.x + s, a.y + s, a.z + s, a.w + s};
}
INLINE DEVICE float4 operator+(float s, const float4& a)
{
    return {a.x + s, a.y + s, a.z + s, a.w + s};
}

INLINE DEVICE float4 operator+=(const float4& a, const float4& b)
{
    return a + b;
}

#endif

INLINE DEVICE float3 cross(float3 a, float3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
INLINE DEVICE float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
INLINE DEVICE float length(float3 a) { return sqrtf(dot(a, a)); }
INLINE DEVICE float3 normalize(float3 a) { return a / length(a); }
INLINE DEVICE float3 mix(float3 a, float3 b, float t)
{
    return a + (b - a) * t;
}

INLINE DEVICE float luminance(const float3& a)
{
    // CIE RGB -> XYZ
    // http://www.brucelindbloom.com/index.html
    return dot(a, float3{0.1762044f, 0.8129847f, 0.0108109f});
}

INLINE DEVICE void perspectiveFov(float m[16], float fov, float width,
                                  float height, float zNear, float zFar)
{
    float rad = fov;
    float h = cosf(0.5f * rad) / sinf(0.5f * rad);
    float w = h * height / width;

    auto AT = [](int col, int row) { return col * 4 + row; };
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) m[AT(i, j)] = 0.0f;

    m[AT(0, 0)] = w;
    m[AT(1, 1)] = h;
    m[AT(2, 2)] = -(zFar + zNear) / (zFar - zNear);
    m[AT(2, 3)] = -1.0f;
    m[AT(3, 2)] = -(2.0f * zFar * zNear) / (zFar - zNear);
}

INLINE DEVICE void lookAt(float m[16], float3 eye, float3 center, float3 up)
{
    float3 f(normalize(center - eye));
    float3 s(normalize(cross(f, up)));
    float3 u(cross(s, f));

    auto AT = [](int col, int row) { return col * 4 + row; };
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) m[AT(i, j)] = i == j ? 1.0f : 0.0f;

    m[AT(0, 0)] = s.x;
    m[AT(1, 0)] = s.y;
    m[AT(2, 0)] = s.z;
    m[AT(0, 1)] = u.x;
    m[AT(1, 1)] = u.y;
    m[AT(2, 1)] = u.z;
    m[AT(0, 2)] = -f.x;
    m[AT(1, 2)] = -f.y;
    m[AT(2, 2)] = -f.z;
    m[AT(3, 0)] = -dot(s, eye);
    m[AT(3, 1)] = -dot(u, eye);
    m[AT(3, 2)] = dot(f, eye);
}

INLINE DEVICE int ceiling_div(int dividend, int divisor)
{
    return (dividend + divisor - 1) / divisor;
}
INLINE DEVICE int next_multiple(int dividend, int divisor)
{
    return ceiling_div(dividend, divisor) * divisor;
}

template <typename T>
INLINE DEVICE float clamp(const T& x, const T& xmin, const T& xmax)
{
    return min(max(x, xmin), xmax);
}
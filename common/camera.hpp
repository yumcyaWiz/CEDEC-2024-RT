#pragma once
#include "common/math.hpp"
#include "common/types.hpp"

struct RayGenerator
{
    float3 m_origin = {0.0f, 0.0f, 0.0f};
    float3 m_right = {1.0f, 0.0f, 0.0f};
    float3 m_up = {0.0f, 1.0f, 0.0f};

    INLINE DEVICE void lookat(const float3& eye, const float3& center,
                              const float3& up, float fovy, int imageWidth,
                              int imageHeight)
    {
        const float3 f(normalize(center - eye));
        const float3 s(normalize(cross(f, up)));
        const float3 u(cross(s, f));

        const float tanThetaY = tan(fovy * 0.5f);
        const float tanThetaX = tanThetaY / imageHeight * imageWidth;

        m_origin = eye;
        m_right = s * tanThetaX;
        m_up = u * tanThetaY;
    }

    INLINE DEVICE void shoot(float3* ro, float3* rd, float u, float v)
    {
        const float3 from = m_origin;
        const float3 forward = normalize(cross(m_up, m_right));
        const float3 to = m_origin + forward + mix(-m_right, m_right, u) +
                          mix(m_up, -m_up, v);
        *ro = from;
        *rd = normalize(to - from);
    }
};
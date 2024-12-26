#pragma once

#include <cstdarg>
#include <cstdio>
#include <vector>

#include "GLFW/glfw3.h"
#include "GLFW/glfw3native.h"
#include "bitmapString.h"
#include "common/math.hpp"
#include "stb_image_write.h"

class ImageTexture
{
   public:
    void init(int w, int h)
    {
        glGenTextures(1, &m_texture);
        m_width = w;
        m_height = h;
        m_pixelsRGBA.resize(w * h * 4);
    }
    void updateAndDraw()
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, m_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, m_pixelsRGBA.data());

        glBegin(GL_QUADS);
        glColor3f(1.0f, 1.0f, 1.0f);
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-1.0, -1.0, 0.0);
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-1.0, 1.0, 0.0);
        glTexCoord2f(1.0, 1.0);
        glVertex3f(1.0, 1.0, 0.0);
        glTexCoord2f(1.0, 0.0);
        glVertex3f(1.0, -1.0, 0.0);
        glEnd();

        glDisable(GL_TEXTURE_2D);
    }

    const uint8_t* data() const { return m_pixelsRGBA.data(); }
    uint8_t* data() { return m_pixelsRGBA.data(); }
    int width() const { return m_width; }
    int height() const { return m_width; }

    GLuint m_texture = 0;
    int m_width = 0;
    int m_height = 0;
    std::vector<uint8_t> m_pixelsRGBA;
};

inline void drawLine(float3 a, float3 b, float3 color)
{
    glBegin(GL_LINES);
    glColor3f(color.x, color.y, color.z);
    glVertex3f(a.x, a.y, a.z);
    glVertex3f(b.x, b.y, b.z);
    glEnd();
}

inline void drawGridXZ(float brightness = 0.5f, int hSizeGrid = 4)
{
    glBegin(GL_LINES);
    glColor3f(brightness, brightness, brightness);
    for (int z = -hSizeGrid; z <= hSizeGrid; z++)
    {
        glVertex3f(-hSizeGrid, 0, z);
        glVertex3f(hSizeGrid, 0, z);
    }
    for (int x = -hSizeGrid; x <= hSizeGrid; x++)
    {
        glVertex3f(x, 0, -hSizeGrid);
        glVertex3f(x, 0, hSizeGrid);
    }
    glEnd();
}

inline void drawXYZAxis()
{
    glLineWidth(2);
    glBegin(GL_LINES);
    glColor3f(1, 0, 0);
    glVertex3f(0, 0, 0);
    glVertex3f(1, 0, 0);
    glColor3f(0, 1, 0);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 1, 0);
    glColor3f(0, 0, 1);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 1);
    glEnd();
    glLineWidth(1);
}

class CameraControl
{
   public:
    bool is_updated()
    {
        if (m_updated)
        {
            m_updated = false;
            return true;
        }
        else { return false; }
    }

    void mouseButtonCallback(GLFWwindow* window, int button, int action,
                             int mods)
    {
        if (action == GLFW_PRESS && button == 0) { m_isMouseLDown = true; }
        if (action == GLFW_RELEASE && button == 0) { m_isMouseLDown = false; }
        if (action == GLFW_PRESS && button == 1) { m_isMouseRDown = true; }
        if (action == GLFW_RELEASE && button == 1) { m_isMouseRDown = false; }
        if (action == GLFW_PRESS && button == 2) { m_isMouseMDown = true; }
        if (action == GLFW_RELEASE && button == 2) { m_isMouseMDown = false; }
    }

    void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
    {
        if (m_isInit == false)
        {
            m_xpos = xpos;
            m_ypos = ypos;
            m_isInit = true;
            return;
        }

        float dx = xpos - m_xpos;
        float dy = ypos - m_ypos;

        m_xpos = xpos;
        m_ypos = ypos;

        float3 cameraLocal = m_cameraOrig - m_cameraLookat;
        float r = length(cameraLocal);

        if (m_isMouseLDown)
        {
            const float sensitivity = 0.004f;
            {
                float sinTheta = sinf(dx * sensitivity);
                float cosTheta = cosf(dx * sensitivity);
                float new_x =
                    cosTheta * cameraLocal.x - sinTheta * cameraLocal.z;
                float new_z =
                    sinTheta * cameraLocal.x + cosTheta * cameraLocal.z;

                cameraLocal.x = new_x;
                cameraLocal.z = new_z;
            }

            {
                float xz = sqrtf(cameraLocal.x * cameraLocal.x +
                                 cameraLocal.z * cameraLocal.z);

                float sinTheta = sinf(dy * sensitivity);
                float cosTheta = cosf(dy * sensitivity);
                float new_xz = cosTheta * xz - sinTheta * cameraLocal.y;
                float new_y = sinTheta * xz + cosTheta * cameraLocal.y;

                if (-r + r * 0.01f < new_y && new_y < r - r * 0.01f)
                {
                    cameraLocal.x = cameraLocal.x * (new_xz / xz);
                    cameraLocal.z = cameraLocal.z * (new_xz / xz);
                    cameraLocal.y = new_y;
                }
            }

            m_cameraOrig = m_cameraLookat + cameraLocal;
            m_updated = true;
        }

        if (m_isMouseRDown)
        {
            const float sensitivity = 0.002f;
            float new_r = std::max(r - r * sensitivity * dy, 0.01f);
            float s = new_r / r;
            m_cameraOrig = m_cameraLookat + cameraLocal * s;
            m_updated = true;
        }

        if (m_isMouseMDown)
        {
            const float sensitivity = 0.001f;
            float3 forward = normalize(m_cameraLookat - m_cameraOrig);
            float3 right = normalize(cross(forward, {0, 1, 0}));
            float3 up = cross(right, forward);

            float amount = std::max(r * sensitivity, 0.01f);
            float3 delta = -right * dx * amount + up * dy * amount;
            m_cameraOrig = m_cameraOrig + delta;
            m_cameraLookat = m_cameraLookat + delta;
            m_updated = true;
        }
    }

    float3 cameraOrigin() const { return m_cameraOrig; }
    float3 cameraLookAt() const { return m_cameraLookat; }
    bool m_isInit = false;
    bool m_isMouseLDown = false;
    bool m_isMouseRDown = false;
    bool m_isMouseMDown = false;
    float3 m_cameraOrig = {8.0f, 8.0f, 8.0f};
    float3 m_cameraLookat = {0.0f, 0.0f, 0.0f};

    float m_xpos = 0.0f;
    float m_ypos = 0.0f;

    bool m_updated = false;
};

inline void saveScreenshot(const char* file, GLFWwindow* window)
{
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    std::vector<uint8_t> image(width * height * 3);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, image.data());

    std::vector<uint8_t> imageF(width * height * 3);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int dst_y = height - y - 1;
            for (int i = 0; i < 3; i++)
            {
                imageF[(dst_y * width + x) * 3 + i] =
                    image[(y * width + x) * 3 + i];
            }
        }
    stbi_write_png(file, width, height, 3, imageF.data(), 0);
}

inline void drawString2d(GLFWwindow* window, float x, float y,
                         const char* format, ...)
{
    char buffer[256];

    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    int w, h;
    glfwGetWindowSize(window, &w, &h);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glRasterPos2f((x / w) * 2.0f - 1.0f, (y / h) * 2.0f - 1.0f);
    glutBitmapString(buffer);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

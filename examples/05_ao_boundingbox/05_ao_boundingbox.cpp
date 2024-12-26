
// #include "GL/glew.h"
#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>

#include <cmath>
#include <filesystem>
#include <vector>

#include "GLFW/glfw3.h"
#include "GLFW/glfw3native.h"
#include "common/camera.hpp"
#include "common/core.hpp"
#include "common/loader.hpp"
#include "common/math.hpp"
#include "common/misc.hpp"
#include "common/shader.hpp"
#include "common/typedbuffer.hpp"
#include "tiny_obj_loader.h"

int main()
{
    if (oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0))
    {
        printf("failed to init..\n");
        return 0;
    }
    int deviceIdx = 0;

    oroError err;
    err = oroInit(0);
    oroDevice device;
    err = oroDeviceGet(&device, deviceIdx);
    oroCtx ctx;
    err = oroCtxCreate(&ctx, 0, device);
    oroCtxSetCurrent(ctx);

    oroStream stream = 0;
    oroStreamCreate(&stream);
    oroDeviceProp props;
    oroGetDeviceProperties(&props, device);

    bool isNvidia = oroGetCurAPI(0) & ORO_API_CUDADRIVER;

    printf("Device: %s\n", props.name);
    printf("Cuda: %s\n", isNvidia ? "Yes" : "No");

    std::string baseDir = "../"; /* repository root */

    std::vector<std::string> options;
    options.push_back("-I" + baseDir);

    if (isNvidia)
    {
        // options.push_back("--gpu-architecture=compute_70");
        options.push_back(NV_ARG_LINE_INFO);
    }
    else { options.push_back(AMD_ARG_LINE_INFO); }
#if defined(DEBUG_GPU)
    if (isNvidia) { options.push_back("-G"); }
    else { options.push_back("-O0"); }
#endif

    const std::filesystem::path shader_path = {
        baseDir + "/examples/05_ao_boundingbox/05_ao_boundingbox.cu"};

    Shader shader(shader_path.generic_string().c_str(), "05_ao_boundingbox.cu",
                  options);

    int screenWidth = 1920;
    int screenHeight = 1080;

    glfwInit();
    GLFWwindow* window =
        glfwCreateWindow(screenWidth, screenHeight, "", NULL, NULL);
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);

    TypedBuffer<uint8_t> pixelsBuffer(TYPED_BUFFER_DEVICE);
    pixelsBuffer.allocate(screenWidth * screenHeight * 4);

    ImageTexture imageTexture;
    imageTexture.init(screenWidth, screenHeight);

    static CameraControl camera;

    glfwSetCursorPosCallback(window,
                             [](GLFWwindow* window, double xpos, double ypos)
                             { camera.cursorPosCallback(window, xpos, ypos); });
    glfwSetMouseButtonCallback(
        window, [](GLFWwindow* window, int button, int action, int mods)
        { camera.mouseButtonCallback(window, button, action, mods); });

    glfwSetKeyCallback(
        window,
        [](GLFWwindow* window, int key, int scancode, int action, int mods)
        {
            if (key == GLFW_KEY_S && action == GLFW_PRESS)
            {
                saveScreenshot("ss.png", window);
            }
        });
    std::vector<Triangle> triangles =
        loadTrianglesFromObj("../assets/blocks_ao.obj", "../assets/");

    TypedBuffer<Triangle> trianglesBuf(TYPED_BUFFER_DEVICE);
    trianglesBuf.allocate(triangles.size());
    oroMemcpyHtoD((oroDeviceptr)trianglesBuf.data(), triangles.data(),
                  trianglesBuf.bytes());

    float time = 0.0f;
    while (glfwWindowShouldClose(window) == 0)
    {
        time += 0.01f;

        glfwPollEvents();
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        float3 cameraOrig = camera.cameraOrigin();

        float fovy = PI / 4.0f;
        float viewM[16];
        lookAt(viewM, cameraOrig, camera.cameraLookAt(), {0, 1, 0});
        float projM[16];
        perspectiveFov(projM, fovy, screenWidth, screenHeight, 0.10000000f,
                       1000.00000);

        RayGenerator rayGen;
        rayGen.lookat(cameraOrig, camera.cameraLookAt(), {0, 1, 0}, fovy,
                      screenWidth, screenHeight);

        OroStopwatch sw(stream);
        sw.start();
        shader.launch("kernelMain",
                      ShaderArgument()
                          .ptr(&pixelsBuffer)
                          .ptr(&rayGen)
                          .value(screenWidth)
                          .value(screenHeight)
                          .ptr(&trianglesBuf),
                      ceiling_div(screenWidth * screenHeight, 128), 1, 1, 128,
                      1, 1, stream);

        sw.stop();
        float kernelTimeMS = sw.getMs();

        oroMemcpyDtoHAsync(imageTexture.data(),
                           (oroDeviceptr)pixelsBuffer.data(),
                           pixelsBuffer.bytes(), stream);
        oroStreamSynchronize(stream);

        imageTexture.updateAndDraw();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glLoadMatrixf(projM);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glLoadMatrixf(viewM);

        // GRID
        drawGridXZ();

        // XYZ
        drawXYZAxis();

        // for (const Triangle& tri : triangles)
        //{
        //	drawLine(tri.vertices[0], tri.vertices[1], { 1,1,1 });
        //	drawLine(tri.vertices[1], tri.vertices[2], { 1,1,1 });
        //	drawLine(tri.vertices[2], tri.vertices[0], { 1,1,1 });
        // }

        glColor4f(1.0f, 0.2, 1.0f, 1.0f);
        drawString2d(window, 10, 10, "kenel %.1f ms", kernelTimeMS);

        glfwSwapBuffers(window);
    }

    return 0;
}
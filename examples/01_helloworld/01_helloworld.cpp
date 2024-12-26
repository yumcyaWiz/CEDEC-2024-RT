
// #include "GL/glew.h"
#include <Orochi/Orochi.h>

#include <cmath>
#include <filesystem>
#include <vector>

#include "GLFW/glfw3.h"
#include "GLFW/glfw3native.h"
#include "common/math.hpp"
#include "common/misc.hpp"
#include "common/shader.hpp"
#include "common/typedbuffer.hpp"

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
        options.push_back("--gpu-architecture=compute_70");
        options.push_back(NV_ARG_LINE_INFO);
    }
    else { options.push_back(AMD_ARG_LINE_INFO); }
#if defined(DEBUG_GPU)
    if (isNvidia) { options.push_back("-G"); }
    else { options.push_back("-O0"); }
#endif

    const std::filesystem::path shader_path = {
        baseDir + "/examples/01_helloworld/01_helloworld.cu"};
    Shader shader(shader_path.generic_string().c_str(), "01_helloworld.cu",
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

    float time = 0.0f;
    while (glfwWindowShouldClose(window) == 0)
    {
        time += 0.01f;

        glfwPollEvents();
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        // update image
        shader.launch("kernelMain",
                      ShaderArgument()
                          .ptr(&pixelsBuffer)
                          .value(screenWidth)
                          .value(screenHeight),
                      ceiling_div(screenWidth * screenHeight, 256), 1, 1, 256,
                      1, 1, stream);

        oroMemcpyDtoHAsync(imageTexture.data(),
                           (oroDeviceptr)pixelsBuffer.data(),
                           pixelsBuffer.bytes(), stream);
        oroStreamSynchronize(stream);

        imageTexture.updateAndDraw();

        glfwSwapBuffers(window);
    }

    return 0;
}
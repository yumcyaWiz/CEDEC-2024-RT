// #include "GL/glew.h"
#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>
#include <hiprt/hiprt.h>

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
#include "common/options.hpp"
#include "common/reservoir.hpp"
#include "common/shader.hpp"
#include "common/typedbuffer.hpp"
#include "tiny_obj_loader.h"

int main()
{
    // 画面サイズ
    constexpr int width = 1920;
    constexpr int height = 1080;

    // Orochiを初期化
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

    // HIPRTCのコンパイルオプション
    std::vector<std::string> shader_options;
    shader_options.push_back("-I" + baseDir);
    shader_options.push_back("-I" + baseDir + "libs/hiprt");
    shader_options.push_back(
        "-DNO_VECTOR_OP_OVERLOAD");  // A workaround for operator overload
                                     // conflicts with HIPRT library.

    if (isNvidia) { shader_options.push_back(NV_ARG_LINE_INFO); }
    else { shader_options.push_back(AMD_ARG_LINE_INFO); }
#if defined(DEBUG_GPU)
    if (isNvidia) { shader_options.push_back("-G"); }
    else { shader_options.push_back("-O0"); }
#endif

    // HIPRTの初期化
    // hiprtSetLogLevel(hiprtLogLevelInfo);
    hiprtContext hContext = 0;
    hiprtError herr;
    herr = hiprtCreateContext(HIPRT_API_VERSION,
                              {oroGetRawCtx(ctx), oroGetRawDevice(device),
                               isNvidia ? hiprtDeviceNVIDIA : hiprtDeviceAMD},
                              hContext);

    // HIPコードのコンパイル
    const std::filesystem::path shader_path = {
        baseDir + "/examples/10_restir_di/10_restir_di.cu"};
    Shader shader(shader_path.generic_string().c_str(), "10_restir_di.cu",
                  shader_options, UseHiprt(hContext).func("raycast"));

    // GLFWの初期化とウィンドウ生成
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(width, height, "", NULL, NULL);
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);

    // デバイス上にPixel bufferを作成
    // NOTE (Kenta Eto): pixel bufferは最終的に画面に表示されるRGB値を保存する
    TypedBuffer<uint8_t> pixel_buffer(TYPED_BUFFER_DEVICE);
    pixel_buffer.allocate(4 * width * height);

    // デバイス上にaccumulation bufferを作成
    // NOTE (Kenta Eto): accumulation
    // bufferはパストレーシングによる計算結果とサンプル数を保存する
    TypedBuffer<float4> accumulation_buffer(TYPED_BUFFER_DEVICE);
    accumulation_buffer.allocate(width * height);

    // デバイス上にvisibility bufferを作成
    // NOTE (Kenta Eto): visibility bufferは最初の交差点の三角形インデックス,
    // barycentric coordinateを保存する
    TypedBuffer<Visibility> visibility_buffer(TYPED_BUFFER_DEVICE);
    visibility_buffer.allocate(width * height);

    // デバイス上にreservoir bufferを作成
    // NOTE (Kenta Eto): reservoir bufferはピクセルごとのReservoirを保存する
    TypedBuffer<Reservoir> reservoir_buffer0(TYPED_BUFFER_DEVICE);
    reservoir_buffer0.allocate(width * height);

    // Spatial resampling用にreservoir bufferをdouble buffering
    TypedBuffer<Reservoir> reservoir_buffer1(TYPED_BUFFER_DEVICE);
    reservoir_buffer1.allocate(width * height);

    // Temporal resampling後のReservoirを保存するためのbuffer
    TypedBuffer<Reservoir> temporal_reservoir_buffer(TYPED_BUFFER_DEVICE);
    temporal_reservoir_buffer.allocate(width * height);

    // ホスト上にPixel bufferを作成
    ImageTexture imageTexture;
    imageTexture.init(width, height);

    static CameraControl camera;
    static Options options;

    // GLFWのコールバック関数をセット
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
            // Temporal resamplingのOn/Offの切り替え
            if (key == GLFW_KEY_1 && action == GLFW_PRESS)
            {
                options.use_temporal_resampling =
                    !options.use_temporal_resampling;
            }

            // Spatial resamplingのOn/Offの切り替え
            if (key == GLFW_KEY_2 && action == GLFW_PRESS)
            {
                options.use_spatial_resampling =
                    !options.use_spatial_resampling;
            }

            // Unshadowed, Shadowed target functionの切り替え
            if (key == GLFW_KEY_3 && action == GLFW_PRESS)
            {
                options.use_shadowed_target_function =
                    !options.use_shadowed_target_function;
            }

            // Visibility reuseのOn/Off
            if (key == GLFW_KEY_4 && action == GLFW_PRESS)
            {
                options.use_visibility_reuse = !options.use_visibility_reuse;
            }

            // accumulationの切り替え
            if (key == GLFW_KEY_A && action == GLFW_PRESS)
            {
                options.accumulate = !options.accumulate;
            }

            // スクリーンショットの保存
            if (key == GLFW_KEY_S && action == GLFW_PRESS)
            {
                saveScreenshot("ss.png", window);
            }
        });

    // シーンの読み込み
    std::vector<Triangle> triangles =
        loadTrianglesFromObj("../assets/blocks_restir.obj", "../assets/");

    // blocks_restir.obj 1
    camera.m_cameraOrig = float3{-0.579885, 22.194597, -6.567105};
    camera.m_cameraLookat = float3{5.224952, 20.847435, 1.431192};

    // blocks_restir.obj 2
    // camera.m_cameraOrig = float3{-14.853887, 27.826096, -46.717506};
    // camera.m_cameraLookat = float3{-3.813484, 11.610589, 4.625816};

    // Area lightになっている三角形のインデックスの配列を作成
    std::vector<uint32_t> light_indices;
    for (int i = 0; i < triangles.size(); ++i)
    {
        const auto& triangle = triangles[i];
        if (triangle.emissive.x > 0.0f || triangle.emissive.y > 0.0f ||
            triangle.emissive.z > 0.0f)
        {
            light_indices.push_back(i);
        }
    }
    printf("lights: %d\n", light_indices.size());

    // シーンデータをホストからデバイスにコピー
    TypedBuffer<Triangle> triangle_buffer(TYPED_BUFFER_DEVICE);
    triangle_buffer.allocate(triangles.size());
    oroMemcpyHtoD((oroDeviceptr)triangle_buffer.data(), triangles.data(),
                  triangle_buffer.bytes());

    TypedBuffer<uint32_t> light_buffer(TYPED_BUFFER_DEVICE);
    light_buffer.allocate(light_indices.size());
    oroMemcpyHtoD((oroDeviceptr)light_buffer.data(), light_indices.data(),
                  light_buffer.bytes());

    // HIPRT Geometryの作成
    hiprtGeometry geom = buildHiprtGeometry(hContext, triangles);

    // accumulation bufferを初期化
    shader.launch(
        "clear",
        ShaderArgument().ptr(&accumulation_buffer).value(width).value(height),
        ceiling_div(width * height, 256), 1, 1, 256, 1, 1, stream);

    // application loop
    float time = 0.0f;
    int frame = 0;
    while (glfwWindowShouldClose(window) == 0)
    {
        time += 0.01f;
        frame++;

        glfwPollEvents();
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        float3 cameraOrig = camera.cameraOrigin();

        float fovy = PI / 4.0f;
        float viewM[16];
        lookAt(viewM, cameraOrig, camera.cameraLookAt(), {0, 1, 0});
        float projM[16];
        perspectiveFov(projM, fovy, width, height, 0.10000000f, 1000.00000);

        // RayGeneratorのセットアップ
        RayGenerator rayGen;
        rayGen.lookat(cameraOrig, camera.cameraLookAt(), {0, 1, 0}, fovy, width,
                      height);

        // タイマー計測開始
        OroStopwatch sw(stream);
        sw.start();

        if (camera.is_updated())
        {
            // カメラが動いた時にaccumulation bufferを初期化
            shader.launch("clear",
                          ShaderArgument()
                              .ptr(&accumulation_buffer)
                              .value(width)
                              .value(height),
                          ceiling_div(width * height, 256), 1, 1, 256, 1, 1,
                          stream);
        }

        // 最初の交差点の情報をVisibility bufferに保存
        shader.launch("raycast",
                      ShaderArgument()
                          .value(width)
                          .value(height)
                          .value(geom)
                          .ptr(&triangle_buffer)
                          .ptr(&rayGen)
                          .ptr(&visibility_buffer),
                      ceiling_div(width * height, 256), 1, 1, 256, 1, 1,
                      stream);

        // RISによって初期サンプルを生成
        shader.launch("generate_candidate",
                      ShaderArgument()
                          .value(width)
                          .value(height)
                          .value(frame)
                          .value(geom)
                          .ptr(&triangle_buffer)
                          .ptr(&visibility_buffer)
                          .value(cameraOrig)
                          .ptr(&light_buffer)
                          .value(options)
                          .ptr(&reservoir_buffer0),
                      ceiling_div(width * height, 256), 1, 1, 256, 1, 1,
                      stream);

        // Temporal resampling
        shader.launch("temporal_resampling",
                      ShaderArgument()
                          .value(width)
                          .value(height)
                          .value(frame)
                          .value(geom)
                          .ptr(&triangle_buffer)
                          .ptr(&visibility_buffer)
                          .value(cameraOrig)
                          .value(options)
                          .ptr(&temporal_reservoir_buffer)
                          .ptr(&reservoir_buffer0),
                      ceiling_div(width * height, 256), 1, 1, 256, 1, 1,
                      stream);

        // Temporal resampling後のReservoirをtemporal reservoir bufferに保存
        shader.launch("save_temporal_reservoir",
                      ShaderArgument()
                          .value(width)
                          .value(height)
                          .ptr(&reservoir_buffer0)
                          .ptr(&temporal_reservoir_buffer),
                      ceiling_div(width * height, 256), 1, 1, 256, 1, 1,
                      stream);

        // double buffering
        TypedBuffer<Reservoir>* buf_input = &reservoir_buffer0;
        TypedBuffer<Reservoir>* buf_output = &reservoir_buffer1;

        // Spatial resampling
        // NOTE (Kenta Eto): ここではSpatial resamplingのプロセスをspatial
        // resampling passes回だけ繰り返す.
        for (int k = 0; k < options.spatial_resampling_passes; ++k)
        {
            if (k != 0)
            {
                // swap buffers
                std::swap(buf_input, buf_output);
            }

            shader.launch("spatial_resampling",
                          ShaderArgument()
                              .value(width)
                              .value(height)
                              .value(frame)
                              .value(k)
                              .value(geom)
                              .ptr(&triangle_buffer)
                              .ptr(&visibility_buffer)
                              .value(cameraOrig)
                              .value(options)
                              .ptr(buf_input)
                              .ptr(buf_output),
                          ceiling_div(width * height, 256), 1, 1, 256, 1, 1,
                          stream);
        }

        // 寄与の計算
        shader.launch("resolve",
                      ShaderArgument()
                          .ptr(&accumulation_buffer)
                          .value(width)
                          .value(height)
                          .value(geom)
                          .ptr(&triangle_buffer)
                          .ptr(&visibility_buffer)
                          .value(cameraOrig)
                          .value(options)
                          .ptr(buf_output),
                      ceiling_div(width * height, 256), 1, 1, 256, 1, 1,
                      stream);

        // accumulation bufferの平均を取ってトーンマッピングをかけ、結果をpixel
        // bufferに書き込む
        shader.launch("tone_mapping",
                      ShaderArgument()
                          .ptr(&pixel_buffer)
                          .ptr(&accumulation_buffer)
                          .value(width)
                          .value(height),
                      ceiling_div(width * height, 256), 1, 1, 256, 1, 1,
                      stream);

        // タイマー計測終了
        sw.stop();
        float kernelTimeMS = sw.getMs();

        // pixel bufferをデバイスからホストにコピー
        oroMemcpyDtoHAsync(imageTexture.data(),
                           (oroDeviceptr)pixel_buffer.data(),
                           pixel_buffer.bytes(), stream);
        oroStreamSynchronize(stream);

        // pixel bufferを画面に描画
        imageTexture.updateAndDraw();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glLoadMatrixf(projM);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glLoadMatrixf(viewM);

        // Gridを描画
        drawGridXZ();

        // XYZ Axisを描画
        drawXYZAxis();

        // カーネルの実行時間を描画
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        drawString2d(window, 10, 10, "kernel: %.1f ms", kernelTimeMS);

        glfwSwapBuffers(window);
    }

    return 0;
}
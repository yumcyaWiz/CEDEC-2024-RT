#include "common/camera.hpp"
#include "common/core.hpp"
#include "common/kernels/common.cu"
#include "common/options.hpp"
#include "common/raytrace.hpp"
#include "common/rng.hpp"

#define USE_VISIBILITY_REUSE
#include "common/reservoir.hpp"

KERNEL void path_trace(int width, int height, int frame,
                       const hiprtGeometry hiprt_geom,
                       const TypedBuffer<Triangle> triangles,
                       const TypedBuffer<uint32_t> lights, RayGenerator raygen,
                       Options options, TypedBuffer<float4> accumulation)
{
    // スレッドIDを計算
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    // Pixel bufferのインデックスを計算
    const int xi = tid % width;
    const int yi = tid / width;
    const int pixel_idx = xi + (height - yi - 1) * width;

    // 乱数生成器の初期化
    PCG random(hashPCG3(xi, yi, frame), 0);

    // カメラからレイを生成
    float3 ro, rd;
    raygen.shoot(&ro, &rd, (float)xi / width, (float)yi / height);

    // 初期値をセット
    Ray ray = make_ray(ro, rd);
    float3 radiance = {0.0f, 0.0f, 0.0f};
    float3 throughput = {1.0f, 1.0f, 1.0f};

    // パストレーシングのループ
    for (int depth = 0; depth < options.max_depth; ++depth)
    {
        // レイトレース
        Intersection isect;
        if (!raytrace(ray, hiprt_geom, isect))
        {
            // 空に当たった場合
            // NOTE (Kenta Eto):
            // この例では空が光源の場合を考慮していない
            break;
        }

        // 光源に当たった場合
        const Triangle& hit_triangle = triangles[isect.index];
        if (has_emission(hit_triangle))
        {
            // 光源が直接見えるケースの時だけ寄与を追加
            // NOTE (Kenta Eto):
            // それ以外のケースでも寄与を追加すると寄与を二重にカウントしてしまい、明るすぎるレンダリング結果が得られる
            if (depth == 0) { radiance += throughput * hit_triangle.emissive; }

            break;
        }

        // 交差点における法線などを計算
        const SurfaceInfo surf = make_surface_info(ray, isect, triangles);

        // RIS
        Reservoir reservoir;
        for (int i = 0; i < options.ris_sample_count; ++i)
        {
            ReservoirSample sample;
            sample.origin_position = surf.p;
            sample.origin_normal = surf.n;

            // 光源上の点をサンプリング
            const LightSample light_sample =
                sample_light(triangles, lights, random.uniformf(),
                             random.uniformf(), random.uniformf());
            sample.hit_position = light_sample.p;
            sample.hit_normal = light_sample.n;

            // 光源のemissionを保存
            const Triangle& light_triangle = triangles[light_sample.index];
            sample.radiance = light_triangle.emissive;

            // 光源サンプリングの確率密度関数の値の評価
            const float light_pdf =
                1.0f / lights.size() * 1.0f / area_of(light_triangle);

            // Target functionの評価
            const float p_hat = evaluate_target_function(
                sample.origin_position, sample.origin_normal,
                sample.hit_position, sample.hit_normal, sample.radiance,
                hiprt_geom, triangles, options.use_shadowed_target_function);

            // リサンプリング重みの計算
            const float weight = p_hat / light_pdf;

            // リサンプリング
            reservoir.update(sample, weight, random.uniformf());
        }

        // 寄与の評価
        {
            // BRDFの評価
            const float3 brdf = 1.0f / PI * hit_triangle.color;
            // 幾何項の評価
            const float G =
                geometry_term(surf.p, surf.n, reservoir.sample.hit_position,
                              reservoir.sample.hit_normal);

            // 光源の可視性のチェック
            const float V = check_visibility(
                surf.p, surf.n, reservoir.sample.hit_position, hiprt_geom);

            // Target functionの評価
            const float p_hat = evaluate_target_function(
                surf.p, surf.n, reservoir.sample.hit_position,
                reservoir.sample.hit_normal, reservoir.sample.radiance,
                hiprt_geom, triangles, options.use_shadowed_target_function);

            // Unbiased contribution weightの評価
            const float ucw =
                p_hat > 0.0f ? reservoir.w_sum / (reservoir.M * p_hat) : 0.0f;

            // 寄与を加算
            radiance +=
                throughput * brdf * G * V * reservoir.sample.radiance * ucw;
        }

        // 次のレイの方向を生成
        float3 wo;
        {
            const TangentBasis basis =
                make_tangent_basis(surf.n, isect.index, triangles);

            const float3 wo_local = sample_hemisphere(
                random.uniformf(), random.uniformf(), random.uniformf());
            wo = local_to_world(wo_local, basis);
        }

        // スループットを更新
        // NOTE (Kenta Eto):
        // 本来であれば(BRDF * 幾何項 / 方向サンプリングのpdf *
        // ヤコビアン)をthroughputにかける必要があるが、この例ではそれらが互いに相殺されるため、albedoしか残らない
        // BRDF = 1 / PI * albedo
        // 幾何項 = G(x1, x2) = cos(theta1) * cos(theta2) / |x2 - x1|^2
        // 方向サンプリングによるpdf = 1 / PI * cos(theta1)
        // ヤコビアン = G(x1, x2) / cos(theta1)
        throughput *= hit_triangle.color;

        // 次のレイを生成
        // NOTE (Kenta Eto):
        // 次のレイが地面にめり込まないように微小なオフセット分だけずらす必要がある
        ray = make_ray(offset_ray_position(surf.p, surf.n), wo);
    }

    // radianceとサンプル数をaccumulation bufferに書き込む
    if (options.accumulate)
    {
        accumulation[pixel_idx] += {radiance.x, radiance.y, radiance.z, 1.0f};
    }
    else
    {
        accumulation[pixel_idx] = {radiance.x, radiance.y, radiance.z, 1.0f};
    }
}
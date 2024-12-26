#include "common/camera.hpp"
#include "common/core.hpp"
#include "common/kernels/common.cu"
#include "common/options.hpp"
#include "common/raytrace.hpp"
#include "common/reservoir.hpp"
#include "common/rng.hpp"

KERNEL void raycast(int width, int height, const hiprtGeometry hiprt_geometry,
                    const TypedBuffer<Triangle> triangles, RayGenerator raygen,
                    TypedBuffer<Visibility> visibility_buffer)
{
    // スレッドIDを計算
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    // Visibility bufferのインデックスを計算
    const int xi = tid % width;
    const int yi = tid / width;
    const int pixel_idx = xi + (height - yi - 1) * width;

    // カメラからレイを生成
    float3 ro, rd;
    raygen.shoot(&ro, &rd, (float)xi / width, (float)yi / height);
    const Ray ray = make_ray(ro, rd);

    // レイトレース
    Intersection isect;
    raytrace(ray, hiprt_geometry, isect);

    // 交差点のbarycentric coordinateと三角形インデックスをVisibility
    // bufferに保存
    visibility_buffer[pixel_idx] = Visibility{isect.uv, isect.index};
}

KERNEL void generate_candidate(int width, int height, int frame,
                               const hiprtGeometry hiprt_geometry,
                               const TypedBuffer<Triangle> triangles,
                               const TypedBuffer<Visibility> visibility_buffer,
                               const float3 eye,
                               const TypedBuffer<uint32_t> lights,
                               const Options options,
                               TypedBuffer<Reservoir> reservoirs)
{
    // スレッドIDを計算
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    // reservoir bufferのインデックスを計算
    const int xi = tid % width;
    const int yi = tid / width;
    const int pixel_idx = xi + (height - yi - 1) * width;

    const Visibility visibility = visibility_buffer[pixel_idx];

    if (visibility.index == -1)
    {
        // 空に当たった場合
        reservoirs[pixel_idx] = Reservoir{};
        return;
    }

    const Triangle& triangle = triangles[visibility.index];

    if (has_emission(triangle))
    {
        // 光源に当たった場合
        reservoirs[pixel_idx] = Reservoir{};
        return;
    }

    // 乱数生成器の初期化
    PCG random(hashPCG4(xi, yi, frame, 0), 0);

    // 交差点における法線などを計算
    const SurfaceInfo surf = make_surface_info(visibility, triangles, eye);

    // RISによる候補サンプル生成
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
            surf.p, surf.n, sample.hit_position, sample.hit_normal,
            sample.radiance, hiprt_geometry, triangles, false);

        // リサンプリング重みの計算
        const float weight = p_hat / light_pdf;

        // リサンプリング
        reservoir.update(sample, weight, random.uniformf());
    }

    // Unbiased contribution weightの評価
    {
        const float p_hat = evaluate_target_function(
            surf.p, surf.n, reservoir.sample.hit_position,
            reservoir.sample.hit_normal, reservoir.sample.radiance,
            hiprt_geometry, triangles, options.use_shadowed_target_function);

        float ucw =
            p_hat > 0.0f ? reservoir.w_sum / (reservoir.M * p_hat) : 0.0f;

        reservoir.ucw = ucw;
    }

    // Visibilty reuse
    if (options.use_visibility_reuse)
    {
        reservoir.sample.visibility = check_visibility(
            surf.p, surf.n, reservoir.sample.hit_position, hiprt_geometry);
    }

    // save reservoir
    reservoirs[pixel_idx] = reservoir;
}

KERNEL void temporal_resampling(
    int width, int height, int frame, const hiprtGeometry hiprt_geometry,
    const TypedBuffer<Triangle> triangles,
    const TypedBuffer<Visibility> visibility_buffer, const float3 eye,
    const Options options, const TypedBuffer<Reservoir> previous_reservoirs,
    TypedBuffer<Reservoir> reservoirs)
{
    // スレッドIDを計算
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    // Reservoir bufferのインデックスを計算
    const int xi = tid % width;
    const int yi = tid / width;
    const int pixel_idx = xi + (height - yi - 1) * width;

    const Visibility visibility = visibility_buffer[pixel_idx];

    if (visibility.index == -1)
    {
        // 空に当たった場合
        return;
    }

    const Triangle& triangle = triangles[visibility.index];

    if (has_emission(triangle))
    {
        // 光源に当たった場合
        return;
    }

    if (!options.use_temporal_resampling) { return; }

    // 乱数生成器の初期化
    PCG random(hashPCG4(xi, yi, frame, 1), 0);

    // 交差点における法線などを計算
    const SurfaceInfo surf = make_surface_info(visibility, triangles, eye);

    // 現在と過去のフレームのReservoirを取得
    Reservoir previous_reservoir = previous_reservoirs[pixel_idx];
    Reservoir reservoir = reservoirs[pixel_idx];

    // M-capping
    // NOTE (Kenta Eto):
    // Reservoirのサンプル数(M)が大きくなりすぎると、新しいサンプルを受け入れる確率が0に近づいて同じサンプルがReservoirに残り続けてしまい、アーティファクトが出ることがある
    // これを防ぐためにMの値を抑えるという処理が実用上は必要になる
    previous_reservoir.M =
        min(previous_reservoir.M,
            20 * options.ris_sample_count);  // 初期サンプル数の20倍

    // temporal resampling
    {
        // リサンプリング重みの評価
        float weight;
        {
            // Target functionの評価
            float p_hat_y = evaluate_target_function(
                surf.p, surf.n, previous_reservoir.sample.hit_position,
                previous_reservoir.sample.hit_normal,
                previous_reservoir.sample.radiance, hiprt_geometry, triangles,
                options.use_shadowed_target_function);

            // Visibility Reuse
            if (options.use_visibility_reuse)
            {
                p_hat_y *= previous_reservoir.sample.visibility;
            }

            // Target
            // functionの類似度によって隣接Reservoirのサンプルの重みを動的に変える
            // Probabilistic Rejection [Tokyuoshi 2023 "Efficient Spatial
            // Resampling Using the PDF Similarity", p.5].
            previous_reservoir.M *=
                rejection_heuristics(reservoir, previous_reservoir, eye);

            // リサンプリング重みの評価
            weight = p_hat_y * previous_reservoir.ucw * previous_reservoir.M;
        }

        // リサンプリング
        reservoir.merge(previous_reservoir, weight, random.uniformf());
    }

    // Unbiased contribution weightの評価
    {
        const float p_hat = evaluate_target_function(
            surf.p, surf.n, reservoir.sample.hit_position,
            reservoir.sample.hit_normal, reservoir.sample.radiance,
            hiprt_geometry, triangles, options.use_shadowed_target_function);

        float ucw =
            p_hat > 0.0f ? reservoir.w_sum / (reservoir.M * p_hat) : 0.0f;

        reservoir.ucw = ucw;
    }

    // ReservoirをReservoir bufferに保存
    reservoirs[pixel_idx] = reservoir;
}

KERNEL void save_temporal_reservoir(
    int width, int height, const TypedBuffer<Reservoir> previous_reservoirs,
    TypedBuffer<Reservoir> reservoirs)
{
    // スレッドIDを計算
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    // Reservoir bufferのインデックスを計算
    const int xi = tid % width;
    const int yi = tid / width;
    const int pixel_idx = xi + (height - yi - 1) * width;

    // Temporal resampling後のReservoirを保存
    reservoirs[pixel_idx] = previous_reservoirs[pixel_idx];
}

KERNEL void spatial_resampling(int width, int height, int frame, int pass,
                               const hiprtGeometry hiprt_geometry,
                               const TypedBuffer<Triangle> triangles,
                               const TypedBuffer<Visibility> visibility_buffer,
                               const float3 eye, const Options options,
                               const TypedBuffer<Reservoir> previous_reservoirs,
                               TypedBuffer<Reservoir> reservoirs)
{
    // スレッドIDを計算
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    // Reservoir bufferのインデックスを計算
    const int xi = tid % width;
    const int yi = tid / width;
    const int pixel_idx = xi + (height - yi - 1) * width;

    const Visibility visibility = visibility_buffer[pixel_idx];

    if (visibility.index == -1)
    {
        // 空に当たった場合
        return;
    }

    const Triangle& triangle = triangles[visibility.index];

    if (has_emission(triangle))
    {
        // 光源に当たった場合
        return;
    }

    // 乱数生成器の初期化
    PCG random(hashPCG4(xi, yi, frame, 2 + pass), 0);

    // 交差点における法線などを計算
    const SurfaceInfo surf = make_surface_info(visibility, triangles, eye);

    // 現在のピクセル位置のReservoirを取得
    Reservoir reservoir = previous_reservoirs[pixel_idx];

    if (!options.use_spatial_resampling)
    {
        reservoirs[pixel_idx] = reservoir;
        return;
    }

    // Spatial resampling
    for (int k = 0; k < options.spatial_resampling_sample_count; ++k)
    {
        // 隣接ピクセルを二次元ガウス分布でサンプリングする
        // 95%のサンプルがspatial_resampling_radius内に位置するように分散を設定
        const float2 gaussian =
            sample_2d_gaussian(random.uniformf(), random.uniformf());
        int x = xi + options.spatial_resampling_radius / 1.96f * gaussian.x;
        int y = yi + options.spatial_resampling_radius / 1.96f * gaussian.y;

        if (x < 0 || x >= width || y < 0 || y >= height)
        {
            // スクリーン外の場合
            continue;
        }

        // 同じピクセルからリサンプリングされないようにする
        if (x == xi && y == yi) { continue; }

        const int pid = x + (height - y - 1) * width;
        const Visibility vis = visibility_buffer[pid];

        if (vis.index == -1)
        {
            // 空に当たった場合
            continue;
        }

        const Triangle& triangle = triangles[vis.index];

        if (has_emission(triangle))
        {
            // 光源に当たった場合
            continue;
        }

        Reservoir neighbour_reservoir = previous_reservoirs[pid];

        // リサンプリング重みの計算
        float weight;
        {
            // Target functionの評価
            float p_hat_y = evaluate_target_function(
                surf.p, surf.n, neighbour_reservoir.sample.hit_position,
                neighbour_reservoir.sample.hit_normal,
                neighbour_reservoir.sample.radiance, hiprt_geometry, triangles,
                options.use_shadowed_target_function);

            // Visibility Reuse
            if (options.use_visibility_reuse)
            {
                p_hat_y *= neighbour_reservoir.sample.visibility;
            }

            // Target
            // functionの類似度によって隣接Reservoirのサンプルの重みを動的に変える
            // Probabilistic Rejection [Tokyuoshi 2023 "Efficient Spatial
            // Resampling Using the PDF Similarity", p.5].
            neighbour_reservoir.M *=
                rejection_heuristics(reservoir, neighbour_reservoir, eye);

            // リサンプリング重みの評価
            weight = p_hat_y * neighbour_reservoir.ucw * neighbour_reservoir.M;
        }

        // リサンプリング
        reservoir.merge(neighbour_reservoir, weight, random.uniformf());
    }

    // Unbiased contribution weightの評価
    {
        const float p_hat = evaluate_target_function(
            surf.p, surf.n, reservoir.sample.hit_position,
            reservoir.sample.hit_normal, reservoir.sample.radiance,
            hiprt_geometry, triangles, options.use_shadowed_target_function);

        const float ucw =
            p_hat > 0.0f ? reservoir.w_sum / (reservoir.M * p_hat) : 0.0f;

        reservoir.ucw = ucw;
    }

    // ReservoirをReservoir bufferに保存
    reservoirs[pixel_idx] = reservoir;
}

KERNEL void resolve(TypedBuffer<float4> accumulation, int width, int height,
                    const hiprtGeometry hiprt_geometry,
                    const TypedBuffer<Triangle> triangles,
                    const TypedBuffer<Visibility> visibility_buffer,
                    const float3 eye, const Options options,
                    const TypedBuffer<Reservoir> reservoirs)
{
    // スレッドIDを計算
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (width * height <= tid) { return; }

    // reservoir bufferのインデックスを計算
    const int xi = tid % width;
    const int yi = tid / width;
    const int pixel_idx = xi + (height - yi - 1) * width;

    const Visibility visibility = visibility_buffer[pixel_idx];

    if (visibility.index == -1)
    {
        // 空に当たった場合
        accumulation[pixel_idx] = {0.0f, 0.0f, 0.0f, 1.0f};

        return;
    }

    const Triangle& triangle = triangles[visibility.index];

    if (has_emission(triangle))
    {
        // 光源に当たった場合
        accumulation[pixel_idx] = {triangle.emissive.x, triangle.emissive.y,
                                   triangle.emissive.z, 1.0f};

        return;
    }

    // 交差点における法線などを計算
    const SurfaceInfo surf = make_surface_info(visibility, triangles, eye);

    // Reservoirを取得
    const Reservoir& reservoir = reservoirs[pixel_idx];

    // 寄与の計算
    float3 radiance;
    {
        // BRDFの評価
        const float3 brdf = 1.0f / PI * triangle.color;
        // 幾何項の評価
        const float G =
            geometry_term(surf.p, surf.n, reservoir.sample.hit_position,
                          reservoir.sample.hit_normal);
        // 光源の可視性のチェック
        const float V = check_visibility(
            surf.p, surf.n, reservoir.sample.hit_position, hiprt_geometry);

        // 寄与を加算
        radiance = brdf * G * V * reservoir.sample.radiance * reservoir.ucw;
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
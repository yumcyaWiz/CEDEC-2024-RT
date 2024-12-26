#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

#include "common/camera.hpp"
#include "common/core.hpp"
#include "common/kernels/common.cu"
#include "common/options.hpp"
#include "common/raytrace.hpp"
#include "common/rng.hpp"

KERNEL void path_trace(int width, int height, int frame,
                       const hiprtGeometry hiprt_geom,
                       const TypedBuffer<Triangle> triangles,
                       RayGenerator raygen, Options options,
                       TypedBuffer<float4> accumulation)
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
            radiance += throughput * options.sky_color;
            break;
        }

        const Triangle& hit_triangle = triangles[isect.index];
        if (has_emission(hit_triangle))
        {
            // 光源に当たった場合
            radiance += throughput * triangles[isect.index].emissive;
            break;
        }

        // 交差点における法線などを計算
        const SurfaceInfo surf = make_surface_info(ray, isect, triangles);

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
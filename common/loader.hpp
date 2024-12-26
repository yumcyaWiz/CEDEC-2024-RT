#pragma once
#include "common/core.hpp"

#if !defined(__KERNELCC__)
#include <hiprt/hiprt.h>

#include <vector>

#include "tiny_obj_loader.h"

INLINE std::vector<Triangle> loadTrianglesFromObj(const char* filename,
                                                  const char* mtl_basedir)
{
    std::vector<Triangle> triangles;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string objError;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &objError,
                                filename, mtl_basedir);

    for (size_t s = 0; s < shapes.size(); s++)
    {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
        {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            float3 vertices[3];
            Triangle triangle;

            for (size_t v = 0; v < fv; v++)
            {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t vx =
                    attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy =
                    attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz =
                    attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                triangle.vertices[v] = {vx, vy, vz};
            }
            index_offset += fv;

            // per-face material
            int matId = shapes[s].mesh.material_ids[f];

            triangle.color = {materials[matId].diffuse[0],
                              materials[matId].diffuse[1],
                              materials[matId].diffuse[2]};
            triangle.emissive = {materials[matId].emission[0],
                                 materials[matId].emission[1],
                                 materials[matId].emission[2]};
            triangles.push_back(triangle);
        }
    }
    return triangles;
}

INLINE hiprtGeometry buildHiprtGeometry(hiprtContext hContext,
                                        const std::vector<Triangle>& triangles)
{
    TypedBuffer<Triangle> trianglesBuf(TYPED_BUFFER_DEVICE);
    trianglesBuf.allocate(triangles.size());
    oroMemcpyHtoD((oroDeviceptr)trianglesBuf.data(), (void*)triangles.data(),
                  trianglesBuf.bytes());

    TypedBuffer<float3> geomVertices(TYPED_BUFFER_HOST);
    geomVertices.allocate(triangles.size() * 3);
    for (int i = 0; i < triangles.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            geomVertices[i * 3 + j] = triangles[i].vertices[j];
        }
    }
    TypedBuffer<float3> geomVerticesBuffer = geomVertices.toDevice();

    hiprtTriangleMeshPrimitive mesh = {};
    mesh.triangleCount = triangles.size();
    mesh.vertexCount = triangles.size() * 3;
    mesh.vertexStride = sizeof(float3);
    mesh.vertices = geomVerticesBuffer.data();

    hiprtGeometryBuildInput geomInput = {};
    geomInput.type = hiprtPrimitiveTypeTriangleMesh;
    geomInput.primitive.triangleMesh = mesh;

    size_t geomTempSize = 0;
    hiprtBuildOptions buildOptions = {};
    buildOptions.buildFlags = hiprtBuildFlagBitPreferHighQualityBuild;
    hiprtGetGeometryBuildTemporaryBufferSize(hContext, geomInput, buildOptions,
                                             geomTempSize);

    TypedBuffer<char> geomTemp(TYPED_BUFFER_DEVICE);
    geomTemp.allocate(geomTempSize);

    hiprtGeometry geom = 0;
    hiprtCreateGeometry(hContext, geomInput, buildOptions, geom);
    hiprtBuildGeometry(hContext, hiprtBuildOperationBuild, geomInput,
                       buildOptions, geomTemp.data(), 0 /*stream*/, geom);
    oroStreamSynchronize(0);
    return geom;
}
#endif
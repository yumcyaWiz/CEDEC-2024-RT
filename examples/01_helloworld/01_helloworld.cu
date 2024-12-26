#include "common/core.hpp"
#include "common/typedbuffer.hpp"

KERNEL void kernelMain(TypedBuffer<uint8_t> pixels, int width, int height)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int xi = tid % width;
    int yi = tid / width;

    if (width * height <= tid) { return; }

    int pixelIdx = xi + (height - yi - 1) * width;
    pixels[pixelIdx * 4 + 0] = (float)xi / width * 255.0f;
    pixels[pixelIdx * 4 + 1] = (float)yi / height * 255.0f;
    pixels[pixelIdx * 4 + 2] = 128;
    pixels[pixelIdx * 4 + 3] = 255;
}

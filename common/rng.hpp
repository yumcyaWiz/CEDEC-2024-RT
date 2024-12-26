#pragma once
#include "common/types.hpp"

/*
 * PCG random number generator from
 * https://www.pcg-random.org/download.html#minimal-c-implementation
 */
struct PCG
{
    INLINE DEVICE PCG(uint64_t seed, uint64_t sequence)
    {
        state = 0U;
        inc = (sequence << 1u) | 1u;

        uniform();
        state += seed;
        uniform();
    }

    INLINE DEVICE uint32_t uniform()
    {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    INLINE DEVICE float uniformf()
    {
        uint32_t bits = (uniform() >> 9) | 0x3f800000;
        float value;
        memcpy(&value, &bits, sizeof(float));
        return value - 1.0f;
    }

    uint64_t state;  // RNG state.  All values are possible.
    uint64_t inc;    // Controls which RNG sequence(stream) is selected. Must
                     // *always* be odd.
};

// https://jcgt.org/published/0009/03/02/
INLINE DEVICE uint32_t hashPCG(uint32_t v)
{
    uint32_t state = v * 747796405 + 2891336453;
    uint32_t word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    return (word >> 22) ^ word;
}

INLINE DEVICE uint32_t hashPCG3(uint32_t x, uint32_t y, uint32_t z)
{
    return hashPCG(hashPCG(hashPCG(x) + y) + z);
}

INLINE DEVICE uint32_t hashPCG4(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
    return hashPCG(hashPCG(hashPCG(hashPCG(x) + y) + z) + w);
}
#pragma once

// Inlined float L2 so the memory-index batch loop does not pay a virtual call
// per neighbor. Several accumulators keep the kernel on FMA throughput.

#include <cstdint>
#include <immintrin.h>

namespace diskann {

#ifdef USE_AVX2
inline float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}

#if defined(__AVX512F__)
inline float hsum512_ps(__m512 v)
{
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
    return hsum256_ps(_mm256_add_ps(lo, hi));
}

inline float l2sqr_f32_avx512(const float *a, const float *b, uint32_t size)
{
    uint32_t i = 0;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    for (; i + 64 <= size; i += 64)
    {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32));
        __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48));
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2);
        acc3 = _mm512_fmadd_ps(d3, d3, acc3);
    }
    for (; i + 16 <= size; i += 16)
    {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        acc0 = _mm512_fmadd_ps(d, d, acc0);
    }
    float result = hsum512_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
    for (; i < size; ++i)
    {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}
#endif

inline float l2sqr_f32_avx2(const float *a, const float *b, uint32_t size)
{
    uint32_t i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i + 32 <= size; i += 32)
    {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    for (; i + 8 <= size; i += 8)
    {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }
    float result = hsum256_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; i < size; ++i)
    {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}
#endif

inline float l2sqr_f32(const float *a, const float *b, uint32_t size)
{
#ifdef USE_AVX2
#if defined(__AVX512F__)
    return l2sqr_f32_avx512(a, b, size);
#else
    return l2sqr_f32_avx2(a, b, size);
#endif
#else
    float result = 0.f;
    for (uint32_t i = 0; i < size; ++i)
    {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
#endif
}

} // namespace diskann

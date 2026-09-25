
#ifndef AKER_DISTANCEFUNCS_H
#define AKER_DISTANCEFUNCS_H


#include <stddef.h>

// Needed for std::sqrt() used by cosine distance implementations.
#include <cmath>

#ifdef __AVX512F__
#include <immintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif
#ifdef __SSE__
#include <xmmintrin.h>
#endif

// -------------------------------------------- L2 Distance Functions --------------------------------------------
/**
 * Plain C fallback: no SIMD, squared L2 distance
 */
static inline float l2_dist_plain(const float *a, const float *b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#ifdef __SSE__
/**
 * SSE version (128-bit wide, processes 4 floats at a time), squared L2 distance
 */
static inline float l2_dist_sse(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m128 acc = _mm_setzero_ps();
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
    }
    float buf[4];
    _mm_storeu_ps(buf, acc);
    float sum = buf[0] + buf[1] + buf[2] + buf[3];
    for (; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
#endif

#ifdef __AVX__
/**
 * AVX version (256-bit wide, processes 8 floats at a time), squared L2 distance
 */
static inline float l2_dist_avx(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
    }
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    float sum = buf[0] + buf[1] + buf[2] + buf[3]
              + buf[4] + buf[5] + buf[6] + buf[7];
    for (; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
#endif

#ifdef __AVX2__
/**
 * AVX2 + FMA version (256-bit with fused multiply-add), squared L2 distance
 */
static inline float l2_dist_avx2(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    float sum = buf[0] + buf[1] + buf[2] + buf[3]
              + buf[4] + buf[5] + buf[6] + buf[7];
    for (; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
#endif

#ifdef __AVX512F__
/**
 * AVX-512 version (512-bit wide, processes 16 floats at a time), squared L2 distance
 */
static inline float l2_dist_avx512(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m512 acc = _mm512_setzero_ps();
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(diff, diff));
    }
    float buf[16];
    _mm512_storeu_ps(buf, acc);
    float sum = 0.0f;
    for (int j = 0; j < 16; ++j) sum += buf[j];
    for (; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
#endif

/**
 * Runtime dispatcher: chooses best available implementation, returns squared L2 distance
 * Refer to SPTAG implementation for guidance 
 */
static inline float l2_dist(const float *a, const float *b, size_t n) {
#ifdef __AVX512F__
    return l2_dist_avx512(a, b, n);
#elif defined(__AVX2__)
    return l2_dist_avx2(a, b, n);
#elif defined(__AVX__)
    return l2_dist_avx(a, b, n);
#elif defined(__SSE__)
    return l2_dist_sse(a, b, n);
#else
    return l2_dist_plain(a, b, n);
#endif
}

// -------------------------------------------- Inner Product Distance Functions --------------------------------------------

/**
 * Plain C fallback: no SIMD, inner product distance with a constant
 */
static inline float inner_product_plain(const float *a, const float *b, size_t n) {
    double sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return (float)(-sum);
}

#ifdef __SSE__
static inline float inner_product_sse(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m128 acc = _mm_setzero_ps();
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }
    float buf[4];
    _mm_storeu_ps(buf, acc);
    float sum = buf[0] + buf[1] + buf[2] + buf[3];
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return (float)(-sum);
}
#endif

#ifdef __AVX__
static inline float ip_dist_avx(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    double sum = 0;
    for (int j = 0; j < 8; ++j) sum += buf[j];
    for (; i < n; ++i) sum += (double)a[i] * (double)b[i];
    return (float)(-sum);
}
#endif

#ifdef __AVX2__
static inline float ip_dist_avx2(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    double sum = 0;
    for (int j = 0; j < 8; ++j) sum += buf[j];
    for (; i < n; ++i) sum += (double)a[i] * (double)b[i];
    return (float)(-sum);
}
#endif

#ifdef __AVX512F__
static inline float ip_dist_avx512(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m512 acc = _mm512_setzero_ps();
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(va, vb));
    }
    float buf[16];
    _mm512_storeu_ps(buf, acc);
    double sum = 0;
    for (int j = 0; j < 16; ++j) sum += buf[j];
    for (; i < n; ++i) sum += (double)a[i] * (double)b[i];
    return (float)(-sum);
}
#endif

/**
 * Runtime dispatcher for inner product distance
 * Refer to SPTAG implementation for guidance 
 */
static inline float inner_product_dist(const float *a, const float *b, size_t n) {
#ifdef __AVX512F__
    return ip_dist_avx512(a, b, n);
#elif defined(__AVX2__)
    return ip_dist_avx2(a, b, n);
#elif defined(__AVX__)
    return ip_dist_avx(a, b, n);
#elif defined(__SSE__)
    return inner_product_sse(a, b, n);
#else
    return inner_product_plain(a, b, n);
#endif
}

// -------------------------------------------- Cosine Distance Functions with Constant --------------------------------------------

/**
 * Plain C fallback: no SIMD, cosine distance with a constant
 * Computes 1 - (a . b) / (||a|| * ||b||)
 */
static inline float cosine_dist_plain(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double va = a[i], vb = b[i];
        dot += va * vb;
        na  += va * va;
        nb  += vb * vb;
    }
    return 1.0f - (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}

#ifdef __SSE__
static inline float cosine_dist_sse(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m128 accDot = _mm_setzero_ps();
    __m128 accAA  = _mm_setzero_ps();
    __m128 accBB  = _mm_setzero_ps();
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        accDot = _mm_add_ps(accDot, _mm_mul_ps(va, vb));
        accAA  = _mm_add_ps(accAA, _mm_mul_ps(va, va));
        accBB  = _mm_add_ps(accBB, _mm_mul_ps(vb, vb));
    }
    float dotBuf[4], aaBuf[4], bbBuf[4];
    _mm_storeu_ps(dotBuf, accDot);
    _mm_storeu_ps(aaBuf,  accAA);
    _mm_storeu_ps(bbBuf,  accBB);
    double dot = 0, na = 0, nb = 0;
    for (int j = 0; j < 4; ++j) {
        dot += dotBuf[j]; na += aaBuf[j]; nb += bbBuf[j];
    }
    for (; i < n; ++i) {
        double va = a[i], vb = b[i];
        dot += va * vb; na += va * va; nb += vb * vb;
    }
    return 1.0f - (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}
#endif

#ifdef __AVX__
static inline float cosine_dist_avx(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m256 accDot = _mm256_setzero_ps();
    __m256 accAA  = _mm256_setzero_ps();
    __m256 accBB  = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        accDot = _mm256_add_ps(accDot, _mm256_mul_ps(va, vb));
        accAA  = _mm256_add_ps(accAA, _mm256_mul_ps(va, va));
        accBB  = _mm256_add_ps(accBB, _mm256_mul_ps(vb, vb));
    }
    float dotBuf[8], aaBuf[8], bbBuf[8];
    _mm256_storeu_ps(dotBuf, accDot);
    _mm256_storeu_ps(aaBuf,  accAA);
    _mm256_storeu_ps(bbBuf,  accBB);
    double dot = 0, na = 0, nb = 0;
    for (int j = 0; j < 8; ++j) {
        dot += dotBuf[j]; na += aaBuf[j]; nb += bbBuf[j];
    }
    for (; i < n; ++i) {
        double va = a[i], vb = b[i];
        dot += va * vb; na += va * va; nb += vb * vb;
    }
    return 1.0f - (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}
#endif

#ifdef __AVX2__
static inline float cosine_dist_avx2(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m256 accDot = _mm256_setzero_ps();
    __m256 accAA  = _mm256_setzero_ps();
    __m256 accBB  = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        accDot = _mm256_fmadd_ps(va, vb, accDot);
        accAA  = _mm256_fmadd_ps(va, va, accAA);
        accBB  = _mm256_fmadd_ps(vb, vb, accBB);
    }
    float dotBuf[8], aaBuf[8], bbBuf[8];
    _mm256_storeu_ps(dotBuf, accDot);
    _mm256_storeu_ps(aaBuf,  accAA);
    _mm256_storeu_ps(bbBuf,  accBB);
    double dot = 0, na = 0, nb = 0;
    for (int j = 0; j < 8; ++j) {
        dot += dotBuf[j]; na += aaBuf[j]; nb += bbBuf[j];
    }
    for (; i < n; ++i) {
        double va = a[i], vb = b[i];
        dot += va * vb; na += va * va; nb += vb * vb;
    }
    return 1.0f - (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}
#endif

#ifdef __AVX512F__
static inline float cosine_dist_avx512(const float *a, const float *b, size_t n) {
    size_t i = 0;
    __m512 accDot = _mm512_setzero_ps();
    __m512 accAA  = _mm512_setzero_ps();
    __m512 accBB  = _mm512_setzero_ps();
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        accDot = _mm512_fmadd_ps(va, vb, accDot);
        accAA  = _mm512_fmadd_ps(va, va, accAA);
        accBB  = _mm512_fmadd_ps(vb, vb, accBB);
    }
    float dotBuf[16], aaBuf[16], bbBuf[16];
    _mm512_storeu_ps(dotBuf, accDot);
    _mm512_storeu_ps(aaBuf,  accAA);
    _mm512_storeu_ps(bbBuf,  accBB);
    double dot = 0, na = 0, nb = 0;
    for (int j = 0; j < 16; ++j) {
        dot += dotBuf[j]; na += aaBuf[j]; nb += bbBuf[j];
    }
    for (; i < n; ++i) {
        double va = a[i], vb = b[i];
        dot += va * vb; na += va * va; nb += vb * vb;
    }
    return 1.0f - (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}
#endif

/**
 * Runtime dispatcher for cosine distance with a constant
 * Refer to SPTAG implementation for guidance 
 */
static inline float cosine_dist(const float *a, const float *b, size_t n) {
#ifdef __AVX512F__
    return cosine_dist_avx512(a, b, n);
#elif defined(__AVX2__)
    return cosine_dist_avx2(a, b, n);
#elif defined(__AVX__)
    return cosine_dist_avx(a, b, n);
#elif defined(__SSE__)
    return cosine_dist_sse(a, b, n);
#else
    return cosine_dist_plain(a, b, n);
#endif
}

#endif /* AKER_DISTANCEFUNCS_H */

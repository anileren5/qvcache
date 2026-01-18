#ifdef _WINDOWS
#include <immintrin.h>
#include <intrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#else
#include <immintrin.h>
#endif

#include "simd_utils.h"
#include <cosine_similarity.h>
#include <iostream>

#include "distance.h"

namespace greator {

// Cosine distance (matching DiskANN implementation exactly).
float DistanceCosineInt8::compare(const int8_t *a, const int8_t *b,
                                  uint32_t length) const {
#ifdef _WINDOWS
  return diskann::CosineSimilarity2<int8_t>(a, b, length);
#else
  // Match DiskANN's Linux implementation exactly
  int magA = 0, magB = 0, scalarProduct = 0;
  for (uint32_t i = 0; i < length; i++)
  {
      magA += ((int32_t)a[i]) * ((int32_t)a[i]);
      magB += ((int32_t)b[i]) * ((int32_t)b[i]);
      scalarProduct += ((int32_t)a[i]) * ((int32_t)b[i]);
  }
  // similarity == 1-cosine distance
  return 1.0f - (float)(scalarProduct / (sqrt(magA) * sqrt(magB)));
#endif
}

float DistanceCosineFloat::compare(const float *a, const float *b,
                                   uint32_t length) const {
#ifdef _WINDOWS
  return diskann::CosineSimilarity2<float>(a, b, length);
#else
  // Match DiskANN's Linux implementation exactly
  float magA = 0, magB = 0, scalarProduct = 0;
  for (uint32_t i = 0; i < length; i++)
  {
      magA += (a[i]) * (a[i]);
      magB += (b[i]) * (b[i]);
      scalarProduct += (a[i]) * (b[i]);
  }
  // similarity == 1-cosine distance
  return 1.0f - (scalarProduct / (sqrt(magA) * sqrt(magB)));
#endif
}

float SlowDistanceCosineUInt8::compare(const uint8_t *a, const uint8_t *b,
                                       uint32_t length) const {
  int magA = 0, magB = 0, scalarProduct = 0;
  for (uint32_t i = 0; i < length; i++) {
    magA += ((uint32_t)a[i]) * ((uint32_t)a[i]);
    magB += ((uint32_t)b[i]) * ((uint32_t)b[i]);
    scalarProduct += ((uint32_t)a[i]) * ((uint32_t)b[i]);
  }
  // similarity == 1-cosine distance
  return 1.0f - (float)(scalarProduct / (sqrt(magA) * sqrt(magB)));
}

// L2 distance functions.
float DistanceL2Int8::compare(const int8_t *a, const int8_t *b,
                              uint32_t size) const {
  int32_t result = 0;

#ifdef _WINDOWS
#ifdef USE_AVX2
  __m256 r = _mm256_setzero_ps();
  char *pX = (char *)a, *pY = (char *)b;
  while (size >= 32) {
    __m256i r1 = _mm256_subs_epi8(_mm256_loadu_si256((__m256i *)pX),
                                  _mm256_loadu_si256((__m256i *)pY));
    r = _mm256_add_ps(r, _mm256_mul_epi8(r1, r1));
    pX += 32;
    pY += 32;
    size -= 32;
  }
  while (size > 0) {
    __m128i r2 = _mm_subs_epi8(_mm_loadu_si128((__m128i *)pX),
                               _mm_loadu_si128((__m128i *)pY));
    r = _mm256_add_ps(r, _mm256_mul32_pi8(r2, r2));
    pX += 4;
    pY += 4;
    size -= 4;
  }
  r = _mm256_hadd_ps(_mm256_hadd_ps(r, r), r);
  return r.m256_f32[0] + r.m256_f32[4];
#else
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
  for (int32_t i = 0; i < (int32_t)size; i++) {
    result += ((int32_t)((int16_t)a[i] - (int16_t)b[i])) *
              ((int32_t)((int16_t)a[i] - (int16_t)b[i]));
  }
  return (float)result;
#endif
#else
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
  for (int32_t i = 0; i < (int32_t)size; i++) {
    result += ((int32_t)((int16_t)a[i] - (int16_t)b[i])) *
              ((int32_t)((int16_t)a[i] - (int16_t)b[i]));
  }
  return (float)result;
#endif
}

float DistanceL2UInt8::compare(const uint8_t *a, const uint8_t *b,
                               uint32_t size) const {
  uint32_t result = 0;
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
#endif
  for (int32_t i = 0; i < (int32_t)size; i++) {
    result += ((int32_t)((int16_t)a[i] - (int16_t)b[i])) *
              ((int32_t)((int16_t)a[i] - (int16_t)b[i]));
  }
  return (float)result;
}

#ifndef _WINDOWS
float DistanceL2::compare(const float *a, const float *b, uint32_t size) const {
  a = (const float *)__builtin_assume_aligned(a, 32);
  b = (const float *)__builtin_assume_aligned(b, 32);
#else
float DistanceL2::compare(const float *a, const float *b, uint32_t size) const {
#endif

  float result = 0;
#ifdef USE_AVX2
  // assume size is divisible by 8
  uint16_t niters = (uint16_t)(size / 8);
  __m256 sum = _mm256_setzero_ps();
  for (uint16_t j = 0; j < niters; j++) {
    // scope is a[8j:8j+7], b[8j:8j+7]
    // load a_vec
    if (j < (niters - 1)) {
      _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
      _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
    }
    __m256 a_vec = _mm256_load_ps(a + 8 * j);
    // load b_vec
    __m256 b_vec = _mm256_load_ps(b + 8 * j);
    // a_vec - b_vec
    __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);
    /*
// (a_vec - b_vec)**2
    __m256 tmp_vec2 = _mm256_mul_ps(tmp_vec, tmp_vec);
// accumulate sum
    sum = _mm256_add_ps(sum, tmp_vec2);
*/
    // sum = (tmp_vec**2) + sum
    sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
  }

  // horizontal add sum
  result = _mm256_reduce_add_ps(sum);
#else
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result) aligned(a, b : 32)
#endif
  for (int32_t i = 0; i < (int32_t)size; i++) {
    result += (a[i] - b[i]) * (a[i] - b[i]);
  }
#endif
  return result;
}

float SlowDistanceL2Float::compare(const float *a, const float *b,
                                   uint32_t length) const {
  float result = 0.0f;
  for (uint32_t i = 0; i < length; i++) {
    result += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return result;
}

#ifdef _WINDOWS
float AVXDistanceL2Int8::compare(const int8_t *a, const int8_t *b,
                                 uint32_t length) const {
  __m128 r = _mm_setzero_ps();
  __m128i r1;
  while (length >= 16) {
    r1 = _mm_subs_epi8(_mm_load_si128((__m128i *)a),
                       _mm_load_si128((__m128i *)b));
    r = _mm_add_ps(r, _mm_mul_epi8(r1));
    a += 16;
    b += 16;
    length -= 16;
  }
  r = _mm_hadd_ps(_mm_hadd_ps(r, r), r);
  float res = r.m128_f32[0];

  if (length >= 8) {
    __m128 r2 = _mm_setzero_ps();
    __m128i r3 = _mm_subs_epi8(_mm_load_si128((__m128i *)(a - 8)),
                               _mm_load_si128((__m128i *)(b - 8)));
    r2 = _mm_add_ps(r2, _mm_mulhi_epi8(r3));
    a += 8;
    b += 8;
    length -= 8;
    r2 = _mm_hadd_ps(_mm_hadd_ps(r2, r2), r2);
    res += r2.m128_f32[0];
  }

  if (length >= 4) {
    __m128 r2 = _mm_setzero_ps();
    __m128i r3 = _mm_subs_epi8(_mm_load_si128((__m128i *)(a - 12)),
                               _mm_load_si128((__m128i *)(b - 12)));
    r2 = _mm_add_ps(r2, _mm_mulhi_epi8_shift32(r3));
    res += r2.m128_f32[0] + r2.m128_f32[1];
  }

  return res;
}

float AVXDistanceL2Float::compare(const float *a, const float *b,
                                  uint32_t length) const {
  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (length >= 4) {
    v1 = _mm_loadu_ps(a);
    a += 4;
    v2 = _mm_loadu_ps(b);
    b += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    length -= 4;
  }

  return sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] + sum.m128_f32[3];
}
#else
float AVXDistanceL2Int8::compare(const int8_t *, const int8_t *,
                                 uint32_t) const {
  return 0;
}
float AVXDistanceL2Float::compare(const float *, const float *,
                                  uint32_t) const {
  return 0;
}
#endif

// Inner product distance implementations
// Slow implementation matches DiskANN's scalar fallback exactly
float SlowDistanceInnerProductFloat::compare(const float *a, const float *b,
                                             uint32_t size) const {
  float result = 0.0f;
  float dot0, dot1, dot2, dot3;
  const float *last = a + size;
  const float *unroll_group = last - 3;

  /* Process 4 items with each loop for efficiency (matching DiskANN exactly). */
  while (a < unroll_group) {
    dot0 = a[0] * b[0];
    dot1 = a[1] * b[1];
    dot2 = a[2] * b[2];
    dot3 = a[3] * b[3];
    result += dot0 + dot1 + dot2 + dot3;
    a += 4;
    b += 4;
  }
  /* Process last 0-3 pixels. Not needed for standard vector lengths. */
  while (a < last) {
    result += *a++ * *b++;
  }
  // Return negative inner product (for distance, smaller is better)
  // This matches DiskANN's AVXDistanceInnerProductFloat::compare()
  return -result;
}

float AVXDistanceInnerProductFloat::compare(const float *a, const float *b,
                                            uint32_t size) const {
  // This implementation matches DiskANN's AVXDistanceInnerProductFloat::compare() exactly
  float result = 0.0f;
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2)                                                                        \
    tmp1 = _mm256_loadu_ps(addr1);                                                                                     \
    tmp2 = _mm256_loadu_ps(addr2);                                                                                     \
    tmp1 = _mm256_mul_ps(tmp1, tmp2);                                                                                  \
    dest = _mm256_add_ps(dest, tmp1);

  __m256 sum;
  __m256 l0, l1;
  __m256 r0, r1;
  uint32_t D = (size + 7) & ~7U;
  uint32_t DR = D % 16;
  uint32_t DD = D - DR;
  const float *l = (float *)a;
  const float *r = (float *)b;
  const float *e_l = l + DD;
  const float *e_r = r + DD;
#ifndef _WINDOWS
  float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
#else
  __declspec(align(32)) float unpack[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#endif

  sum = _mm256_loadu_ps(unpack);
  if (DR) {
    AVX_DOT(e_l, e_r, sum, l0, r0);
  }

  for (uint32_t i = 0; i < DD; i += 16, l += 16, r += 16) {
    AVX_DOT(l, r, sum, l0, r0);
    AVX_DOT(l + 8, r + 8, sum, l1, r1);
  }
  _mm256_storeu_ps(unpack, sum);
  result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];

  return -result;
}
} // namespace greator

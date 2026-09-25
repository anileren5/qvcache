#pragma once
// Aker: C ABI helpers. Aker's transform_callback_t passes dim as uint32_t into
// a C function that takes size_t; a garbage high half makes dim >> src_size and
// the old `src_size < dim` check skipped conversion, so FAISS indexed noise.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <omp.h>

#include "ak_anns_cache_c_wrapper.h"

// FAISS (Aker query filter) sizes OpenMP teams from omp_get_max_threads() and
// searches nq=1. The harness query loop is already `#pragma omp parallel for`,
// so each lookup is a nested team. DiskANN/Greator warms that pool at index
// load (~1 ms); pgvector never loads Greator, so the same fork is ~10 ms.
// Disable nested teams and pin the default width to the search thread count.
inline void aker_pin_openmp(int nthreads) {
    if (nthreads < 1)
        nthreads = 1;
    omp_set_max_active_levels(1);
    omp_set_nested(0);
    omp_set_num_threads(nthreads);
}

inline std::atomic<unsigned> aker_xf_calls{0};
inline std::atomic<unsigned> aker_xf_ok{0};
inline std::atomic<size_t> aker_xf_last_dim{0};
inline std::atomic<size_t> aker_xf_last_src{0};
inline std::atomic<unsigned> aker_xf_last_n{0};

// Aker: C ABI transform is stored in std::function<..., uint32_t dim, ...> but the
// function pointer takes size_t dim. On x86-64 the high 32 bits of that register
// are undefined, so `dim` can look huge and `src_size < dim` used to no-op.
inline size_t aker_sane_elem_count(size_t src_elems, size_t dim) {
    const size_t dim32 = static_cast<uint32_t>(dim);
    if (dim32 > 0 && dim32 <= src_elems)
        return dim32;
    if (dim > 0 && dim <= src_elems)
        return dim;
    return src_elems;
}

inline void aker_note_transform(size_t src_size, size_t dim, size_t n, bool ok) {
    aker_xf_calls.fetch_add(1, std::memory_order_relaxed);
    if (ok)
        aker_xf_ok.fetch_add(1, std::memory_order_relaxed);
    aker_xf_last_dim.store(dim, std::memory_order_relaxed);
    aker_xf_last_src.store(src_size, std::memory_order_relaxed);
    aker_xf_last_n.store(static_cast<unsigned>(n), std::memory_order_relaxed);
}

inline bool aker_transform_float(void* src, size_t src_size, size_t dim, void* dst, uint8_t* aux) {
    (void)aux;
    if (src == nullptr || dst == nullptr || src_size < sizeof(float)) {
        aker_note_transform(src_size, dim, 0, false);
        return false;
    }
    const size_t n = aker_sane_elem_count(src_size / sizeof(float), dim);
    std::memcpy(dst, src, n * sizeof(float));
    aker_note_transform(src_size, dim, n, true);
    return true;
}

inline bool aker_transform_int8(void* src, size_t src_size, size_t dim, void* dst, uint8_t* aux) {
    (void)aux;
    if (src == nullptr || dst == nullptr || src_size == 0) {
        aker_note_transform(src_size, dim, 0, false);
        return false;
    }
    const size_t n = aker_sane_elem_count(src_size, dim);
    const auto* in = static_cast<const int8_t*>(src);
    auto* out = static_cast<float*>(dst);
    for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<float>(in[i]);
    aker_note_transform(src_size, dim, n, true);
    return true;
}

inline bool aker_transform_uint8(void* src, size_t src_size, size_t dim, void* dst, uint8_t* aux) {
    (void)aux;
    if (src == nullptr || dst == nullptr || src_size == 0) {
        aker_note_transform(src_size, dim, 0, false);
        return false;
    }
    const size_t n = aker_sane_elem_count(src_size, dim);
    const auto* in = static_cast<const uint8_t*>(src);
    auto* out = static_cast<float*>(dst);
    for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<float>(in[i]);
    aker_note_transform(src_size, dim, n, true);
    return true;
}

inline float aker_l2_float_bytes(uint8_t* a, uint8_t* b, size_t dim) {
    dim = static_cast<uint32_t>(dim);
    return akerL2Distance(reinterpret_cast<float*>(a), reinterpret_cast<float*>(b), dim);
}

inline float aker_ip_float_bytes(uint8_t* a, uint8_t* b, size_t dim) {
    return akerInnerProductDistance(reinterpret_cast<float*>(a), reinterpret_cast<float*>(b), dim);
}

inline float aker_l2_int8_bytes(uint8_t* a, uint8_t* b, size_t dim) {
    dim = static_cast<uint32_t>(dim);
    const auto* x = reinterpret_cast<const int8_t*>(a);
    const auto* y = reinterpret_cast<const int8_t*>(b);
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float d = static_cast<float>(x[i]) - static_cast<float>(y[i]);
        sum += d * d;
    }
    return sum;
}

inline float aker_l2_uint8_bytes(uint8_t* a, uint8_t* b, size_t dim) {
    dim = static_cast<uint32_t>(dim);
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        sum += d * d;
    }
    return sum;
}

// Aker: find a C-ABI sequence that yields similar=1. Does not edit Aker sources.
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "ak_anns_cache_c_wrapper.h"

static bool xf_memcpy(void* src, size_t src_size, size_t dim, void* dst, uint8_t* aux) {
    (void)aux;
    (void)dim;
    if (src == nullptr || dst == nullptr || src_size < sizeof(float))
        return false;
    std::memcpy(dst, src, src_size);
    return true;
}

static float dist_l2(uint8_t* a, uint8_t* b, size_t dim) {
    return akerL2Distance(reinterpret_cast<float*>(a), reinterpret_cast<float*>(b), dim);
}

static float dist_zero(uint8_t*, uint8_t*, size_t) { return 0.0f; }

static anns_cache_c_wrapper_t* make_cache(uint32_t dim, size_t vinb, uint32_t k, uint32_t td) {
    anns_cache_parameter_c_t p{};
    akerImportAnnsCacheConfig(const_cast<char*>("external/Aker/bootstrap/aker-standard.ini"), &p);
    p.vector_format.dimension = dim;
    p.vector_format.vector_in_bytes = vinb;
    p.capacity.in_topk = k;
    p.capacity.top_delta = td;
    p.capacity.pool_size = 4096;
    p.distance_metric = 0;
    return akerCreateAnnsCache(p);
}

static std::vector<char*> make_neighbors(uint32_t n, size_t vinb, float min_d) {
    std::vector<char*> slots(n, nullptr);
    std::vector<float> z(vinb / sizeof(float), 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        const float d = (i == 0) ? min_d : (min_d + static_cast<float>(i));
        slots[i] = akerCreateVectorSlot(10000 + i, vinb, reinterpret_cast<char*>(z.data()), 0, 0, d);
    }
    return slots;
}

static void free_neighbors(std::vector<char*>& slots) {
    for (char* s : slots)
        akerDestroyVectorSlot(s);
}

struct Hit {
    bool hit = false;
    bool similar = false;
    bool invalid = false;
};

static Hit lookup(anns_cache_c_wrapper_t* cache, float* q, uint32_t dim, size_t vinb, uint64_t id,
                  float (*dist)(uint8_t*, uint8_t*, size_t)) {
    char* slot = akerCreateVectorSlot(id, vinb, reinterpret_cast<char*>(q), 0, 0, 0.0f);
    char* view = akerCreateVectorView(slot, dim, vinb, xf_memcpy);
    Hit h;
    char* e = akerGetCacheEntry(cache, view, &h.similar, &h.invalid, dist);
    h.hit = (e != nullptr && !h.invalid);
    if (e)
        akerDestroyCacheEntry(e);
    akerDestroyVectorView(view);
    akerDestroyVectorSlot(slot);
    return h;
}

static void report(const char* name, Hit h) {
    std::cout << name << " hit=" << h.hit << " similar=" << h.similar << " invalid=" << h.invalid << "\n";
}

int main() {
    const uint32_t dim = 8;
    const uint32_t k = 2;
    const uint32_t td = 2;
    const uint32_t nnb = k + td;
    const size_t vinb = static_cast<size_t>(dim) * sizeof(float);
    std::vector<float> q(dim, 1.0f);

    {
        auto* cache = make_cache(dim, vinb, k, td);
        auto nbs = make_neighbors(nnb, vinb, 4.0f); // thresh = 1.0
        char* qs = akerCreateVectorSlot(42, vinb, reinterpret_cast<char*>(q.data()), 0, 0, 0.0f);
        char* qv = akerCreateVectorView(qs, dim, vinb, xf_memcpy);
        char* ent = akerCreateCacheEntry(cache, qs, nnb, nbs.data());
        if (!akerInsertCacheEntry(cache, 42, ent, qv)) {
            std::cerr << "insert failed\n";
            return 1;
        }
        akerDestroyVectorView(qv);
        akerDestroyVectorSlot(qs);
        for (int i = 0; i < 15; ++i) {
            const uint64_t id = 1000 + i;
            char* s = akerCreateVectorSlot(id, vinb, reinterpret_cast<char*>(q.data()), 0, 0, 0.0f);
            char* v = akerCreateVectorView(s, dim, vinb, xf_memcpy);
            char* e = akerCreateCacheEntry(cache, s, nnb, nbs.data());
            if (e && !akerInsertCacheEntry(cache, id, e, v))
                akerDestroyCacheEntry(e);
            akerDestroyVectorView(v);
            akerDestroyVectorSlot(s);
        }
        report("exact42/l2", lookup(cache, q.data(), dim, vinb, 42, dist_l2));
        report("same99/l2", lookup(cache, q.data(), dim, vinb, 99, dist_l2));
        report("same99/zero-dist", lookup(cache, q.data(), dim, vinb, 99, dist_zero));
        free_neighbors(nbs);
        akerDestroyAnnsCache(cache);
    }

    {
        auto* cache = make_cache(dim, vinb, k, td);
        auto nbs = make_neighbors(nnb, vinb, 4.0f);
        char* qs = akerCreateVectorSlot(42, vinb, reinterpret_cast<char*>(q.data()), 0, 0, 0.0f);
        char* qv = akerCreateVectorView(qs, dim, vinb, xf_memcpy);
        char* ent = akerCreateCacheEntry(cache, qs, nnb, nbs.data());
        akerInsertCacheEntry(cache, 42, ent, qv);
        akerDestroyVectorView(qv);
        akerDestroyVectorSlot(qs);

        char* ls = akerCreateVectorSlot(99, vinb, reinterpret_cast<char*>(q.data()), 0, 0, 0.0f);
        char* le = akerCreateCacheEntry(cache, ls, nnb, nbs.data());
        const bool linked = akerLinkCacheEntry(cache, le, 42);
        std::cout << "link99->42 ok=" << linked << "\n";
        if (!linked && le)
            akerDestroyCacheEntry(le);
        akerDestroyVectorSlot(ls);
        report("linked99", lookup(cache, q.data(), dim, vinb, 99, dist_l2));
        free_neighbors(nbs);
        akerDestroyAnnsCache(cache);
    }

    {
        auto* cache = make_cache(dim, vinb, k, td);
        auto nbs = make_neighbors(nnb, vinb, 4.0f);
        std::vector<float> first;
        for (int i = 0; i < 64; ++i) {
            std::vector<float> v(dim);
            for (uint32_t d = 0; d < dim; ++d)
                v[d] = static_cast<float>(i + 1) * 0.1f + static_cast<float>(d);
            if (i == 0)
                first = v;
            const uint64_t id = static_cast<uint64_t>(i + 1);
            char* s = akerCreateVectorSlot(id, vinb, reinterpret_cast<char*>(v.data()), 0, 0, 0.0f);
            char* vw = akerCreateVectorView(s, dim, vinb, xf_memcpy);
            char* e = akerCreateCacheEntry(cache, s, nnb, nbs.data());
            if (e && !akerInsertCacheEntry(cache, id, e, vw))
                akerDestroyCacheEntry(e);
            akerDestroyVectorView(vw);
            akerDestroyVectorSlot(s);
        }
        report("diverse64/copy-id99/l2", lookup(cache, first.data(), dim, vinb, 99, dist_l2));
        report("diverse64/copy-id99/zero", lookup(cache, first.data(), dim, vinb, 99, dist_zero));
        free_neighbors(nbs);
        akerDestroyAnnsCache(cache);
    }
    return 0;
}

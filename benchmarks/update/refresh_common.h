#pragma once

// Aker §5.4 refresh stress-test helpers, scaled to SPACEV-1M.
// Protocol: warmup simZipf (search only, no GT) → insert 5% → delete D%
// → exact top-k on the live set (base + inserts − deletes) → re-search.
// Search-only GT files are left alone. The live re-search top-k is
// written to a separate sidecar path.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <limits>
#include <map>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <spdlog/spdlog.h>

namespace qvcache_refresh {

struct Neighbor {
    uint32_t id = 0;
    float dist = std::numeric_limits<float>::max();
};

inline bool neighbor_less(const Neighbor& a, const Neighbor& b) {
    if (a.dist != b.dist) {
        return a.dist < b.dist;
    }
    return a.id < b.id;
}

template <typename T>
inline float l2sq(const T* a, const T* b, size_t dim) {
    float s = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        const float diff = static_cast<float>(a[d]) - static_cast<float>(b[d]);
        s += diff * diff;
    }
    return s;
}

template <typename T>
inline T lerp_comp(T a, T b, float lam) {
    const float v = (1.0f - lam) * static_cast<float>(a) + lam * static_cast<float>(b);
    if constexpr (std::is_same_v<T, int8_t>) {
        return static_cast<T>(std::max(-128.0f, std::min(127.0f, std::round(v))));
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return static_cast<T>(std::max(0.0f, std::min(255.0f, std::round(v))));
    } else {
        return static_cast<T>(v);
    }
}

template <typename T>
struct LiveVectors {
    const T* base = nullptr;
    size_t n_base = 0;
    size_t dim = 0;
    size_t aligned = 0;
    std::vector<uint32_t> extra_ids;
    std::vector<T> extra;
    std::unordered_map<uint32_t, size_t> extra_slot;
    std::unordered_set<uint32_t> deleted;

    const T* ptr(uint32_t id) const {
        if (deleted.count(id) != 0) {
            return nullptr;
        }
        if (id < n_base) {
            return base + static_cast<size_t>(id) * aligned;
        }
        auto it = extra_slot.find(id);
        if (it == extra_slot.end()) {
            return nullptr;
        }
        return extra.data() + it->second * aligned;
    }

    void add_extra(uint32_t id, const T* vec) {
        extra_slot[id] = extra_ids.size();
        extra_ids.push_back(id);
        extra.insert(extra.end(), vec, vec + aligned);
    }

    void mark_deleted(uint32_t id) {
        deleted.insert(id);
    }

    bool is_live(uint32_t id) const {
        return deleted.count(id) == 0 && ptr(id) != nullptr;
    }
};

struct LiveGroundtruth {
    uint32_t K = 10;
    std::vector<std::vector<Neighbor>> lists;

    std::unordered_set<uint32_t> topk(size_t q) const {
        std::unordered_set<uint32_t> s;
        if (q >= lists.size()) {
            return s;
        }
        const auto& lst = lists[q];
        const size_t n = std::min(static_cast<size_t>(K), lst.size());
        for (size_t j = 0; j < n; ++j) {
            s.insert(lst[j].id);
        }
        return s;
    }
};

template <typename T>
void pack_live(const LiveVectors<T>& live, std::vector<uint32_t>& ids, std::vector<T>& packed) {
    ids.clear();
    packed.clear();
    const size_t cap = live.n_base + live.extra_ids.size();
    ids.reserve(cap);
    packed.reserve(cap * live.dim);
    for (uint32_t id = 0; id < static_cast<uint32_t>(live.n_base); ++id) {
        const T* v = live.ptr(id);
        if (v == nullptr) {
            continue;
        }
        ids.push_back(id);
        packed.insert(packed.end(), v, v + live.dim);
    }
    for (uint32_t id : live.extra_ids) {
        const T* v = live.ptr(id);
        if (v == nullptr) {
            continue;
        }
        ids.push_back(id);
        packed.insert(packed.end(), v, v + live.dim);
    }
}

// Exact top-K of n_eval queries against the live collection.
template <typename T>
LiveGroundtruth compute_exact_gt(const T* queries, size_t qaligned, size_t n_eval,
                                 const LiveVectors<T>& live, uint32_t K) {
    std::vector<uint32_t> ids;
    std::vector<T> packed;
    pack_live(live, ids, packed);
    const size_t n_live = ids.size();
    const size_t dim = live.dim;

    LiveGroundtruth gt;
    gt.K = K;
    gt.lists.assign(n_eval, {});
    std::atomic<size_t> done{0};

#pragma omp parallel for schedule(dynamic, 1)
    for (int q = 0; q < static_cast<int>(n_eval); ++q) {
        const T* query = queries + static_cast<size_t>(q) * qaligned;
        std::vector<Neighbor> best;
        best.reserve(K);
        for (size_t i = 0; i < n_live; ++i) {
            Neighbor nb{ids[i], l2sq(query, packed.data() + i * dim, dim)};
            if (best.size() < K) {
                best.push_back(nb);
                if (best.size() == K) {
                    std::sort(best.begin(), best.end(), neighbor_less);
                }
            } else if (nb.dist < best.back().dist) {
                best.back() = nb;
                std::sort(best.begin(), best.end(), neighbor_less);
            }
        }
        gt.lists[static_cast<size_t>(q)] = std::move(best);
        const size_t n = done.fetch_add(1) + 1;
        if (n % 100 == 0 || n == n_eval) {
            spdlog::info("{{\"event\": \"refresh_gt_progress\", \"completed\": {}, "
                         "\"total\": {}, \"n_live\": {}}}",
                         n, n_eval, n_live);
        }
    }
    return gt;
}

// DiskANN truthset: npts, dim, then npts*K ids, then npts*K distances.
// Writes a sidecar file; does not overwrite the search-only GT.
inline void save_live_groundtruth(const std::string& path, const LiveGroundtruth& gt) {
    if (path.empty()) {
        return;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to write live groundtruth: " + path);
    }
    const int32_t npts = static_cast<int32_t>(gt.lists.size());
    const int32_t k = static_cast<int32_t>(gt.K);
    out.write(reinterpret_cast<const char*>(&npts), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&k), sizeof(int32_t));
    std::vector<uint32_t> ids(static_cast<size_t>(k), 0);
    std::vector<float> dists(static_cast<size_t>(k), 0.0f);
    for (const auto& lst : gt.lists) {
        std::fill(ids.begin(), ids.end(), 0);
        std::fill(dists.begin(), dists.end(), 0.0f);
        const size_t n = std::min(lst.size(), static_cast<size_t>(k));
        for (size_t j = 0; j < n; ++j) {
            ids[j] = lst[j].id;
            dists[j] = lst[j].dist;
        }
        out.write(reinterpret_cast<const char*>(ids.data()), static_cast<size_t>(k) * sizeof(uint32_t));
    }
    for (const auto& lst : gt.lists) {
        std::fill(dists.begin(), dists.end(), 0.0f);
        const size_t n = std::min(lst.size(), static_cast<size_t>(k));
        for (size_t j = 0; j < n; ++j) {
            dists[j] = lst[j].dist;
        }
        out.write(reinterpret_cast<const char*>(dists.data()), static_cast<size_t>(k) * sizeof(float));
    }
    spdlog::info("{{\"event\": \"refresh_gt_write\", \"path\": \"{}\", \"n_queries\": {}, \"K\": {}}}",
                 path, npts, k);
}

// DiskANN truthset reader. Returns false if the sidecar is missing,
// truncated (still being written), or too small for n_eval / K.
inline bool load_live_groundtruth(const std::string& path, size_t n_eval, uint32_t K,
                                  LiveGroundtruth& out) {
    if (path.empty() || n_eval == 0 || K == 0) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const auto file_sz = static_cast<std::streamoff>(in.tellg());
    in.seekg(0, std::ios::beg);
    int32_t npts = 0, k = 0;
    in.read(reinterpret_cast<char*>(&npts), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&k), sizeof(int32_t));
    if (!in || npts <= 0 || k <= 0) {
        return false;
    }
    if (static_cast<size_t>(npts) < n_eval || static_cast<uint32_t>(k) < K) {
        return false;
    }
    const uint64_t expected =
        8ull + 2ull * static_cast<uint64_t>(npts) * static_cast<uint64_t>(k) * 4ull;
    if (file_sz < static_cast<std::streamoff>(expected)) {
        return false;
    }

    out.K = K;
    out.lists.assign(n_eval, {});
    std::vector<uint32_t> ids(static_cast<size_t>(k), 0);
    for (int32_t q = 0; q < npts; ++q) {
        in.read(reinterpret_cast<char*>(ids.data()),
                static_cast<size_t>(k) * sizeof(uint32_t));
        if (!in) {
            return false;
        }
        if (static_cast<size_t>(q) < n_eval) {
            out.lists[static_cast<size_t>(q)].reserve(K);
            for (uint32_t j = 0; j < K; ++j) {
                out.lists[static_cast<size_t>(q)].push_back(Neighbor{ids[j], 0.0f});
            }
        }
    }
    std::vector<float> dists(static_cast<size_t>(k), 0.0f);
    for (int32_t q = 0; q < npts; ++q) {
        in.read(reinterpret_cast<char*>(dists.data()),
                static_cast<size_t>(k) * sizeof(float));
        if (!in) {
            return false;
        }
        if (static_cast<size_t>(q) < n_eval) {
            for (uint32_t j = 0; j < K; ++j) {
                out.lists[static_cast<size_t>(q)][j].dist = dists[j];
            }
        }
    }
    spdlog::info("{{\"event\": \"refresh_gt_reuse\", \"path\": \"{}\", "
                 "\"n_queries\": {}, \"file_K\": {}, \"used_K\": {}}}",
                 path, n_eval, k, K);
    return true;
}

template <typename T>
void synthesize_extra(LiveVectors<T>& live, size_t n_extra, uint32_t seed) {
    if (n_extra == 0 || live.n_base < 2) {
        return;
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, live.n_base - 1);
    std::uniform_real_distribution<float> lam(0.02f, 0.49f);
    std::vector<T> tmp(live.aligned, static_cast<T>(0));
    uint32_t next_id = static_cast<uint32_t>(live.n_base);
    for (size_t i = 0; i < n_extra; ++i) {
        const T* a = live.base + pick(rng) * live.aligned;
        const T* b = live.base + pick(rng) * live.aligned;
        const float l = lam(rng);
        for (size_t d = 0; d < live.dim; ++d) {
            tmp[d] = lerp_comp(a[d], b[d], l);
        }
        live.add_extra(next_id++, tmp.data());
    }
}

inline double recall_of(const std::unordered_set<uint32_t>& gt, const uint32_t* tags,
                        uint32_t K, uint32_t invalid) {
    if (K == 0) {
        return 0.0;
    }
    size_t hits = 0;
    for (uint32_t j = 0; j < K; ++j) {
        if (tags[j] == 0 || tags[j] == invalid) {
            continue;
        }
        if (gt.count(tags[j] - 1) != 0) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(K);
}

struct QuerySample {
    double latency_ms = 0.0;
    double miss_penalty_ms = 0.0;
    double recall = -1.0;
    bool hit = false;
};

struct CacheSnapshot {
    size_t memory_active_vectors = 0;
    size_t memory_max_points = 0;
    size_t pca_active_regions = 0;
    size_t region_directory_size = 0;
    uint64_t point_evictions = 0;
    uint64_t region_invalidations = 0;
    std::map<size_t, size_t> index_vectors;
};

struct PhaseMetrics {
    size_t n_search = 0;
    size_t n_hit = 0;
    size_t n_stale_hit = 0;
    double search_ms = 0.0;
    double hit_ms = 0.0;
    double miss_ms = 0.0;
    double miss_penalty_sum = 0.0;
    double recall_sum = 0.0;
    double hit_recall_sum = 0.0;
    double miss_recall_sum = 0.0;
    std::vector<QuerySample> samples;

    size_t n_miss() const { return n_search > n_hit ? n_search - n_hit : 0; }
    double hit_ratio() const { return n_search ? static_cast<double>(n_hit) / static_cast<double>(n_search) : 0.0; }
    double miss_ratio() const { return n_search ? static_cast<double>(n_miss()) / static_cast<double>(n_search) : 0.0; }
    double recall_all() const { return n_search ? recall_sum / static_cast<double>(n_search) : 0.0; }
    double recall_hits() const { return n_hit ? hit_recall_sum / static_cast<double>(n_hit) : 0.0; }
    double recall_misses() const { return n_miss() ? miss_recall_sum / static_cast<double>(n_miss()) : 0.0; }
    double stale_hit_ratio() const { return n_hit ? static_cast<double>(n_stale_hit) / static_cast<double>(n_hit) : 0.0; }
    double avg_miss_penalty_ms() const {
        return n_miss() ? miss_penalty_sum / static_cast<double>(n_miss()) : 0.0;
    }

    void add(double latency_ms, bool hit, double recall, double miss_penalty_ms, bool stale_hit) {
        samples.push_back(QuerySample{latency_ms, hit ? 0.0 : miss_penalty_ms, recall, hit});
        n_search++;
        search_ms += latency_ms;
        if (recall >= 0.0) {
            recall_sum += recall;
            if (hit) {
                hit_recall_sum += recall;
            } else {
                miss_recall_sum += recall;
            }
        }
        if (hit) {
            n_hit++;
            hit_ms += latency_ms;
            if (stale_hit) {
                n_stale_hit++;
            }
        } else {
            miss_ms += latency_ms;
            miss_penalty_sum += miss_penalty_ms;
        }
    }
};

struct SliceStats {
    size_t n = 0;
    size_t hits = 0;
    size_t misses = 0;
    double latency_sum = 0.0;
    double hit_latency_sum = 0.0;
    double miss_latency_sum = 0.0;
    double miss_penalty_sum = 0.0;
    double recall_sum = 0.0;
    double hit_recall_sum = 0.0;
    double miss_recall_sum = 0.0;
    size_t n_recall = 0;
    size_t n_hit_recall = 0;
    size_t n_miss_recall = 0;
    size_t low_recall_queries = 0;
    size_t very_low_recall_queries = 0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;

    double hit_ratio() const { return n ? static_cast<double>(hits) / static_cast<double>(n) : 0.0; }
    double avg_latency_ms() const { return n ? latency_sum / static_cast<double>(n) : 0.0; }
    double avg_hit_latency_ms() const { return hits ? hit_latency_sum / static_cast<double>(hits) : 0.0; }
    double avg_miss_latency_ms() const { return misses ? miss_latency_sum / static_cast<double>(misses) : 0.0; }
    double avg_miss_penalty_ms() const {
        return misses ? miss_penalty_sum / static_cast<double>(misses) : 0.0;
    }
    double recall_all() const { return n_recall ? recall_sum / static_cast<double>(n_recall) : 0.0; }
    double recall_hits() const {
        return n_hit_recall ? hit_recall_sum / static_cast<double>(n_hit_recall) : 0.0;
    }
    double recall_misses() const {
        return n_miss_recall ? miss_recall_sum / static_cast<double>(n_miss_recall) : 0.0;
    }
    double qps(double elapsed_ms) const {
        return elapsed_ms > 0.0 ? static_cast<double>(n) / (elapsed_ms / 1000.0) : 0.0;
    }
};

inline SliceStats stats_of(const std::vector<QuerySample>& samples, size_t begin, size_t end) {
    SliceStats st;
    if (end > samples.size()) {
        end = samples.size();
    }
    if (begin >= end) {
        return st;
    }
    std::vector<double> lats;
    lats.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        const QuerySample& q = samples[i];
        st.n++;
        st.latency_sum += q.latency_ms;
        lats.push_back(q.latency_ms);
        if (q.hit) {
            st.hits++;
            st.hit_latency_sum += q.latency_ms;
            if (q.recall >= 0.0) {
                st.hit_recall_sum += q.recall;
                st.n_hit_recall++;
            }
        } else {
            st.misses++;
            st.miss_latency_sum += q.latency_ms;
            st.miss_penalty_sum += q.miss_penalty_ms;
            if (q.recall >= 0.0) {
                st.miss_recall_sum += q.recall;
                st.n_miss_recall++;
            }
        }
        if (q.recall >= 0.0) {
            st.recall_sum += q.recall;
            st.n_recall++;
            if (q.recall < 0.5) {
                st.low_recall_queries++;
            }
            if (q.recall < 0.1) {
                st.very_low_recall_queries++;
            }
        }
    }
    if (!lats.empty()) {
        std::sort(lats.begin(), lats.end());
        auto pct = [&](double p) {
            size_t idx = static_cast<size_t>(std::ceil(p * static_cast<double>(lats.size()))) - 1;
            if (idx >= lats.size()) {
                idx = lats.size() - 1;
            }
            return lats[idx];
        };
        st.p50 = pct(0.50);
        st.p90 = pct(0.90);
        st.p95 = pct(0.95);
        st.p99 = pct(0.99);
    }
    return st;
}

inline std::string index_vectors_json(const std::map<size_t, size_t>& index_vectors) {
    std::string out;
    for (const auto& [idx, count] : index_vectors) {
        if (!out.empty()) {
            out += ", ";
        }
        out += "\"index_" + std::to_string(idx) + "_vectors\": " + std::to_string(count);
    }
    if (out.empty()) {
        out = "\"index_vectors\": {}";
    }
    return out;
}

inline void log_search_progress(const char* cache_name, const char* phase, size_t completed,
                                size_t total, const PhaseMetrics& m, bool with_recall,
                                uint32_t K, const CacheSnapshot& snap, size_t interval_begin,
                                uint32_t threads = 1) {
    const size_t begin = std::min(interval_begin, m.samples.size());
    const size_t end = std::min(completed, m.samples.size());
    const SliceStats iv = stats_of(m.samples, begin, end);
    const SliceStats cum = stats_of(m.samples, 0, end);
    const double iv_elapsed = iv.latency_sum;
    const double cum_elapsed = cum.latency_sum;
    const double iv_qps = iv.qps(iv_elapsed);
    const std::string indexes = index_vectors_json(snap.index_vectors);
    if (with_recall) {
        spdlog::info("{{\"event\": \"refresh_progress\", \"cache\": \"{}\", "
                     "\"phase\": \"{}\", \"completed\": {}, \"total\": {}, "
                     "\"query_begin\": {}, \"query_end\": {}, \"interval_queries\": {}, "
                     "\"hit_ratio\": {}, \"hits\": {}, \"n_hit\": {}, \"n_miss\": {}, "
                     "\"avg_latency_ms\": {}, \"avg_hit_latency_ms\": {}, "
                     "\"avg_hit_ms\": {}, \"avg_miss_ms\": {}, \"miss_penalty_ms\": {}, "
                     "\"qps\": {}, \"qps_per_thread\": {}, "
                     "\"memory_active_vectors\": {}, \"memory_max_points\": {}, "
                     "\"pca_active_regions\": {}, \"region_directory_size\": {}, \"point_evictions\": {}, \"region_invalidations\": {}, "
                     "{}, "
                     "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                     "\"recall_all\": {}, \"K\": {}, "
                     "\"low_recall_queries\": {}, \"very_low_recall_queries\": {}, "
                     "\"recall_cache_hits\": {}, \"cache_hit_count\": {}, "
                     "\"recall_hits\": {}, \"recall_misses\": {}, "
                     "\"cum_hit_ratio\": {}, \"cum_hits\": {}, "
                     "\"cum_avg_latency_ms\": {}, \"cum_qps\": {}, \"elapsed_ms\": {}, "
                     "\"cum_recall_all\": {}, \"cum_recall_hits\": {}, \"cum_recall_misses\": {}}}",
                     cache_name, phase, completed, total,
                     begin, end > begin ? end - 1 : begin, iv.n,
                     iv.hit_ratio(), iv.hits, m.n_hit, m.n_miss(),
                     iv.avg_latency_ms(), iv.avg_hit_latency_ms(),
                     iv.avg_hit_latency_ms(), iv.avg_miss_latency_ms(), iv.avg_miss_penalty_ms(),
                     iv_qps, threads ? iv_qps / static_cast<double>(threads) : iv_qps,
                     snap.memory_active_vectors, snap.memory_max_points,
                     snap.pca_active_regions, snap.region_directory_size, snap.point_evictions, snap.region_invalidations,
                     indexes,
                     iv.p50, iv.p90, iv.p95, iv.p99,
                     iv.recall_all(), K,
                     iv.low_recall_queries, iv.very_low_recall_queries,
                     iv.recall_hits(), iv.hits,
                     iv.recall_hits(), iv.recall_misses(),
                     cum.hit_ratio(), cum.hits,
                     cum.avg_latency_ms(), cum.qps(cum_elapsed), cum_elapsed,
                     cum.recall_all(), cum.recall_hits(), cum.recall_misses());
    } else {
        spdlog::info("{{\"event\": \"refresh_progress\", \"cache\": \"{}\", "
                     "\"phase\": \"{}\", \"completed\": {}, \"total\": {}, "
                     "\"query_begin\": {}, \"query_end\": {}, \"interval_queries\": {}, "
                     "\"hit_ratio\": {}, \"hits\": {}, \"n_hit\": {}, \"n_miss\": {}, "
                     "\"avg_latency_ms\": {}, \"avg_hit_latency_ms\": {}, "
                     "\"avg_hit_ms\": {}, \"avg_miss_ms\": {}, \"miss_penalty_ms\": {}, "
                     "\"qps\": {}, \"qps_per_thread\": {}, "
                     "\"memory_active_vectors\": {}, \"memory_max_points\": {}, "
                     "\"pca_active_regions\": {}, \"region_directory_size\": {}, \"point_evictions\": {}, \"region_invalidations\": {}, "
                     "{}, "
                     "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                     "\"recall_all\": null, \"K\": {}, "
                     "\"recall_cache_hits\": null, \"cache_hit_count\": {}, "
                     "\"cum_hit_ratio\": {}, \"cum_hits\": {}, "
                     "\"cum_avg_latency_ms\": {}, \"cum_qps\": {}, \"elapsed_ms\": {}}}",
                     cache_name, phase, completed, total,
                     begin, end > begin ? end - 1 : begin, iv.n,
                     iv.hit_ratio(), iv.hits, m.n_hit, m.n_miss(),
                     iv.avg_latency_ms(), iv.avg_hit_latency_ms(),
                     iv.avg_hit_latency_ms(), iv.avg_miss_latency_ms(), iv.avg_miss_penalty_ms(),
                     iv_qps, threads ? iv_qps / static_cast<double>(threads) : iv_qps,
                     snap.memory_active_vectors, snap.memory_max_points,
                     snap.pca_active_regions, snap.region_directory_size, snap.point_evictions, snap.region_invalidations,
                     indexes,
                     iv.p50, iv.p90, iv.p95, iv.p99,
                     K, iv.hits,
                     cum.hit_ratio(), cum.hits,
                     cum.avg_latency_ms(), cum.qps(cum_elapsed), cum_elapsed);
    }
}

inline void log_search_phase(const char* cache_name, const char* phase, const PhaseMetrics& m,
                             const CacheSnapshot& snap, bool with_recall, uint32_t K,
                             uint32_t threads = 1) {
    const SliceStats all = stats_of(m.samples, 0, m.samples.size());
    const size_t n_miss = m.n_miss();
    const double sec = m.search_ms / 1000.0;
    const double qps = sec > 0.0 ? static_cast<double>(m.n_search) / sec : 0.0;
    const std::string indexes = index_vectors_json(snap.index_vectors);
    if (with_recall) {
        spdlog::info("{{\"event\": \"refresh_{}\", \"cache\": \"{}\", "
                     "\"n_search\": {}, \"n_hit\": {}, \"n_miss\": {}, "
                     "\"hit_ratio\": {}, \"miss_ratio\": {}, \"stale_hit_ratio\": {}, "
                     "\"hits\": {}, \"total_queries\": {}, \"threads\": {}, "
                     "\"avg_latency_ms\": {}, \"avg_search_ms\": {}, "
                     "\"avg_hit_latency_ms\": {}, \"avg_hit_ms\": {}, \"avg_miss_ms\": {}, "
                     "\"miss_penalty_ms\": {}, "
                     "\"qps\": {}, \"qps_per_thread\": {}, \"elapsed_ms\": {}, "
                     "\"memory_active_vectors\": {}, \"memory_max_points\": {}, "
                     "\"pca_active_regions\": {}, \"region_directory_size\": {}, \"point_evictions\": {}, \"region_invalidations\": {}, "
                     "{}, "
                     "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                     "\"recall_all\": {}, \"K\": {}, "
                     "\"low_recall_queries\": {}, \"very_low_recall_queries\": {}, "
                     "\"recall_cache_hits\": {}, \"cache_hit_count\": {}, "
                     "\"recall_hits\": {}, \"recall_misses\": {}}}",
                     phase, cache_name, m.n_search, m.n_hit, n_miss,
                     m.hit_ratio(), m.miss_ratio(), m.stale_hit_ratio(),
                     m.n_hit, m.n_search, threads,
                     all.avg_latency_ms(), m.n_search ? m.search_ms / static_cast<double>(m.n_search) : 0.0,
                     all.avg_hit_latency_ms(), all.avg_hit_latency_ms(), all.avg_miss_latency_ms(),
                     all.avg_miss_penalty_ms(),
                     qps, threads ? qps / static_cast<double>(threads) : qps, m.search_ms,
                     snap.memory_active_vectors, snap.memory_max_points,
                     snap.pca_active_regions, snap.region_directory_size, snap.point_evictions, snap.region_invalidations,
                     indexes,
                     all.p50, all.p90, all.p95, all.p99,
                     m.recall_all(), K,
                     all.low_recall_queries, all.very_low_recall_queries,
                     m.recall_hits(), m.n_hit,
                     m.recall_hits(), m.recall_misses());
    } else {
        spdlog::info("{{\"event\": \"refresh_{}\", \"cache\": \"{}\", "
                     "\"n_search\": {}, \"n_hit\": {}, \"n_miss\": {}, "
                     "\"hit_ratio\": {}, \"miss_ratio\": {}, \"stale_hit_ratio\": {}, "
                     "\"hits\": {}, \"total_queries\": {}, \"threads\": {}, "
                     "\"avg_latency_ms\": {}, \"avg_search_ms\": {}, "
                     "\"avg_hit_latency_ms\": {}, \"avg_hit_ms\": {}, \"avg_miss_ms\": {}, "
                     "\"miss_penalty_ms\": {}, "
                     "\"qps\": {}, \"qps_per_thread\": {}, \"elapsed_ms\": {}, "
                     "\"memory_active_vectors\": {}, \"memory_max_points\": {}, "
                     "\"pca_active_regions\": {}, \"region_directory_size\": {}, \"point_evictions\": {}, \"region_invalidations\": {}, "
                     "{}, "
                     "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                     "\"recall_all\": null, \"K\": {}, "
                     "\"recall_cache_hits\": null, \"cache_hit_count\": {}}}",
                     phase, cache_name, m.n_search, m.n_hit, n_miss,
                     m.hit_ratio(), m.miss_ratio(), m.stale_hit_ratio(),
                     m.n_hit, m.n_search, threads,
                     all.avg_latency_ms(), m.n_search ? m.search_ms / static_cast<double>(m.n_search) : 0.0,
                     all.avg_hit_latency_ms(), all.avg_hit_latency_ms(), all.avg_miss_latency_ms(),
                     all.avg_miss_penalty_ms(),
                     qps, threads ? qps / static_cast<double>(threads) : qps, m.search_ms,
                     snap.memory_active_vectors, snap.memory_max_points,
                     snap.pca_active_regions, snap.region_directory_size, snap.point_evictions, snap.region_invalidations,
                     indexes,
                     all.p50, all.p90, all.p95, all.p99,
                     K, m.n_hit);
    }
}

inline void log_mutate_phase(const char* cache_name, const char* phase, size_t n,
                             double elapsed_ms, double extra = -1.0,
                             uint64_t region_invalidations = 0) {
    const double avg = n ? elapsed_ms / static_cast<double>(n) : 0.0;
    const double sec = elapsed_ms / 1000.0;
    const double qps = sec > 0.0 ? static_cast<double>(n) / sec : 0.0;
    if (extra >= 0.0) {
        spdlog::info("{{\"event\": \"refresh_{}\", \"cache\": \"{}\", \"n\": {}, "
                     "\"delete_rate\": {}, \"elapsed_ms\": {}, \"avg_ms\": {}, \"qps\": {}, "
                     "\"region_invalidations\": {}}}",
                     phase, cache_name, n, extra, elapsed_ms, avg, qps, region_invalidations);
    } else {
        spdlog::info("{{\"event\": \"refresh_{}\", \"cache\": \"{}\", \"n\": {}, "
                     "\"elapsed_ms\": {}, \"avg_ms\": {}, \"qps\": {}, "
                     "\"region_invalidations\": {}}}",
                     phase, cache_name, n, elapsed_ms, avg, qps, region_invalidations);
    }
}

inline LiveGroundtruth require_search_only_gt(const std::string& path, size_t n_queries,
                                              uint32_t K) {
    LiveGroundtruth gt;
    if (!load_live_groundtruth(path, n_queries, K, gt)) {
        throw std::runtime_error(
            "warmup groundtruth missing or incomplete (not computed): " + path);
    }
    return gt;
}

inline void log_gt_phase(const char* cache_name, size_t n_eval, size_t n_live,
                         double elapsed_ms, bool reused) {
    spdlog::info("{{\"event\": \"refresh_gt\", \"cache\": \"{}\", \"n_eval\": {}, "
                 "\"n_live\": {}, \"elapsed_ms\": {}, \"avg_ms_per_query\": {}, "
                 "\"reused\": {}}}",
                 cache_name, n_eval, n_live, elapsed_ms,
                 n_eval ? elapsed_ms / static_cast<double>(n_eval) : 0.0,
                 reused);
}

}  // namespace qvcache_refresh

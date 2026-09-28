#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include <spdlog/spdlog.h>

namespace qvcache_update {

struct UpdateMetrics {
    size_t n_search = 0;
    size_t n_insert = 0;
    size_t n_delete = 0;
    size_t search_hits = 0;
    size_t stale_hits = 0;
    size_t visibility_checks = 0;
    size_t visibility_hits = 0;
    double search_ms = 0.0;
    double insert_ms = 0.0;
    double delete_ms = 0.0;
    double hit_ms = 0.0;
};

inline void log_interval(const char* cache_name, size_t done, size_t total,
                         const UpdateMetrics& interval, const UpdateMetrics& cum,
                         size_t memory_active_vectors = 0) {
    const double search_n = static_cast<double>(std::max<size_t>(interval.n_search, 1));
    const double insert_n = static_cast<double>(std::max<size_t>(interval.n_insert, 1));
    const double delete_n = static_cast<double>(std::max<size_t>(interval.n_delete, 1));
    spdlog::info("{{\"event\": \"update_metrics\", \"cache\": \"{}\", "
                 "\"completed\": {}, \"total\": {}, "
                 "\"n_search\": {}, \"n_insert\": {}, \"n_delete\": {}, "
                 "\"hit_ratio\": {}, \"stale_hit_ratio\": {}, "
                 "\"avg_search_ms\": {}, \"avg_hit_ms\": {}, "
                 "\"avg_insert_ms\": {}, \"avg_delete_ms\": {}, "
                 "\"visibility_ratio\": {}, "
                 "\"cum_search\": {}, \"cum_insert\": {}, \"cum_delete\": {}, "
                 "\"cum_hit_ratio\": {}, \"memory_active_vectors\": {}}}",
                 cache_name, done, total,
                 interval.n_search, interval.n_insert, interval.n_delete,
                 interval.n_search ? static_cast<double>(interval.search_hits) / search_n : 0.0,
                 interval.search_hits ? static_cast<double>(interval.stale_hits) / static_cast<double>(interval.search_hits) : 0.0,
                 interval.n_search ? interval.search_ms / search_n : 0.0,
                 interval.search_hits ? interval.hit_ms / static_cast<double>(interval.search_hits) : 0.0,
                 interval.n_insert ? interval.insert_ms / insert_n : 0.0,
                 interval.n_delete ? interval.delete_ms / delete_n : 0.0,
                 interval.visibility_checks ? static_cast<double>(interval.visibility_hits) / static_cast<double>(interval.visibility_checks) : 0.0,
                 cum.n_search, cum.n_insert, cum.n_delete,
                 cum.n_search ? static_cast<double>(cum.search_hits) / static_cast<double>(cum.n_search) : 0.0,
                 memory_active_vectors);
}

inline int pick_op(double u, double insert_ratio, double delete_ratio) {
    if (u < delete_ratio) {
        return 2;
    }
    if (u < delete_ratio + insert_ratio) {
        return 1;
    }
    return 0;
}

}  // namespace qvcache_update

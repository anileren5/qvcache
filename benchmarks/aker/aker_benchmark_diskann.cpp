// Aker: windowed DiskANN benchmark with Aker as the result cache (QVCache counterpart).
// Workload / metrics match benchmarks/qvcache/qvcache_benchmark_diskann.cpp.

#include <cstddef>
#include <omp.h>
#include <boost/program_options.hpp>
#include <atomic>
#include <chrono>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <vector>
#include <algorithm>
#include <set>
#include <iostream>
#include <cmath>
#include <map>
#include <limits>
#include <utility>
#include <random>
#include <cstring>
#include <fstream>
#include <memory>
#include <type_traits>

#include "greator_backend.h"
#include "diskann/distance.h"
#include "greator/utils.h"
#include "ak_anns_cache_c_wrapper.h"
#include "aker_c_abi_helpers.h"

namespace po = boost::program_options;

struct HybridMetrics {
    double hit_ratio;
    size_t hits;
    size_t exact_hits;
    size_t approx_hits;
    double exact_hit_ratio;
    double approx_hit_ratio;
    size_t total_queries;
    uint32_t threads;
    double avg_latency_ms;
    double avg_hit_latency_ms;
    double avg_exact_hit_latency_ms;
    double avg_approx_hit_latency_ms;
    double avg_miss_penalty_ms;
    double qps;
    double qps_per_thread;
    size_t memory_active_vectors;
    size_t memory_max_points;
    size_t pca_active_regions;
    std::map<size_t, size_t> index_vectors;
    double p50;
    double p90;
    double p95;
    double p99;
};

struct RecallAllMetrics {
    double recall_all;
    uint32_t K;
    size_t low_recall_queries;
    size_t very_low_recall_queries;
};

struct RecallHitMetrics {
    double recall_cache_hits;
    size_t cache_hit_count;
};

template <typename T>
bool (*aker_transform_for_type())(void*, size_t, size_t, void*, uint8_t*) {
    if constexpr (std::is_same_v<T, float>)
        return aker_transform_float;
    else if constexpr (std::is_same_v<T, int8_t>)
        return aker_transform_int8;
    else
        return aker_transform_uint8;
}

template <typename T>
float (*aker_distance_for_type(diskann::Metric metric))(uint8_t*, uint8_t*, size_t) {
    if constexpr (std::is_same_v<T, float>) {
        if (metric == diskann::Metric::INNER_PRODUCT)
            return aker_ip_float_bytes;
        return aker_l2_float_bytes;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return aker_l2_int8_bytes;
    } else {
        return aker_l2_uint8_bytes;
    }
}

template <typename T, typename TagT = uint32_t>
RecallAllMetrics calculate_recall(size_t K, TagT* groundtruth_ids, std::vector<TagT>& query_result_tags, size_t query_num, size_t groundtruth_dim) {
    double total_recall = 0.0;
    std::vector<double> recall_by_query;
    const TagT INVALID_ID = std::numeric_limits<TagT>::max();

    for (int32_t i = 0; i < query_num; i++) {
        std::set<uint32_t> groundtruth_closest_neighbors;
        std::set<uint32_t> calculated_closest_neighbors;
        for (int32_t j = 0; j < K; j++) {
            groundtruth_closest_neighbors.insert(*(groundtruth_ids + i * groundtruth_dim + j));
        }
        for (int32_t j = 0; j < K; j++) {
            TagT tag = *(query_result_tags.data() + i * K + j);
            if (tag != INVALID_ID) {
                calculated_closest_neighbors.insert(tag);
            }
        }
        uint32_t matching_neighbors = 0;
        for (uint32_t x : calculated_closest_neighbors) {
            if (groundtruth_closest_neighbors.count(x - 1)) matching_neighbors++;
        }
        double recall = matching_neighbors / (double)K;
        recall_by_query.push_back(recall);
        total_recall += recall;
    }
    double average_recall = total_recall / (query_num);

    size_t low_recall_count = 0;
    size_t very_low_recall_count = 0;
    for (double r : recall_by_query) {
        if (r < 0.5) low_recall_count++;
        if (r < 0.1) very_low_recall_count++;
    }

    RecallAllMetrics metrics;
    metrics.recall_all = average_recall;
    metrics.K = K;
    metrics.low_recall_queries = low_recall_count;
    metrics.very_low_recall_queries = very_low_recall_count;
    return metrics;
}

template <typename T, typename TagT = uint32_t>
RecallHitMetrics calculate_hit_recall(size_t K, TagT* groundtruth_ids, std::vector<TagT>& query_result_tags,
                         const std::vector<bool>& hit_results, size_t query_num, size_t groundtruth_dim) {
    double total_recall = 0.0;
    size_t hit_count = 0;
    const TagT INVALID_ID = std::numeric_limits<TagT>::max();

    for (int32_t i = 0; i < query_num; i++) {
        if (hit_results[i]) {
            std::set<uint32_t> groundtruth_closest_neighbors;
            std::set<uint32_t> calculated_closest_neighbors;
            for (int32_t j = 0; j < K; j++) {
                groundtruth_closest_neighbors.insert(*(groundtruth_ids + i * groundtruth_dim + j));
            }
            for (int32_t j = 0; j < K; j++) {
                TagT tag = *(query_result_tags.data() + i * K + j);
                if (tag != INVALID_ID) {
                    calculated_closest_neighbors.insert(tag);
                }
            }
            uint32_t matching_neighbors = 0;
            for (uint32_t x : calculated_closest_neighbors) {
                if (groundtruth_closest_neighbors.count(x - 1)) matching_neighbors++;
            }
            total_recall += matching_neighbors / (double)K;
            hit_count++;
        }
    }

    RecallHitMetrics metrics;
    metrics.recall_cache_hits = (hit_count > 0) ? total_recall / hit_count : -1.0;
    metrics.cache_hit_count = hit_count;
    return metrics;
}

template <typename T, typename TagT = uint32_t>
void log_window_metrics(int window_idx, int repeat_idx, const HybridMetrics& hybrid_metrics, const RecallAllMetrics& recall_all_metrics, const RecallHitMetrics& recall_hit_metrics) {
    std::string mini_index_counts = "";
    for (const auto& [idx, count] : hybrid_metrics.index_vectors) {
        if (!mini_index_counts.empty()) mini_index_counts += ", ";
        mini_index_counts += "\"index_" + std::to_string(idx) + "_vectors\": " + std::to_string(count);
    }
    if (mini_index_counts.empty()) {
        mini_index_counts = "\"aker_pool_size\": " + std::to_string(hybrid_metrics.memory_max_points);
    }

    if (recall_hit_metrics.recall_cache_hits < 0) {
        spdlog::info("{{\"event\": \"window_metrics\", "
                  "\"window_idx\": {}, "
                  "\"repeat_idx\": {}, "
                  "\"hit_ratio\": {}, "
                  "\"hits\": {}, "
                  "\"exact_hits\": {}, "
                  "\"approx_hits\": {}, "
                  "\"exact_hit_ratio\": {}, "
                  "\"approx_hit_ratio\": {}, "
                  "\"total_queries\": {}, "
                  "\"threads\": {}, "
                  "\"avg_latency_ms\": {}, "
                  "\"avg_hit_latency_ms\": {}, "
                  "\"avg_exact_hit_latency_ms\": {}, "
                  "\"avg_approx_hit_latency_ms\": {}, "
                  "\"miss_penalty_ms\": {}, "
                  "\"qps\": {}, "
                  "\"qps_per_thread\": {}, "
                  "\"memory_active_vectors\": {}, "
                  "\"memory_max_points\": {}, "
                  "\"pca_active_regions\": {}, "
                  "{}, "
                  "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                  "\"recall_all\": {}, "
                  "\"K\": {}, "
                  "\"low_recall_queries\": {}, "
                  "\"very_low_recall_queries\": {}, "
                  "\"recall_cache_hits\": null, "
                  "\"cache_hit_count\": {}}}",
                  window_idx, repeat_idx,
                  hybrid_metrics.hit_ratio, hybrid_metrics.hits,
                  hybrid_metrics.exact_hits, hybrid_metrics.approx_hits,
                  hybrid_metrics.exact_hit_ratio, hybrid_metrics.approx_hit_ratio,
                  hybrid_metrics.total_queries,
                  hybrid_metrics.threads, hybrid_metrics.avg_latency_ms, hybrid_metrics.avg_hit_latency_ms,
                  hybrid_metrics.avg_exact_hit_latency_ms, hybrid_metrics.avg_approx_hit_latency_ms,
                  hybrid_metrics.avg_miss_penalty_ms,
                  hybrid_metrics.qps, hybrid_metrics.qps_per_thread,
                  hybrid_metrics.memory_active_vectors, hybrid_metrics.memory_max_points,
                  hybrid_metrics.pca_active_regions, mini_index_counts,
                  hybrid_metrics.p50, hybrid_metrics.p90, hybrid_metrics.p95, hybrid_metrics.p99,
                  recall_all_metrics.recall_all, recall_all_metrics.K,
                  recall_all_metrics.low_recall_queries, recall_all_metrics.very_low_recall_queries,
                  recall_hit_metrics.cache_hit_count);
    } else {
        spdlog::info("{{\"event\": \"window_metrics\", "
                  "\"window_idx\": {}, "
                  "\"repeat_idx\": {}, "
                  "\"hit_ratio\": {}, "
                  "\"hits\": {}, "
                  "\"exact_hits\": {}, "
                  "\"approx_hits\": {}, "
                  "\"exact_hit_ratio\": {}, "
                  "\"approx_hit_ratio\": {}, "
                  "\"total_queries\": {}, "
                  "\"threads\": {}, "
                  "\"avg_latency_ms\": {}, "
                  "\"avg_hit_latency_ms\": {}, "
                  "\"avg_exact_hit_latency_ms\": {}, "
                  "\"avg_approx_hit_latency_ms\": {}, "
                  "\"miss_penalty_ms\": {}, "
                  "\"qps\": {}, "
                  "\"qps_per_thread\": {}, "
                  "\"memory_active_vectors\": {}, "
                  "\"memory_max_points\": {}, "
                  "\"pca_active_regions\": {}, "
                  "{}, "
                  "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                  "\"recall_all\": {}, "
                  "\"K\": {}, "
                  "\"low_recall_queries\": {}, "
                  "\"very_low_recall_queries\": {}, "
                  "\"recall_cache_hits\": {}, "
                  "\"cache_hit_count\": {}}}",
                  window_idx, repeat_idx,
                  hybrid_metrics.hit_ratio, hybrid_metrics.hits,
                  hybrid_metrics.exact_hits, hybrid_metrics.approx_hits,
                  hybrid_metrics.exact_hit_ratio, hybrid_metrics.approx_hit_ratio,
                  hybrid_metrics.total_queries,
                  hybrid_metrics.threads, hybrid_metrics.avg_latency_ms, hybrid_metrics.avg_hit_latency_ms,
                  hybrid_metrics.avg_exact_hit_latency_ms, hybrid_metrics.avg_approx_hit_latency_ms,
                  hybrid_metrics.avg_miss_penalty_ms,
                  hybrid_metrics.qps, hybrid_metrics.qps_per_thread,
                  hybrid_metrics.memory_active_vectors, hybrid_metrics.memory_max_points,
                  hybrid_metrics.pca_active_regions, mini_index_counts,
                  hybrid_metrics.p50, hybrid_metrics.p90, hybrid_metrics.p95, hybrid_metrics.p99,
                  recall_all_metrics.recall_all, recall_all_metrics.K,
                  recall_all_metrics.low_recall_queries, recall_all_metrics.very_low_recall_queries,
                  recall_hit_metrics.recall_cache_hits, recall_hit_metrics.cache_hit_count);
    }
}

template <typename T, typename TagT>
bool aker_insert_miss_entry(
    anns_cache_c_wrapper_t* cache,
    qvcache::BackendInterface<T, TagT>& backend,
    char* query_slot,
    char* query_view,
    uint64_t query_id,
    uint32_t dim,
    size_t vector_in_bytes,
    uint32_t slot_list_size,
    TagT* result_tags,
    float* result_dists
) {
    std::vector<TagT> fetch_ids(result_tags, result_tags + slot_list_size);
    std::vector<std::vector<T>> neighbor_vecs = backend.fetch_vectors_by_ids(fetch_ids);
    if (neighbor_vecs.size() < slot_list_size) {
        return false;
    }

    std::vector<char*> neighbor_slots(slot_list_size, nullptr);
    for (uint32_t j = 0; j < slot_list_size; ++j) {
        std::vector<T> packed(dim);
        const size_t copy_n = std::min(static_cast<size_t>(dim), neighbor_vecs[j].size());
        std::memcpy(packed.data(), neighbor_vecs[j].data(), copy_n * sizeof(T));
        const uint64_t nid = static_cast<uint64_t>(result_tags[j]) + 1;
        neighbor_slots[j] = akerCreateVectorSlot(
            nid,
            vector_in_bytes,
            reinterpret_cast<char*>(packed.data()),
            0,
            0,
            result_dists[j]);
    }

    char* entry = akerCreateCacheEntry(
        cache,
        query_slot,
        slot_list_size,
        neighbor_slots.data());

    bool inserted = false;
    if (entry != nullptr) {
        inserted = akerInsertCacheEntry(cache, query_id, entry, query_view);
        if (!inserted)
            akerDestroyCacheEntry(entry);
    }

    for (char* slot : neighbor_slots)
        akerDestroyVectorSlot(slot);

    return inserted;
}

template <typename T, typename TagT = uint32_t>
void log_stream_metrics(
    size_t completed, size_t total, size_t query_begin, size_t query_end,
    const HybridMetrics& interval, const RecallAllMetrics& recall_all,
    const RecallHitMetrics& recall_hits,
    double cum_hit_ratio, size_t cum_hits, double cum_avg_latency_ms, double cum_qps, double elapsed_ms
) {
    // Aker: search-workload progress (file order, no windows).
    if (recall_hits.recall_cache_hits < 0) {
        spdlog::info("{{\"event\": \"stream_metrics\", "
                  "\"completed\": {}, \"total\": {}, \"query_begin\": {}, \"query_end\": {}, "
                  "\"interval_queries\": {}, "
                  "\"hit_ratio\": {}, \"hits\": {}, "
                  "\"exact_hits\": {}, \"approx_hits\": {}, "
                  "\"exact_hit_ratio\": {}, \"approx_hit_ratio\": {}, "
                  "\"avg_latency_ms\": {}, \"avg_hit_latency_ms\": {}, "
                  "\"avg_exact_hit_latency_ms\": {}, \"avg_approx_hit_latency_ms\": {}, "
                  "\"miss_penalty_ms\": {}, "
                  "\"qps\": {}, "
                  "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                  "\"recall_all\": {}, \"K\": {}, "
                  "\"recall_cache_hits\": null, \"cache_hit_count\": {}, "
                  "\"cum_hit_ratio\": {}, \"cum_hits\": {}, "
                  "\"cum_avg_latency_ms\": {}, \"cum_qps\": {}, \"elapsed_ms\": {}}}",
                  completed, total, query_begin, query_end,
                  interval.total_queries,
                  interval.hit_ratio, interval.hits,
                  interval.exact_hits, interval.approx_hits,
                  interval.exact_hit_ratio, interval.approx_hit_ratio,
                  interval.avg_latency_ms, interval.avg_hit_latency_ms,
                  interval.avg_exact_hit_latency_ms, interval.avg_approx_hit_latency_ms,
                  interval.avg_miss_penalty_ms,
                  interval.qps,
                  interval.p50, interval.p90, interval.p95, interval.p99,
                  recall_all.recall_all, recall_all.K,
                  recall_hits.cache_hit_count,
                  cum_hit_ratio, cum_hits, cum_avg_latency_ms, cum_qps, elapsed_ms);
    } else {
        spdlog::info("{{\"event\": \"stream_metrics\", "
                  "\"completed\": {}, \"total\": {}, \"query_begin\": {}, \"query_end\": {}, "
                  "\"interval_queries\": {}, "
                  "\"hit_ratio\": {}, \"hits\": {}, "
                  "\"exact_hits\": {}, \"approx_hits\": {}, "
                  "\"exact_hit_ratio\": {}, \"approx_hit_ratio\": {}, "
                  "\"avg_latency_ms\": {}, \"avg_hit_latency_ms\": {}, "
                  "\"avg_exact_hit_latency_ms\": {}, \"avg_approx_hit_latency_ms\": {}, "
                  "\"miss_penalty_ms\": {}, "
                  "\"qps\": {}, "
                  "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                  "\"recall_all\": {}, \"K\": {}, "
                  "\"recall_cache_hits\": {}, \"cache_hit_count\": {}, "
                  "\"cum_hit_ratio\": {}, \"cum_hits\": {}, "
                  "\"cum_avg_latency_ms\": {}, \"cum_qps\": {}, \"elapsed_ms\": {}}}",
                  completed, total, query_begin, query_end,
                  interval.total_queries,
                  interval.hit_ratio, interval.hits,
                  interval.exact_hits, interval.approx_hits,
                  interval.exact_hit_ratio, interval.approx_hit_ratio,
                  interval.avg_latency_ms, interval.avg_hit_latency_ms,
                  interval.avg_exact_hit_latency_ms, interval.avg_approx_hit_latency_ms,
                  interval.avg_miss_penalty_ms,
                  interval.qps,
                  interval.p50, interval.p90, interval.p95, interval.p99,
                  recall_all.recall_all, recall_all.K,
                  recall_hits.recall_cache_hits, recall_hits.cache_hit_count,
                  cum_hit_ratio, cum_hits, cum_avg_latency_ms, cum_qps, elapsed_ms);
    }
}

template <typename T, typename TagT = uint32_t>
std::pair<std::vector<bool>, HybridMetrics> hybrid_search(
    anns_cache_c_wrapper_t* cache,
    qvcache::BackendInterface<T, TagT>& backend,
    const T* query, size_t query_num, uint32_t query_aligned_dim, uint32_t dim,
    uint32_t K, uint32_t slot_list_size, uint32_t search_threads,
    std::vector<uint32_t>& query_result_tags,
    diskann::Metric metric,
    size_t pool_size
) {
    const size_t vector_in_bytes = static_cast<size_t>(dim) * sizeof(T);
    auto transform = aker_transform_for_type<T>();
    auto distance = aker_distance_for_type<T>(metric);
    const TagT INVALID_ID = std::numeric_limits<TagT>::max();

    std::vector<float> query_result_dists(static_cast<size_t>(slot_list_size) * query_num, 0.0f);
    greator::QueryStats* stats = new greator::QueryStats[query_num];
    std::vector<double> latencies_ms(query_num, 0.0);
    std::vector<double> miss_penalties_ms(query_num, 0.0);
    std::vector<bool> hit_results(query_num, false);
    std::vector<char> similar_hits(query_num, 0);
    std::vector<TagT> miss_tags(query_num * static_cast<size_t>(slot_list_size), 0);
    aker_pin_openmp((int)search_threads);

    auto global_start = std::chrono::high_resolution_clock::now();
    std::atomic<size_t> hit_count{0};
    std::atomic<size_t> exact_hit_count{0};
    std::atomic<size_t> approx_hit_count{0};

    #pragma omp parallel for num_threads((int32_t)search_threads) schedule(dynamic)
    for (size_t i = 0; i < query_num; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<T> packed(dim);
        std::memcpy(packed.data(), query + i * query_aligned_dim, dim * sizeof(T));

        uint64_t query_id = akerDefaultHash(reinterpret_cast<char*>(packed.data()), vector_in_bytes);
        if (query_id == 0)
            query_id = 1;

        char* query_slot = akerCreateVectorSlot(
            query_id,
            vector_in_bytes,
            reinterpret_cast<char*>(packed.data()),
            0,
            0,
            0.0f);
        char* query_view = akerCreateVectorView(
            query_slot,
            static_cast<size_t>(dim),
            vector_in_bytes,
            transform);

        bool similar = false;
        bool invalid = false;
        char* entry = akerGetCacheEntry(cache, query_view, &similar, &invalid, distance);

        TagT* out_tags = query_result_tags.data() + i * K;
        for (uint32_t j = 0; j < K; ++j)
            out_tags[j] = INVALID_ID;

        bool hit = false;
        if (entry != nullptr && !invalid) {
            hit = true;
            for (uint32_t j = 0; j < K; ++j) {
                char* nslot = akerGetResultVectorSlotAt(entry, static_cast<int>(j));
                if (nslot == nullptr)
                    break;
                out_tags[j] = static_cast<TagT>(akerGetVectorIdFromVectorSlot(nslot));
            }
            // Aker: pgvector/hnswgettuple — on approx-hit, dummy entry + link so the
            // same query later exact-hits the lookup table (skips query filter).
            if (similar) {
                const uint64_t found_id = akerGetQueryVectorIdFromCacheEntry(entry);
                char* alias_entry = akerCreateCacheEntry(
                    cache, query_slot, slot_list_size, nullptr);
                if (alias_entry != nullptr && !akerLinkCacheEntry(cache, alias_entry, found_id))
                    akerDestroyCacheEntry(alias_entry);
            }
            akerDestroyCacheEntry(entry);
        } else {
            if (entry != nullptr)
                akerDestroyCacheEntry(entry);

            miss_penalties_ms[i] = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start).count();

            backend.search(
                query + i * query_aligned_dim,
                slot_list_size,
                miss_tags.data() + i * slot_list_size,
                query_result_dists.data() + i * slot_list_size,
                nullptr,
                stats + i);

            for (uint32_t j = 0; j < K; ++j)
                out_tags[j] = miss_tags[i * slot_list_size + j] + 1;
        }

        hit_results[i] = hit;
        if (hit) {
            hit_count.fetch_add(1, std::memory_order_relaxed);
            if (similar) {
                similar_hits[i] = 1;
                approx_hit_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                exact_hit_count.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Official Aker inserts in hnswgettuple on the miss path before the
        // client sees results. Count that cost in miss latency (QVCache insert
        // stays off-path via insert_pool).
        if (!hit) {
            aker_insert_miss_entry(
                cache,
                backend,
                query_slot,
                query_view,
                query_id,
                dim,
                vector_in_bytes,
                slot_list_size,
                miss_tags.data() + i * slot_list_size,
                query_result_dists.data() + i * slot_list_size);
        }

        auto end = std::chrono::high_resolution_clock::now();
        latencies_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();

        akerDestroyVectorView(query_view);
        akerDestroyVectorSlot(query_slot);
    }

    double total_hit_latency_ms = 0.0;
    double total_exact_hit_latency_ms = 0.0;
    double total_approx_hit_latency_ms = 0.0;
    size_t actual_hit_count = 0;
    size_t actual_exact_hit_count = 0;
    size_t actual_approx_hit_count = 0;
    for (size_t i = 0; i < query_num; i++) {
        if (hit_results[i]) {
            total_hit_latency_ms += latencies_ms[i];
            actual_hit_count++;
            if (similar_hits[i]) {
                total_approx_hit_latency_ms += latencies_ms[i];
                actual_approx_hit_count++;
            } else {
                total_exact_hit_latency_ms += latencies_ms[i];
                actual_exact_hit_count++;
            }
        }
    }
    double avg_hit_latency_ms = (actual_hit_count > 0) ? total_hit_latency_ms / actual_hit_count : 0.0;
    double avg_exact_hit_latency_ms =
        (actual_exact_hit_count > 0) ? total_exact_hit_latency_ms / actual_exact_hit_count : 0.0;
    double avg_approx_hit_latency_ms =
        (actual_approx_hit_count > 0) ? total_approx_hit_latency_ms / actual_approx_hit_count : 0.0;
    double total_miss_penalty_ms = 0.0;
    size_t miss_count = 0;
    for (size_t i = 0; i < query_num; i++) {
        if (!hit_results[i]) {
            total_miss_penalty_ms += miss_penalties_ms[i];
            miss_count++;
        }
    }
    double avg_miss_penalty_ms = (miss_count > 0) ? total_miss_penalty_ms / miss_count : 0.0;
    double final_ratio = static_cast<double>(hit_count.load(std::memory_order_relaxed)) / query_num;
    auto global_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(global_end - global_start).count();
    double total_time_sec = total_time_ms / 1000.0;
    double avg_latency_ms = 0.0;
    for (double latency : latencies_ms) avg_latency_ms += latency;
    avg_latency_ms /= query_num;
    double qps = static_cast<double>(query_num) / total_time_sec;
    double qps_per_thread = qps / static_cast<double>(search_threads);
    std::vector<double> sorted_latencies = latencies_ms;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    auto get_percentile = [&](double p) {
        size_t idx = static_cast<size_t>(std::ceil(p * query_num)) - 1;
        if (idx >= query_num) idx = query_num - 1;
        return sorted_latencies[idx];
    };

    HybridMetrics metrics;
    metrics.hit_ratio = final_ratio;
    metrics.hits = hit_count.load(std::memory_order_relaxed);
    metrics.exact_hits = exact_hit_count.load(std::memory_order_relaxed);
    metrics.approx_hits = approx_hit_count.load(std::memory_order_relaxed);
    metrics.exact_hit_ratio = static_cast<double>(metrics.exact_hits) / static_cast<double>(query_num);
    metrics.approx_hit_ratio = static_cast<double>(metrics.approx_hits) / static_cast<double>(query_num);
    metrics.total_queries = query_num;
    metrics.threads = search_threads;
    metrics.avg_latency_ms = avg_latency_ms;
    metrics.avg_hit_latency_ms = avg_hit_latency_ms;
    metrics.avg_exact_hit_latency_ms = avg_exact_hit_latency_ms;
    metrics.avg_approx_hit_latency_ms = avg_approx_hit_latency_ms;
    metrics.avg_miss_penalty_ms = avg_miss_penalty_ms;
    metrics.qps = qps;
    metrics.qps_per_thread = qps_per_thread;
    metrics.memory_active_vectors = 0;
    metrics.memory_max_points = pool_size;
    metrics.pca_active_regions = 0;
    metrics.p50 = get_percentile(0.50);
    metrics.p90 = get_percentile(0.90);
    metrics.p95 = get_percentile(0.95);
    metrics.p99 = get_percentile(0.99);

    delete[] stats;
    return {hit_results, metrics};
}

template <typename T = float, typename TagT = uint32_t>
void experiment_benchmark(
    const std::string& query_path,
    const std::string& groundtruth_path,
    uint32_t K,
    uint32_t search_threads,
    int n_splits,
    int n_split_repeat,
    diskann::Metric metric,
    std::unique_ptr<qvcache::BackendInterface<T, TagT>> greator_backend,
    anns_cache_c_wrapper_t* cache,
    uint32_t slot_list_size,
    size_t pool_size,
    int window_size,
    int n_repeat,
    int stride,
    int n_round,
    int report_interval
) {
    TagT *groundtruth_ids = nullptr;
    float *groundtruth_dists = nullptr;
    size_t n_groundtruth, groundtruth_dim;
    greator::load_truthset(groundtruth_path, groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);
    size_t query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    greator::load_aligned_bin<T>(query_path, query, query_num, query_dim, query_aligned_dim);

    if (report_interval > 0) {
        const size_t interval = static_cast<size_t>(report_interval);
        size_t done = 0;
        size_t cum_hits = 0;
        double cum_latency_sum = 0.0;
        auto t0 = std::chrono::high_resolution_clock::now();
        spdlog::info("{{\"event\": \"stream_start\", \"total_queries\": {}, \"report_interval\": {}}}",
                     query_num, report_interval);
        while (done < query_num) {
            const size_t n = std::min(interval, query_num - done);
            std::vector<TagT> query_result_tags(n * K);
            auto [hit_results, hybrid_metrics] = hybrid_search(
                cache,
                *greator_backend,
                query + done * query_aligned_dim,
                n,
                static_cast<uint32_t>(query_aligned_dim),
                static_cast<uint32_t>(query_dim),
                K,
                slot_list_size,
                search_threads,
                query_result_tags,
                metric,
                pool_size
            );
            RecallAllMetrics recall_all = calculate_recall<T, TagT>(
                K, groundtruth_ids + done * groundtruth_dim,
                query_result_tags, n, groundtruth_dim);
            RecallHitMetrics recall_hits = calculate_hit_recall<T, TagT>(
                K, groundtruth_ids + done * groundtruth_dim,
                query_result_tags, hit_results, n, groundtruth_dim);
            cum_hits += hybrid_metrics.hits;
            cum_latency_sum += hybrid_metrics.avg_latency_ms * static_cast<double>(n);
            done += n;
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double elapsed_sec = elapsed_ms / 1000.0;
            double cum_hit_ratio = static_cast<double>(cum_hits) / static_cast<double>(done);
            double cum_avg_latency_ms = cum_latency_sum / static_cast<double>(done);
            double cum_qps = elapsed_sec > 0.0 ? static_cast<double>(done) / elapsed_sec : 0.0;
            log_stream_metrics<T, TagT>(
                done, query_num, done - n, done - 1,
                hybrid_metrics, recall_all, recall_hits,
                cum_hit_ratio, cum_hits, cum_avg_latency_ms, cum_qps, elapsed_ms);
        }
        spdlog::info("{{\"event\": \"stream_end\", \"total_queries\": {}}}", query_num);
        if (groundtruth_dists) delete[] groundtruth_dists;
        if (groundtruth_ids) delete[] groundtruth_ids;
        return;
    }

    int min_split_repeat = 1 + (window_size / stride) * n_repeat * n_round;
    if (n_split_repeat < min_split_repeat) {
        std::cerr << "Error: n_split_repeat (" << n_split_repeat << ") must be >= 1 + (window_size / stride) * n_repeat * n_round = "
                  << min_split_repeat << " (copy 0 skipped; each split visit needs a fresh perturbation)" << std::endl;
        exit(1);
    }

    size_t queries_per_original_split = query_num / (n_splits * n_split_repeat);
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> next_copy(static_cast<size_t>(n_splits), 1);

    int window_idx = 0;
    for (int round = 0; round < n_round; ++round) {
        spdlog::info("{{\"event\": \"round_start\", \"round\": {}}}", round);
        for (int window_start = 0; window_start + window_size <= n_splits; window_start += stride) {
            int window_end = window_start + window_size - 1;
            spdlog::info("{{\"event\": \"window_start\", \"window_idx\": {}, \"window_start_split\": {}, \"window_end_split\": {}}}",
                         window_idx, window_start, window_end);

        for (int repeat_idx = 0; repeat_idx < n_repeat; ++repeat_idx) {
            struct QueryInfo {
                size_t query_offset;
                size_t query_size;
                size_t gt_offset;
            };
            std::vector<QueryInfo> query_infos;

            for (int offset = 0; offset < window_size; ++offset) {
                int split_idx = window_start + offset;
                int copy_idx = next_copy[static_cast<size_t>(split_idx)]++;
                if (copy_idx >= n_split_repeat) {
                    std::cerr << "Error: split " << split_idx << " needs copy " << copy_idx
                              << " but n_split_repeat=" << n_split_repeat << std::endl;
                    exit(1);
                }
                size_t split_offset = split_idx * n_split_repeat * queries_per_original_split;
                size_t copy_offset = static_cast<size_t>(copy_idx) * queries_per_original_split;
                size_t query_start = split_offset + copy_offset;
                size_t query_end = std::min(query_start + queries_per_original_split, query_num);

                if (query_start < query_end) {
                    QueryInfo info;
                    info.query_offset = query_start;
                    info.query_size = query_end - query_start;
                    info.gt_offset = split_offset + copy_offset;
                    query_infos.push_back(info);
                }
            }

            std::shuffle(query_infos.begin(), query_infos.end(), g);

            size_t total_repeat_queries = 0;
            for (const auto& info : query_infos) {
                total_repeat_queries += info.query_size;
            }
            if (total_repeat_queries == 0) {
                continue;
            }

            std::vector<T> shuffled_queries(total_repeat_queries * query_aligned_dim);
            std::vector<TagT> shuffled_groundtruth(total_repeat_queries * groundtruth_dim);

            size_t current_idx = 0;
            for (const auto& info : query_infos) {
                for (size_t i = 0; i < info.query_size; ++i) {
                    std::memcpy(shuffled_queries.data() + current_idx * query_aligned_dim,
                               query + (info.query_offset + i) * query_aligned_dim,
                               query_aligned_dim * sizeof(T));
                    std::memcpy(shuffled_groundtruth.data() + current_idx * groundtruth_dim,
                               groundtruth_ids + (info.gt_offset + i) * groundtruth_dim,
                               groundtruth_dim * sizeof(TagT));
                    current_idx++;
                }
            }

            std::vector<TagT> query_result_tags(total_repeat_queries * K);
            auto [hit_results, hybrid_metrics] = hybrid_search(
                cache,
                *greator_backend,
                shuffled_queries.data(),
                total_repeat_queries,
                static_cast<uint32_t>(query_aligned_dim),
                static_cast<uint32_t>(query_dim),
                K,
                slot_list_size,
                search_threads,
                query_result_tags,
                metric,
                pool_size
            );

            RecallAllMetrics recall_all = calculate_recall<T, TagT>(
                K, shuffled_groundtruth.data(),
                query_result_tags, total_repeat_queries, groundtruth_dim);
            RecallHitMetrics recall_hits = calculate_hit_recall<T, TagT>(
                K, shuffled_groundtruth.data(),
                query_result_tags, hit_results, total_repeat_queries, groundtruth_dim);

            log_window_metrics<T, TagT>(window_idx, repeat_idx, hybrid_metrics, recall_all, recall_hits);
        }

            spdlog::info("{{\"event\": \"window_end\", \"window_idx\": {}}}", window_idx);
            window_idx++;
        }
        spdlog::info("{{\"event\": \"round_end\", \"round\": {}}}", round);
    }

    if (groundtruth_dists) delete[] groundtruth_dists;
    if (groundtruth_ids) delete[] groundtruth_ids;
}

int main(int argc, char **argv) {
    std::string data_type, data_path, query_path, groundtruth_path, disk_index_prefix;
    std::string aker_config_path = "Aker/bootstrap/aker-standard.ini";
    uint32_t R, disk_L, K, B, M;
    uint32_t build_threads, search_threads, beamwidth;
    int disk_index_already_built;
    uint32_t sector_len = 4096;
    size_t memory_index_max_points = 100000;
    size_t aker_pool_size = 0;
    uint32_t aker_top_delta = 5;
    int n_splits;
    int n_split_repeat;
    std::string metric_str = "l2";
    int window_size;
    int n_repeat;
    int stride;
    int n_round;
    int report_interval = 0;
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Print information on arguments")
            ("data_type", po::value<std::string>(&data_type)->required(), "Type of data")
            ("data_path", po::value<std::string>(&data_path)->required(), "Path to data")
            ("query_path", po::value<std::string>(&query_path)->required(), "Path to query")
            ("groundtruth_path", po::value<std::string>(&groundtruth_path)->required(), "Path to groundtruth")
            ("disk_index_prefix", po::value<std::string>(&disk_index_prefix)->required(), "Prefix to index")
            ("R", po::value<uint32_t>(&R)->required(), "Value of R")
            ("disk_L", po::value<uint32_t>(&disk_L)->required(), "Value of disk L")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K")
            ("B", po::value<uint32_t>(&B)->default_value(8), "Value of B")
            ("M", po::value<uint32_t>(&M)->default_value(8), "Value of M")
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching")
            ("disk_index_already_built", po::value<int>(&disk_index_already_built)->default_value(1), "Disk index already built (0/1)")
            ("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(2), "Beamwidth")
            ("sector_len", po::value<uint32_t>(&sector_len)->default_value(4096), "Sector length in bytes")
            ("memory_index_max_points", po::value<size_t>(&memory_index_max_points)->default_value(100000), "Used to size Aker pool if aker_pool_size is 0")
            ("aker_config", po::value<std::string>(&aker_config_path)->default_value("Aker/bootstrap/aker-standard.ini"), "Aker bootstrap INI/JSON path")
            ("aker_pool_size", po::value<size_t>(&aker_pool_size)->default_value(0), "Aker vector pool size (0 = auto)")
            ("aker_top_delta", po::value<uint32_t>(&aker_top_delta)->default_value(5), "Aker top_delta extra neighbors per entry")
            ("n_splits", po::value<int>(&n_splits)->required(), "Number of splits for queries")
            ("n_split_repeat", po::value<int>(&n_split_repeat)->required(), "Number of repeats per split pattern")
            ("metric", po::value<std::string>(&metric_str)->default_value("l2"), "Distance metric: l2, cosine, or inner_product")
            ("window_size", po::value<int>(&window_size)->required(), "Window size (number of splits per window)")
            ("n_repeat", po::value<int>(&n_repeat)->required(), "N_repeat (number of copies per split in window)")
            ("stride", po::value<int>(&stride)->required(), "Stride (step size for window advancement)")
            ("n_round", po::value<int>(&n_round)->default_value(1), "Number of times to cycle windows over splits (wrapping)")
            ("report_interval", po::value<int>(&report_interval)->default_value(0), "Aker: if >0, stream queries in file order and log metrics every N queries (skip windows)");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        aker_pin_openmp((int)search_threads);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (metric_str == "l2" || metric_str == "L2") {
        metric = diskann::Metric::L2;
    } else if (metric_str == "cosine" || metric_str == "COSINE") {
        metric = diskann::Metric::COSINE;
    } else if (metric_str == "inner_product" || metric_str == "INNER_PRODUCT" || metric_str == "innerproduct") {
        metric = diskann::Metric::INNER_PRODUCT;
    } else {
        std::cerr << "Unsupported metric: " << metric_str << ". Supported metrics: l2, cosine, inner_product" << std::endl;
        return -1;
    }

    {
        std::ifstream config_stream(aker_config_path);
        if (!config_stream.good()) {
            std::cerr << "Aker config not found: " << aker_config_path << std::endl;
            return -1;
        }
    }

    set_sector_len(sector_len);
    auto logger = spdlog::stdout_color_mt("console");
    spdlog::set_pattern("%v");

    uint32_t query_npts_hdr = 0, query_dim = 0;
    {
        std::ifstream qf(query_path, std::ios::binary);
        if (!qf) {
            std::cerr << "Failed to open query file: " << query_path << std::endl;
            return -1;
        }
        qf.read(reinterpret_cast<char*>(&query_npts_hdr), sizeof(uint32_t));
        qf.read(reinterpret_cast<char*>(&query_dim), sizeof(uint32_t));
        if (!qf) {
            std::cerr << "Failed to read query file header: " << query_path << std::endl;
            return -1;
        }
    }
    (void)query_npts_hdr;

    if (data_type != "float" && data_type != "int8" && data_type != "uint8") {
        std::cerr << "Unsupported data type: " << data_type << std::endl;
        return -1;
    }

    anns_cache_parameter_c_t parameter{};
    akerImportAnnsCacheConfig(const_cast<char*>(aker_config_path.c_str()), &parameter);
    parameter.vector_format.dimension = static_cast<uint32_t>(query_dim);
    if (data_type == "float")
        parameter.vector_format.vector_in_bytes = query_dim * sizeof(float);
    else
        parameter.vector_format.vector_in_bytes = query_dim * sizeof(uint8_t);
    parameter.capacity.in_topk = K;
    parameter.capacity.top_delta = aker_top_delta;
    const uint32_t slot_list_size = K + aker_top_delta;
    if (aker_pool_size == 0) {
        aker_pool_size = memory_index_max_points * static_cast<size_t>(slot_list_size + 1);
    }
    parameter.capacity.pool_size = aker_pool_size;
    if (metric == diskann::Metric::INNER_PRODUCT)
        parameter.distance_metric = 1;
    else
        parameter.distance_metric = 0;

    if (disk_L < slot_list_size) {
        std::cerr << "disk_L (" << disk_L << ") must be >= K + aker_top_delta (" << slot_list_size << ")" << std::endl;
        return -1;
    }

    anns_cache_c_wrapper_t* cache = akerCreateAnnsCache(parameter);
    if (cache == nullptr) {
        std::cerr << "Failed to create Aker cache" << std::endl;
        return -1;
    }

    logger->info("{{\n"
        "  \"event\": \"params\",\n"
        "  \"cache\": \"aker\",\n"
        "  \"data_type\": \"{}\",\n"
        "  \"data_path\": \"{}\",\n"
        "  \"query_path\": \"{}\",\n"
        "  \"groundtruth_path\": \"{}\",\n"
        "  \"disk_index_prefix\": \"{}\",\n"
        "  \"R\": {},\n"
        "  \"disk_L\": {},\n"
        "  \"K\": {},\n"
        "  \"B\": {},\n"
        "  \"M\": {},\n"
        "  \"build_threads\": {},\n"
        "  \"search_threads\": {},\n"
        "  \"disk_index_already_built\": {},\n"
        "  \"beamwidth\": {},\n"
        "  \"sector_len\": {},\n"
        "  \"n_splits\": {},\n"
        "  \"n_split_repeat\": {},\n"
        "  \"metric\": \"{}\",\n"
        "  \"window_size\": {},\n"
        "  \"n_repeat\": {},\n"
        "  \"stride\": {},\n"
        "  \"n_round\": {},\n"
        "  \"report_interval\": {},\n"
        "  \"aker_config\": \"{}\",\n"
        "  \"aker_pool_size\": {},\n"
        "  \"aker_top_delta\": {},\n"
        "  \"query_dim\": {}\n"
        "}}",
        data_type, data_path, query_path, groundtruth_path, disk_index_prefix,
        R, disk_L, K, B, M, build_threads, search_threads, disk_index_already_built, beamwidth,
        sector_len, n_splits, n_split_repeat, metric_str, window_size, n_repeat, stride, n_round, report_interval,
        aker_config_path, aker_pool_size, aker_top_delta, query_dim);

    if (data_type == "float") {
        std::unique_ptr<qvcache::BackendInterface<float, uint32_t>> greator_backend = std::make_unique<qvcache::GreatorBackend<float>>(
            data_path, disk_index_prefix, R, disk_L, B, M, build_threads, disk_index_already_built, beamwidth, metric);
        experiment_benchmark<float>(query_path, groundtruth_path, K, search_threads, n_splits, n_split_repeat, metric,
            std::move(greator_backend), cache, slot_list_size, aker_pool_size, window_size, n_repeat, stride, n_round, report_interval);
    } else if (data_type == "int8") {
        std::unique_ptr<qvcache::BackendInterface<int8_t, uint32_t>> greator_backend = std::make_unique<qvcache::GreatorBackend<int8_t>>(
            data_path, disk_index_prefix, R, disk_L, B, M, build_threads, disk_index_already_built, beamwidth, metric);
        experiment_benchmark<int8_t>(query_path, groundtruth_path, K, search_threads, n_splits, n_split_repeat, metric,
            std::move(greator_backend), cache, slot_list_size, aker_pool_size, window_size, n_repeat, stride, n_round, report_interval);
    } else if (data_type == "uint8") {
        std::unique_ptr<qvcache::BackendInterface<uint8_t, uint32_t>> greator_backend = std::make_unique<qvcache::GreatorBackend<uint8_t>>(
            data_path, disk_index_prefix, R, disk_L, B, M, build_threads, disk_index_already_built, beamwidth, metric);
        experiment_benchmark<uint8_t>(query_path, groundtruth_path, K, search_threads, n_splits, n_split_repeat, metric,
            std::move(greator_backend), cache, slot_list_size, aker_pool_size, window_size, n_repeat, stride, n_round, report_interval);
    }

    akerDestroyAnnsCache(cache);
    return 0;
}

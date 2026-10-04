// System headers
#include <cstddef>
#include <omp.h>
#include <boost/program_options.hpp>
#include <atomic>
#include <iomanip>
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
#include <numeric>
#include <cstring>

// Backend header
#include "pgvector_backend.h"

// DiskANN headers
#include "diskann/distance.h"

// Greator header for set_sector_len / QueryStats used by the shared harness
#include "greator/pq_flash_index.h"

// QVCache header
#include "qvcache/qvcache.h"

namespace po = boost::program_options;

// Metrics structures to match Python format
struct HybridMetrics {
    double hit_ratio;
    size_t hits;
    size_t total_queries;
    uint32_t threads;
    double avg_latency_ms;
    double avg_hit_latency_ms;
    double avg_miss_penalty_ms;
    double qps;
    double qps_per_thread;
    size_t memory_active_vectors;
    size_t memory_max_points;
    size_t pca_active_regions;
    std::map<size_t, size_t> index_vectors;  // index_id -> vector_count
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
    double recall_cache_hits;  // Use -1.0 to represent null
    size_t cache_hit_count;
};

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
        // Filter out invalid IDs (padded results)
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
    
    // Count queries with low recall
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
            // Filter out invalid IDs (padded results)
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
            total_recall += recall;
            hit_count++;
        }
    }
    
    RecallHitMetrics metrics;
    if (hit_count > 0) {
        metrics.recall_cache_hits = total_recall / hit_count;
    } else {
        metrics.recall_cache_hits = -1.0;  // Use -1.0 to represent null
    }
    metrics.cache_hit_count = hit_count;
    
    return metrics;
}

template <typename T, typename TagT = uint32_t>
void log_window_metrics(int window_idx, int repeat_idx, const HybridMetrics& hybrid_metrics, const RecallAllMetrics& recall_all_metrics, const RecallHitMetrics& recall_hit_metrics) {
    // Build mini index vector counts string
    std::string mini_index_counts = "";
    for (const auto& [idx, count] : hybrid_metrics.index_vectors) {
        if (!mini_index_counts.empty()) mini_index_counts += ", ";
        mini_index_counts += "\"index_" + std::to_string(idx) + "_vectors\": " + std::to_string(count);
    }
    
    // Log combined metrics in single JSON line (matching Python format)
    if (recall_hit_metrics.recall_cache_hits < 0) {
        // null case
        spdlog::info("{{\"event\": \"window_metrics\", "
                  "\"window_idx\": {}, "
                  "\"repeat_idx\": {}, "
                  "\"hit_ratio\": {}, "
                  "\"hits\": {}, "
                  "\"total_queries\": {}, "
                  "\"threads\": {}, "
                  "\"avg_latency_ms\": {}, "
                  "\"avg_hit_latency_ms\": {}, "
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
                  hybrid_metrics.total_queries,
                  hybrid_metrics.threads, hybrid_metrics.avg_latency_ms, hybrid_metrics.avg_hit_latency_ms,
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
                  "\"total_queries\": {}, "
                  "\"threads\": {}, "
                  "\"avg_latency_ms\": {}, "
                  "\"avg_hit_latency_ms\": {}, "
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
                  hybrid_metrics.total_queries,
                  hybrid_metrics.threads, hybrid_metrics.avg_latency_ms, hybrid_metrics.avg_hit_latency_ms,
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

template <typename T, typename TagT = uint32_t>
void log_stream_metrics(
    size_t completed, size_t total, size_t query_begin, size_t query_end,
    const HybridMetrics& interval, const RecallAllMetrics& recall_all,
    const RecallHitMetrics& recall_hits,
    double cum_hit_ratio, size_t cum_hits, double cum_avg_latency_ms, double cum_qps, double elapsed_ms
) {
    // Aker: search-workload progress (file order, no windows).
    std::string mini_index_counts = "";
    for (const auto& [idx, count] : interval.index_vectors) {
        if (!mini_index_counts.empty()) mini_index_counts += ", ";
        mini_index_counts += "\"index_" + std::to_string(idx) + "_vectors\": " + std::to_string(count);
    }
    if (mini_index_counts.empty())
        mini_index_counts = "\"index_vectors\": {}";

    if (recall_hits.recall_cache_hits < 0) {
        spdlog::info("{{\"event\": \"stream_metrics\", "
                  "\"completed\": {}, \"total\": {}, \"query_begin\": {}, \"query_end\": {}, "
                  "\"interval_queries\": {}, "
                  "\"hit_ratio\": {}, \"hits\": {}, "
                  "\"avg_latency_ms\": {}, \"avg_hit_latency_ms\": {}, "
                  "\"miss_penalty_ms\": {}, "
                  "\"qps\": {}, "
                  "\"memory_active_vectors\": {}, \"memory_max_points\": {}, \"pca_active_regions\": {}, "
                  "{}, "
                  "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                  "\"recall_all\": {}, \"K\": {}, "
                  "\"recall_cache_hits\": null, \"cache_hit_count\": {}, "
                  "\"cum_hit_ratio\": {}, \"cum_hits\": {}, "
                  "\"cum_avg_latency_ms\": {}, \"cum_qps\": {}, \"elapsed_ms\": {}}}",
                  completed, total, query_begin, query_end,
                  interval.total_queries,
                  interval.hit_ratio, interval.hits,
                  interval.avg_latency_ms, interval.avg_hit_latency_ms,
                  interval.avg_miss_penalty_ms,
                  interval.qps,
                  interval.memory_active_vectors, interval.memory_max_points, interval.pca_active_regions,
                  mini_index_counts,
                  interval.p50, interval.p90, interval.p95, interval.p99,
                  recall_all.recall_all, recall_all.K,
                  recall_hits.cache_hit_count,
                  cum_hit_ratio, cum_hits, cum_avg_latency_ms, cum_qps, elapsed_ms);
    } else {
        spdlog::info("{{\"event\": \"stream_metrics\", "
                  "\"completed\": {}, \"total\": {}, \"query_begin\": {}, \"query_end\": {}, "
                  "\"interval_queries\": {}, "
                  "\"hit_ratio\": {}, \"hits\": {}, "
                  "\"avg_latency_ms\": {}, \"avg_hit_latency_ms\": {}, "
                  "\"miss_penalty_ms\": {}, "
                  "\"qps\": {}, "
                  "\"memory_active_vectors\": {}, \"memory_max_points\": {}, \"pca_active_regions\": {}, "
                  "{}, "
                  "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                  "\"recall_all\": {}, \"K\": {}, "
                  "\"recall_cache_hits\": {}, \"cache_hit_count\": {}, "
                  "\"cum_hit_ratio\": {}, \"cum_hits\": {}, "
                  "\"cum_avg_latency_ms\": {}, \"cum_qps\": {}, \"elapsed_ms\": {}}}",
                  completed, total, query_begin, query_end,
                  interval.total_queries,
                  interval.hit_ratio, interval.hits,
                  interval.avg_latency_ms, interval.avg_hit_latency_ms,
                  interval.avg_miss_penalty_ms,
                  interval.qps,
                  interval.memory_active_vectors, interval.memory_max_points, interval.pca_active_regions,
                  mini_index_counts,
                  interval.p50, interval.p90, interval.p95, interval.p99,
                  recall_all.recall_all, recall_all.K,
                  recall_hits.recall_cache_hits, recall_hits.cache_hit_count,
                  cum_hit_ratio, cum_hits, cum_avg_latency_ms, cum_qps, elapsed_ms);
    }
}

template <typename T, typename TagT = uint32_t>
std::pair<std::vector<bool>, HybridMetrics> hybrid_search(
    qvcache::QVCache<T>& qvcache,
    const T* query, size_t query_num, uint32_t query_aligned_dim,
    uint32_t K, uint32_t L, uint32_t search_threads,
    std::vector<uint32_t>& query_result_tags, std::vector<T *>& res,
    uint32_t beamwidth, const std::string& data_path
) {
    std::vector<float> query_result_dists(K * query_num);
    greator::QueryStats* stats = new greator::QueryStats[query_num];
    std::vector<double> latencies_ms(query_num, 0.0);
    std::vector<double> miss_penalties_ms(query_num, 0.0);
    std::vector<bool> hit_results(query_num, false);

    // Pin the OpenMP budget to the configured search thread count, matching the
    // Aker benchmark so both systems are measured under the same thread budget.
    omp_set_num_threads((int32_t)search_threads);

    auto global_start = std::chrono::high_resolution_clock::now();
    std::atomic<size_t> hit_count{0};
    #pragma omp parallel for num_threads((int32_t)search_threads) schedule(dynamic)
    for (size_t i = 0; i < query_num; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        bool hit = qvcache.search(
            query + i * query_aligned_dim,
            K,
            query_result_tags.data() + i * K,
            res,
            query_result_dists.data() + i * K,
            stats + i
        );
        hit_results[i] = hit;
        if (hit) hit_count.fetch_add(1, std::memory_order_relaxed);
        else miss_penalties_ms[i] = qvcache.last_cache_lookup_ms();
        auto end = std::chrono::high_resolution_clock::now();
        latencies_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }
    double total_hit_latency_ms = 0.0;
    size_t actual_hit_count = 0;
    for (size_t i = 0; i < query_num; i++) {
        if (hit_results[i]) {
            total_hit_latency_ms += latencies_ms[i];
            actual_hit_count++;
        }
    }
    double avg_hit_latency_ms = (actual_hit_count > 0) ? total_hit_latency_ms / actual_hit_count : 0.0;
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
    double p50 = get_percentile(0.50);
    double p90 = get_percentile(0.90);
    double p95 = get_percentile(0.95);
    double p99 = get_percentile(0.99);
    
    // Build index vector counts map
    std::map<size_t, size_t> index_vectors;
    size_t num_mini_indexes = qvcache.get_number_of_mini_indexes();
    for (size_t i = 0; i < num_mini_indexes; ++i) {
        index_vectors[i] = qvcache.get_index_vector_count(i);
    }
    
    HybridMetrics metrics;
    metrics.hit_ratio = final_ratio;
    metrics.hits = hit_count.load(std::memory_order_relaxed);
    metrics.total_queries = query_num;
    metrics.threads = search_threads;
    metrics.avg_latency_ms = avg_latency_ms;
    metrics.avg_hit_latency_ms = avg_hit_latency_ms;
    metrics.avg_miss_penalty_ms = avg_miss_penalty_ms;
    metrics.qps = qps;
    metrics.qps_per_thread = qps_per_thread;
    metrics.memory_active_vectors = qvcache.get_number_of_vectors_in_memory_index();
    metrics.memory_max_points = qvcache.get_number_of_max_points_in_memory_index();
    metrics.pca_active_regions = qvcache.get_number_of_active_pca_regions();
    metrics.index_vectors = index_vectors;
    metrics.p50 = p50;
    metrics.p90 = p90;
    metrics.p95 = p95;
    metrics.p99 = p99;
    
    delete[] stats;
    return {hit_results, metrics};
}

// Main experiment logic for window-based benchmark
template <typename T = float, typename TagT = uint32_t>
void experiment_benchmark(
    const std::string& data_type,
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    const std::string& disk_index_prefix,
    uint32_t R, uint32_t memory_L, uint32_t K,
    uint32_t B, uint32_t M,
    float alpha,
    uint32_t build_threads,
    uint32_t search_threads,
    int disk_index_already_built,
    uint32_t beamwidth, 
    int use_reconstructed_vectors,
    double p,
    double deviation_factor,
    size_t memory_index_max_points,
    bool use_regional_theta,
    uint32_t pca_dim,
    uint32_t buckets_per_dim,
    size_t max_regions,
    int n_splits,
    int n_split_repeat,
    uint32_t n_async_insert_threads,
    bool lazy_theta_updates,
    size_t number_of_mini_indexes,
    size_t max_search_threads,
    const std::string& search_strategy,
    diskann::Metric metric,
    std::unique_ptr<qvcache::BackendInterface<T, TagT>> greator_backend,
    int window_size,
    int n_repeat,
    int stride,
    int n_round,
    int report_interval,
    bool learn_pca_from_queries
) {
    if (report_interval <= 0) {
        // Copy 0 is the unperturbed original. Each (split, window visit) uses the next
        // noisy copy so overlapping windows never replay the same vector.
        int min_split_repeat = 1 + (window_size / stride) * n_repeat * n_round;
        if (n_split_repeat < min_split_repeat) {
            std::cerr << "Error: n_split_repeat (" << n_split_repeat << ") must be >= 1 + (window_size / stride) * n_repeat * n_round = "
                      << min_split_repeat << " (copy 0 skipped; each split visit needs a fresh perturbation)" << std::endl;
            exit(1);
        }
    }
    
    qvcache::QVCache<T> qvcache(
       data_path, disk_index_prefix,
       R, memory_L, B, M, alpha, 
       build_threads, search_threads,
       (bool)use_reconstructed_vectors,
       p, deviation_factor,
       memory_index_max_points,
       beamwidth,
       use_regional_theta,
       pca_dim,
       buckets_per_dim,
       max_regions,
       n_async_insert_threads,
       lazy_theta_updates,
       number_of_mini_indexes,
       max_search_threads,
       metric,
       std::move(greator_backend),
       learn_pca_from_queries,
       query_path
    );

    // Set the search strategy
    if (search_strategy == "SEQUENTIAL") {
        qvcache.set_search_strategy(qvcache::QVCache<T>::SearchStrategy::SEQUENTIAL);
    } else if (search_strategy == "PARALLEL") {
        qvcache.set_search_strategy(qvcache::QVCache<T>::SearchStrategy::PARALLEL);
    } else {
        std::cerr << "Unknown search strategy: " << search_strategy
                  << " (expected SEQUENTIAL or PARALLEL)" << std::endl;
        exit(1);
    }

    TagT *groundtruth_ids = nullptr;
    float *groundtruth_dists = nullptr;
    size_t n_groundtruth, groundtruth_dim;
    diskann::load_truthset(groundtruth_path, groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);
    size_t query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    diskann::load_aligned_bin<T>(query_path, query, query_num, query_dim, query_aligned_dim);
    std::vector<T *> res = std::vector<T *>();

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
                qvcache,
                query + done * query_aligned_dim,
                n,
                query_aligned_dim,
                K,
                memory_L,
                search_threads,
                query_result_tags,
                res,
                beamwidth,
                data_path
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
    
    // Query file structure: all copies of split 0, then all copies of split 1, etc.
    // Each split has n_split_repeat copies, and each copy has queries_per_original_split queries
    size_t queries_per_original_split = query_num / (n_splits * n_split_repeat);
    
    // Random number generator for shuffling
    std::random_device rd;
    std::mt19937 g(rd());

    // Per-split cursor into noisy copies (start at 1: skip the unperturbed original).
    std::vector<int> next_copy(static_cast<size_t>(n_splits), 1);
    
    // Process windows in rounds; each round ends when the last split of a window reaches the last global split.
    int window_idx = 0;
    for (int round = 0; round < n_round; ++round) {
        spdlog::info("{{\"event\": \"round_start\", \"round\": {}}}", round);
        for (int window_start = 0; window_start + window_size <= n_splits; window_start += stride) {
            int window_end = window_start + window_size - 1;
            spdlog::info("{{\"event\": \"window_start\", \"window_idx\": {}, \"window_start_split\": {}, \"window_end_split\": {}}}", 
                         window_idx, window_start, window_end);
        
        // Process each repeat separately within this window
        for (int repeat_idx = 0; repeat_idx < n_repeat; ++repeat_idx) {
            // Collect queries from this repeat across all splits in the window
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
            
            // Shuffle query_infos to randomize order
            std::shuffle(query_infos.begin(), query_infos.end(), g);
            
            // Collect all queries and groundtruth in shuffled order
            size_t total_repeat_queries = 0;
            for (const auto& info : query_infos) {
                total_repeat_queries += info.query_size;
            }
            
            if (total_repeat_queries == 0) {
                continue;
            }
            
            // Allocate buffers for shuffled queries and groundtruth
            std::vector<T> shuffled_queries(total_repeat_queries * query_aligned_dim);
            std::vector<TagT> shuffled_groundtruth(total_repeat_queries * groundtruth_dim);
            
            size_t current_idx = 0;
            for (const auto& info : query_infos) {
                // Copy queries
                for (size_t i = 0; i < info.query_size; ++i) {
                    std::memcpy(shuffled_queries.data() + current_idx * query_aligned_dim,
                               query + (info.query_offset + i) * query_aligned_dim,
                               query_aligned_dim * sizeof(T));
                    // Copy groundtruth
                    std::memcpy(shuffled_groundtruth.data() + current_idx * groundtruth_dim,
                               groundtruth_ids + (info.gt_offset + i) * groundtruth_dim,
                               groundtruth_dim * sizeof(TagT));
                    current_idx++;
                }
            }
            
            // Perform search on shuffled queries
            std::vector<TagT> query_result_tags(total_repeat_queries * K);
            auto [hit_results, hybrid_metrics] = hybrid_search(
                qvcache,
                shuffled_queries.data(),
                total_repeat_queries,
                query_aligned_dim,
                K,
                memory_L,
                search_threads,
                query_result_tags,
                res,
                beamwidth,
                data_path
            );
            
            // Calculate recall using shuffled groundtruth
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
    uint32_t R, memory_L, disk_L, K, B, M;
    uint32_t build_threads, search_threads, beamwidth;
    float alpha;
    int single_file_index, disk_index_already_built, use_reconstructed_vectors;
    double hit_rate;
    double p, deviation_factor;
    uint32_t sector_len = 4096;
    bool use_regional_theta = true;
    uint32_t pca_dim, buckets_per_dim;
    size_t memory_index_max_points;
    size_t max_regions = std::numeric_limits<size_t>::max();
    int n_splits;
    int n_split_repeat;
    uint32_t n_async_insert_threads = 4;
    bool lazy_theta_updates = true;
    size_t number_of_mini_indexes = 2;
    size_t max_search_threads = 32;
    std::string search_strategy = "PARALLEL";
    std::string metric_str = "l2";
    int window_size;
    int n_repeat;
    int stride;
    int n_round;
    int report_interval = 0;
    std::string table_name = "spacev_1m";
    std::string db_host = "postgres";
    int db_port = 5432;
    std::string db_name = "postgres";
    std::string db_user = "postgres";
    std::string db_password = "postgres";
    int hnsw_ef_search = 200;
    int learn_pca_from_queries = 0;
    po::options_description desc;
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Print information on arguments")
            ("data_type", po::value<std::string>(&data_type)->required(), "Type of data")
            ("data_path", po::value<std::string>(&data_path)->required(), "Path to data")
            ("query_path", po::value<std::string>(&query_path)->required(), "Path to query")
            ("groundtruth_path", po::value<std::string>(&groundtruth_path)->required(), "Path to groundtruth")
            ("disk_index_prefix", po::value<std::string>(&disk_index_prefix)->required(), "Prefix for PCA / QVCache files")
            ("table_name", po::value<std::string>(&table_name)->default_value("spacev_1m"), "PostgreSQL table with pgvector HNSW")
            ("db_host", po::value<std::string>(&db_host)->default_value("postgres"), "PostgreSQL host")
            ("db_port", po::value<int>(&db_port)->default_value(5432), "PostgreSQL port")
            ("db_name", po::value<std::string>(&db_name)->default_value("postgres"), "PostgreSQL database")
            ("db_user", po::value<std::string>(&db_user)->default_value("postgres"), "PostgreSQL user")
            ("db_password", po::value<std::string>(&db_password)->default_value("postgres"), "PostgreSQL password")
            ("hnsw_ef_search", po::value<int>(&hnsw_ef_search)->default_value(200), "pgvector hnsw.ef_search (higher = better recall, slower)")
            ("R", po::value<uint32_t>(&R)->required(), "Value of R")
            ("memory_L", po::value<uint32_t>(&memory_L)->required(), "Value of memory L")
            ("disk_L", po::value<uint32_t>(&disk_L)->required(), "Value of disk L")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K")
            ("B", po::value<uint32_t>(&B)->default_value(8), "Value of B")
            ("M", po::value<uint32_t>(&M)->default_value(8), "Value of M")
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching")
            ("alpha", po::value<float>(&alpha)->required(), "Alpha parameter")
            ("use_reconstructed_vectors", po::value<int>(&use_reconstructed_vectors)->default_value(true), "Use reconstructed vectors for insertion to memory index")
            ("disk_index_already_built", po::value<int>(&disk_index_already_built)->default_value(1), "Disk index already built (0/1)")
            ("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(2), "Beamwidth")
            ("p", po::value<double>(&p)->default_value(0.75), "Value of p")
            ("deviation_factor", po::value<double>(&deviation_factor)->default_value(0.05), "Value of deviation factor")
            ("sector_len", po::value<uint32_t>(&sector_len)->default_value(4096), "Sector length in bytes")
            ("use_regional_theta", po::value<bool>(&use_regional_theta)->default_value(true), "Use regional theta (true) or global theta (false)")
            ("pca_dim", po::value<uint32_t>(&pca_dim)->required(), "Value of PCA dimension")
            ("buckets_per_dim", po::value<uint32_t>(&buckets_per_dim)->required(), "Value of buckets per dimension")
            ("memory_index_max_points", po::value<size_t>(&memory_index_max_points)->required(), "Max points for memory index")
            ("max_regions", po::value<size_t>(&max_regions)->default_value(std::numeric_limits<size_t>::max()), "Maximum number of regions for regional theta (default: unlimited)")
            ("n_splits", po::value<int>(&n_splits)->required(), "Number of splits for queries")
            ("n_split_repeat", po::value<int>(&n_split_repeat)->required(), "Number of repeats per split pattern")
            ("n_async_insert_threads", po::value<uint32_t>(&n_async_insert_threads)->default_value(4), "Number of async insert threads")
            ("lazy_theta_updates", po::value<bool>(&lazy_theta_updates)->default_value(true), "Enable lazy theta updates (true) or immediate updates (false)")
            ("number_of_mini_indexes", po::value<size_t>(&number_of_mini_indexes)->default_value(2), "Number of mini indexes for shadow cycling")
            ("max_search_threads", po::value<size_t>(&max_search_threads)->default_value(32), "Maximum threads for parallel search")
            ("search_strategy", po::value<std::string>(&search_strategy)->default_value("PARALLEL"), "Search strategy: SEQUENTIAL or PARALLEL")
            ("metric", po::value<std::string>(&metric_str)->default_value("l2"), "Distance metric: l2, cosine, or inner_product")
            ("window_size", po::value<int>(&window_size)->required(), "Window size (number of splits per window)")
            ("n_repeat", po::value<int>(&n_repeat)->required(), "N_repeat (number of copies per split in window)")
            ("stride", po::value<int>(&stride)->required(), "Stride (step size for window advancement)")
            ("n_round", po::value<int>(&n_round)->default_value(1), "Number of times to cycle windows over splits (wrapping)")
            ("report_interval", po::value<int>(&report_interval)->default_value(0), "Aker: if >0, stream queries in file order and log metrics every N queries (skip windows)")
            ("learn_pca_from_queries", po::value<int>(&learn_pca_from_queries)->default_value(0), "Fit regional PCA on the query file (1) instead of sampled data vectors (0)");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }
    // Parse metric string to enum
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
    
    set_sector_len(sector_len);
    auto logger = spdlog::stdout_color_mt("console");
    spdlog::set_pattern("%v");
    logger->info("{{\n"
        "  \"event\": \"params\",\n"
        "  \"data_type\": \"{}\",\n"
        "  \"data_path\": \"{}\",\n"
        "  \"query_path\": \"{}\",\n"
        "  \"groundtruth_path\": \"{}\",\n"
        "  \"disk_index_prefix\": \"{}\",\n"
        "  \"table_name\": \"{}\",\n"
        "  \"hnsw_ef_search\": {},\n"
        "  \"db_host\": \"{}\",\n"
        "  \"db_port\": {},\n"
        "  \"R\": {},\n"
        "  \"memory_L\": {},\n"
        "  \"disk_L\": {},\n"
        "  \"K\": {},\n"
        "  \"B\": {},\n"
        "  \"M\": {},\n"
        "  \"build_threads\": {},\n"
        "  \"search_threads\": {},\n"
        "  \"alpha\": {},\n"
        "  \"use_reconstructed_vectors\": {},\n"
        "  \"disk_index_already_built\": {},\n"
        "  \"beamwidth\": {},\n"
        "  \"p\": {},\n"
        "  \"deviation_factor\": {},\n"
        "  \"sector_len\": {},\n"
        "  \"use_regional_theta\": {},\n"
        "  \"pca_dim\": {},\n"
        "  \"buckets_per_dim\": {},\n"
        "  \"memory_index_max_points\": {},\n"
        "  \"max_regions\": {},\n"
        "  \"n_splits\": {},\n"
        "  \"n_split_repeat\": {},\n"
        "  \"n_async_insert_threads\": {},\n"
        "  \"lazy_theta_updates\": {},\n"
        "  \"number_of_mini_indexes\": {},\n"
        "  \"max_search_threads\": {},\n"
        "  \"search_strategy\": \"{}\",\n"
        "  \"metric\": \"{}\",\n"
        "  \"window_size\": {},\n"
        "  \"n_repeat\": {},\n"
        "  \"stride\": {},\n"
        "  \"n_round\": {},\n"
        "  \"report_interval\": {},\n"
        "  \"learn_pca_from_queries\": {}\n"
        "}}",
        data_type, data_path, query_path, groundtruth_path, disk_index_prefix, table_name, hnsw_ef_search, db_host, db_port, R, memory_L, disk_L, K, B, M, build_threads, search_threads, alpha, use_reconstructed_vectors, disk_index_already_built, beamwidth, p, deviation_factor, sector_len, use_regional_theta, pca_dim, buckets_per_dim, memory_index_max_points, max_regions, n_splits, n_split_repeat, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, max_search_threads, search_strategy, metric_str, window_size, n_repeat, stride, n_round, report_interval, learn_pca_from_queries);
    if (data_type == "float") {
        std::unique_ptr<qvcache::BackendInterface<float, uint32_t>> pg_backend = std::make_unique<qvcache::PgVectorBackend<float>>(
            table_name, data_path, db_host, db_port, db_name, db_user, db_password, metric_str, hnsw_ef_search);
        experiment_benchmark<float>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, memory_L, K, B, M, alpha, build_threads, search_threads, disk_index_already_built, beamwidth, use_reconstructed_vectors, p, deviation_factor, memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, max_regions, n_splits, n_split_repeat, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, max_search_threads, search_strategy, metric, std::move(pg_backend), window_size, n_repeat, stride, n_round, report_interval, (bool)learn_pca_from_queries);
    } else if (data_type == "int8") {
        std::unique_ptr<qvcache::BackendInterface<int8_t, uint32_t>> pg_backend = std::make_unique<qvcache::PgVectorBackend<int8_t>>(
            table_name, data_path, db_host, db_port, db_name, db_user, db_password, metric_str, hnsw_ef_search);
        experiment_benchmark<int8_t>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, memory_L, K, B, M, alpha, build_threads, search_threads, disk_index_already_built, beamwidth, use_reconstructed_vectors, p, deviation_factor, memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, max_regions, n_splits, n_split_repeat, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, max_search_threads, search_strategy, metric, std::move(pg_backend), window_size, n_repeat, stride, n_round, report_interval, (bool)learn_pca_from_queries);
    } else if (data_type == "uint8") {
        std::unique_ptr<qvcache::BackendInterface<uint8_t, uint32_t>> pg_backend = std::make_unique<qvcache::PgVectorBackend<uint8_t>>(
            table_name, data_path, db_host, db_port, db_name, db_user, db_password, metric_str, hnsw_ef_search);
        experiment_benchmark<uint8_t>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, memory_L, K, B, M, alpha, build_threads, search_threads, disk_index_already_built, beamwidth, use_reconstructed_vectors, p, deviation_factor, memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, max_regions, n_splits, n_split_repeat, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, max_search_threads, search_strategy, metric, std::move(pg_backend), window_size, n_repeat, stride, n_round, report_interval, (bool)learn_pca_from_queries);
    } else {
        std::cerr << "Unsupported data type: " << data_type << std::endl;
    }
    return 0;
}


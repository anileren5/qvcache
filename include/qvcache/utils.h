#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <set>
#include <spdlog/spdlog.h>
#include <string>
#include <limits>
#include "diskann/utils.h"

// Metrics structures for structured logging
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

template <typename T>
void load_aligned_binary_data(const std::string& file_path, T*& data, size_t& num, size_t& dim, size_t& aligned_dim) {
    diskann::load_aligned_bin<T>(file_path, data, num, dim, aligned_dim);
}

template <typename TagT>
void load_ground_truth_data(const std::string& file_path, TagT*& ids, float*& dists, size_t& num, size_t& dim) {
    diskann::load_truthset(file_path, ids, dists, num, dim);
}

#endif 
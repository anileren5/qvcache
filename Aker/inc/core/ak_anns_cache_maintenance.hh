#pragma once

#include <vector>

#include "ak_approx_filter.hh"
#include "core/ak_anns_cache_context.hh"

namespace aker
{
    class ANNSCacheEntryStore;
    class ANNSCacheSimilarityEngine;

    /**
     * @brief Maintenance module for ANNSCache.
     *
     * This module contains insertion/eviction and write-log maintenance paths.
     */
    class ANNSCacheMaintenance
    {
    public:
        /**
         * @brief Constructs the module with the shared cache context.
         */
        ANNSCacheMaintenance(ANNSCacheContext* context,
                           ANNSCacheEntryStore* entry_store,
                           ANNSCacheSimilarityEngine* similarity_engine) noexcept;

        /**
         * @brief Inserts a prepared entry into the cache.
         */
        bool insertCacheEntry(
            vector_id_t vector_id,
            anns_cache_entry_t* entry,
            vector_view_t query_vector_data) noexcept;

        /**
         * @brief Consumes aged write-log entries and triggers slow-path updates.
         */
        void processWriteLogEntries(
            distance_function_t distance_function,
            result_transform_callback_t result_transform_callback) noexcept;

        /**
         * @brief Inserts a write-log entry and runs fast/slow maintenance.
         */
        void insertWriteLogEntry(
            vector_view_t write_vector,
            distance_function_t distance_function,
            result_transform_callback_t result_transform_callback,
            const float* write_vector_float) noexcept;

        /**
         * @brief Marks a vector as deleted in the vector pool.
         */
        void markVectorDeleted(vector_id_t vector_id) noexcept;

        /**
         * @brief Clears cache state.
         */
        void resetCache() noexcept;

        /**
         * @brief Stress test helper that invalidates a random portion of pooled vectors.
         */
        void stressTestInvalidateRandom(float percent) noexcept;

        /**
         * @brief Collects pooled vectors.
         */
        void collectPooledVectors(std::vector<VectorSlot*>& pooled_list) noexcept;

    private:
        /**
         * @brief Checks if eviction is required for the incoming list size.
         */
        bool needEvict(size_t vector_list_size) noexcept;

        /**
         * @brief Evicts entries using the configured eviction strategy.
         */
        void evictCacheEntries(size_t to_evicts, std::vector<vector_id_t>& evicted_list) noexcept;

        /**
         * @brief Fast-path write-log update that opportunistically improves a nearby entry.
         */
        void updateWriteLogFastPath(
            vector_view_t write_vector,
            distance_function_t distance_function,
            result_transform_callback_t result_transform_callback,
            const float* write_vector_float) noexcept;

        /**
         * @brief Slow-path write-log update (sweep + top-k refresh).
         */
        /**
         * @brief Potluck global threshold tuning at put().
         */
        void tuneGlobalThresholdAtPut(
            anns_cache_entry_t* allocated_entry,
            vector_view_t query_vector_data) noexcept;

        void runWriteLogSlowPath(
            distance_function_t distance_function,
            result_transform_callback_t result_transform_callback) noexcept;

        ANNSCacheContext*    context_;
        ANNSCacheEntryStore* entry_store_;
        ANNSCacheSimilarityEngine* similarity_engine_;
    };
}

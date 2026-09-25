#pragma once

#include <cstddef>
#include <cstdint>

#include <memory>
#include <string>
#include <vector>

#include "ak_approx_filter.hh"
#include "core/ak_anns_cache_stats.hh"
#include "core/ak_anns_cache_types.hh"

namespace aker
{
    // Forward declarations of internal modules.
    class ANNSCacheEntryStore;
    class ANNSCacheSimilarityEngine;
    class ANNSCacheMaintenance;
    class ANNSCacheTelemetry;

    // Forward declaration for the internal shared context.
    struct ANNSCacheContext;

    /**
     * @brief Geometry-aware result cache sitting on top of an ANNS index / external DB.
     *
     * This class exposes a stable public API while delegating implementation details
     * to internal modules (EntryStore / SimilarityEngine / Maintenance / Telemetry).
     *
     * Lock policy: public API acquires the global cache lock and invokes internal
     * module methods that assume the lock is held. This refactor keeps the lock policy unchanged.
     */
    class ANNSCache
    {
    public:
        /**
         * @brief Constructs a cache instance with the provided parameter set.
         */
        explicit ANNSCache(anns_cache_config_t& parameter_info) noexcept;

        /**
         * @brief Destructor. Exports a final telemetry snapshot if the cache was used.
         */
        virtual ~ANNSCache();

        ANNSCache(const ANNSCache&) = delete;
        ANNSCache& operator=(const ANNSCache&) = delete;

        /**
         * @brief Creates a prepared cache entry before insertion.
         */
        anns_cache_entry_t* createCacheEntry(
            VectorSlot* query_vector,
            std::uint32_t list_size,
            VectorSlot** local_neighbors_list) noexcept;

        /**
         * @brief Legacy name for createCacheEntry().
         */
        [[deprecated("use createCacheEntry()")]]
        anns_cache_entry_t* makeCEntry(
            VectorSlot* query_vector,
            std::uint32_t list_size,
            VectorSlot** local_neighbors_list) noexcept
        {
            return createCacheEntry(query_vector, list_size, local_neighbors_list);
        }

        /**
         * @brief Frees an externally-held cache entry.
         */
        void destroyCacheEntry(anns_cache_entry_t* entry) noexcept;

        /**
         * @brief Legacy name for destroyCacheEntry().
         */
        [[deprecated("use destroyCacheEntry()")]]
        void freeCEntry(anns_cache_entry_t* entry) noexcept
        {
            destroyCacheEntry(entry);
        }

        /**
         * @brief Looks up a cache entry by exact id first, then by approximate similarity.
         */
        anns_cache_entry_t* getCacheEntry(
            vector_view_t query_vector_data,
            bool& similar_entry,
            bool& is_invalid,
            distance_function_t distance_function) noexcept;

        /**
         * @brief Legacy name for getCacheEntry().
         */
        [[deprecated("use getCacheEntry()")]]
        anns_cache_entry_t* simGetCEntry(
            vector_view_t query_vector_data,
            bool& similar_entry,
            bool& is_invalid,
            distance_function_t distance_function) noexcept
        {
            return getCacheEntry(query_vector_data, similar_entry, is_invalid, distance_function);
        }

        /**
         * @brief Inserts a prepared entry into the cache.
         */
        bool insertCacheEntry(
            vector_id_t vector_id,
            anns_cache_entry_t* entry,
            vector_view_t query_vector_data) noexcept;

        /**
         * @brief Legacy name for insertCacheEntry().
         */
        [[deprecated("use insertCacheEntry()")]]
        bool insertCEntry2(
            vector_id_t vector_id,
            anns_cache_entry_t* entry,
            vector_view_t query_vector_data) noexcept
        {
            return insertCacheEntry(vector_id, entry, query_vector_data);
        }

        /**
         * @brief Links an allocated dummy entry to an existing root entry.
         */
        bool linkCacheEntry(anns_cache_entry_t* allocated_entry, vector_id_t found_id) noexcept;

        /**
         * @brief Legacy name for linkCacheEntry().
         */
        [[deprecated("use linkCacheEntry()")]]
        bool linkCEntry(anns_cache_entry_t* allocated_entry, vector_id_t found_id) noexcept
        {
            return linkCacheEntry(allocated_entry, found_id);
        }

        /**
         * @brief Marks a vector as deleted in the vector pool.
         */
        void markVectorDeleted(vector_id_t vector_id) noexcept;

        /**
         * @brief Inserts a write-log entry to reflect an underlying DB mutation.
         */
        void insertWriteLogEntry(
            vector_view_t write_vector,
            distance_function_t distance_function,
            result_transform_callback_t result_transform_callback = nullptr) noexcept;

        /**
         * @brief Legacy name for insertWriteLogEntry().
         */
        [[deprecated("use insertWriteLogEntry()")]]
        void insertWLEntry3(
            vector_view_t write_vector,
            distance_function_t distance_function,
            result_transform_callback_t result_transform_callback = nullptr) noexcept
        {
            insertWriteLogEntry(write_vector, distance_function, result_transform_callback);
        }

        /**
         * @brief Processes aged write-log entries and performs maintenance work.
         */
        void processWriteLogEntries(
            distance_function_t distance_function,
            result_transform_callback_t result_transform_callback = nullptr) noexcept;

        /**
         * @brief Legacy name for processWriteLogEntries().
         */
        [[deprecated("use processWriteLogEntries()")]]
        void consumeAgedWLEntry(
            distance_function_t distance_function,
            result_transform_callback_t result_transform_callback = nullptr) noexcept
        {
            processWriteLogEntries(distance_function, result_transform_callback);
        }

        /**
         * @brief Clears cache state.
         */
        void resetCache() noexcept;

        /**
         * @brief Stress test helper that invalidates a random portion of pooled vectors.
         */
        void stressTestInvalidateRandom(float percent) noexcept;

        /**
         * @brief Collects pooled vectors into a provided list.
         */
        void collectPooledVectors(std::vector<VectorSlot*>& pooled_list) noexcept;

        /**
         * @brief Returns a human-readable status string.
         */
        std::string getStatusText() noexcept;

        /**
         * @brief Returns a concise CSV summary (key,value).
         */
        std::string getSummaryCsv() noexcept;

        /**
         * @brief Exports telemetry traces under /tmp/aker_trace_<timestamp>/.
         */
        void exportTraceToFiles() noexcept;

    private:
        std::unique_ptr<ANNSCacheContext> context_;

        std::unique_ptr<ANNSCacheEntryStore> entry_store_;
        std::unique_ptr<ANNSCacheSimilarityEngine> similarity_engine_;
        std::unique_ptr<ANNSCacheMaintenance> maintenance_;
        std::unique_ptr<ANNSCacheTelemetry> telemetry_;
    };
}

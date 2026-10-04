#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <string>
#include <vector>

#include "ak_timer.hh"
#include "utils/ak_spin_mutex.hh"

namespace aker
{
    /**
     * @brief Telemetry and latency statistics for ANNSCache.
     *
     * This module is intentionally designed for low overhead on the hot path:
     * - Latency series are indexed by a fixed enum instead of a string-keyed map.
     * - Export walks the static key list and writes one CSV per key.
     */
    class ANNSCacheStats
    {
    public:
        /**
         * @brief Per-request cache history sample.
         */
        struct CacheHistorySample
        {
            std::uint64_t sequence;
            const char* op_name;
            double latency_ms;

            size_t cache_entry_count;
            size_t vector_pool_size;
            size_t approx_repr_count;

            size_t cache_hit;
            size_t cache_sim_hit;
            size_t cache_miss;
            size_t cache_evict;

            float exact_hit_ratio;
            float total_hit_ratio;
        };

        /**
         * @brief Identifier for latency measurement series.
         */
        enum class LatencyMetric : std::uint32_t
        {
            k_get_cache_entry = 0,
            k_insert_cache_entry,
            k_link_cache_entry,
            k_insert_write_log_entry,
            k_mark_vector_deleted,
            k_process_write_log_entries,

            k_get_cache_entry_step_1,
            k_get_cache_entry_step_2,
            k_get_cache_entry_step_3,

            k_insert_cache_entry_step_1,
            k_insert_cache_entry_step_2,
            k_insert_cache_entry_step_3,

            k_insert_write_log_entry_step_1,
            k_insert_write_log_entry_step_2,
            k_insert_write_log_entry_step_3,
            k_insert_write_log_entry_step_4,

            k_evict_cache_entries_step_1,
            k_evict_cache_entries_step_2,
            k_evict_cache_entries_step_3,

            k_write_log_slow_path_step_1,
            k_write_log_slow_path_step_2,
            k_write_log_slow_path_step_3,
            k_write_log_slow_path_step_4,

            k_count,
        };

        /**
         * @brief One latency series (key + samples).
         */
        struct LatencySeries
        {
            const char* key;
            std::vector<ElapsedLatencyPair> samples;
        };

        /**
         * @brief Default reserve size to avoid frequent reallocations.
         */
        static constexpr std::size_t k_default_latency_reserve = 100000;

        /**
         * @brief Default reserve size for per-request cache history.
         */
        static constexpr std::size_t k_default_history_reserve = 100000;

        /**
         * @brief Protects all telemetry buffers in this class.
         */
        SpinMutex stats_lock;

        /**
         * @brief Lifetime counters.
         */
        size_t cache_hit{0};
        size_t cache_miss{0};
        size_t cache_invalid_detect{0};
        size_t cache_evict{0};
        size_t cache_sim_hit{0};

        /**
         * @brief Potluck-only counter: number of dropped sim-hit requests.
         */
        size_t cache_dropout{0};

        /**
         * @brief Latency series indexed by LatencyMetric.
         */
        std::array<LatencySeries, static_cast<std::size_t>(LatencyMetric::k_count)> latency_series;

        /**
         * @brief Hit ratio history buffers.
         */
        std::vector<float> cache_total_hit_ratios;
        std::vector<float> cache_exact_hit_ratios;

        /**
         * @brief Approx filter history buffers.
         */
        std::vector<size_t> approx_added_counts;
        std::vector<size_t> approx_representative_counts;

        /**
         * @brief Potluck global threshold tuning history.
         */
        std::vector<float> global_thresh_history;

        /**
         * @brief Per-request cache history samples.
         */
        std::vector<CacheHistorySample> cache_history;

        /**
         * @brief Monotonic sequence number for cache_history.
         */
        std::uint64_t history_sequence{0};

        /**
         * @brief Constructs the stats object and initializes all series.
         */
        ANNSCacheStats() noexcept;

        /**
         * @brief Destructor.
         */
        virtual ~ANNSCacheStats() noexcept = default;

        /**
         * @brief Clears all telemetry counters and buffers.
         */
        virtual void clear() noexcept;

        /**
         * @brief Records hit ratios into history buffers.
         */
        virtual void recordHitHistory() noexcept;

        /**
         * @brief Records a per-request cache status sample.
         */
        virtual void recordCacheHistorySample(
            const char* op_name,
            double latency_ms,
            size_t cache_entry_count,
            size_t vector_pool_size,
            size_t approx_repr_count) noexcept;

        /**
         * @brief Appends one latency sample into a series.
         */
        virtual void appendLatencySample(LatencyMetric metric, const ElapsedLatencyPair& sample) noexcept;

        /**
         * @brief Returns the latency key string for a metric.
         */
        virtual const char* getLatencyKey(LatencyMetric metric) const noexcept;

        /**
         * @brief Exports all telemetry files under /tmp/aker_trace_<timestamp>_pid<PID>_gen<G>/.
         *
         * @return The created trace directory path.
         */
        virtual std::string exportTraceToFiles() noexcept;

        /**
         * @brief Returns the last exported trace directory path.
         */
        virtual const std::string& getTraceDirectoryPath() const noexcept;

        /**
         * @brief Prints all telemetry to stdout.
         */
        virtual void printAll() noexcept;

    private:
        std::string trace_directory_path_;

        /**
         * @brief Creates a new trace directory path based on current time.
         */
        static std::string makeTraceDirectoryPath() noexcept;

        /**
         * @brief Ensures a directory exists.
         */
        static bool ensureDirectoryExists(const std::string& path) noexcept;

        /**
         * @brief Exports latency series and derived summary.
         */
        void exportLatencySeries(const std::string& directory_path) noexcept;

        /**
         * @brief Exports per-request cache history samples.
         */
        void exportCacheHistory(const std::string& directory_path) noexcept;

        /**
         * @brief Exports hit ratio and approx filter histories.
         */
        void exportDerivedHistories(const std::string& directory_path) noexcept;
    };

    /**
     * @brief Legacy alias used across the cache layer.
     */
    using anns_cache_stats_t = ANNSCacheStats;
}

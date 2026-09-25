#include "core/ak_anns_cache_stats.hh"

#include <atomic>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

#include <sys/stat.h>

#include <unistd.h>

#include "ak_logger.hh"

namespace aker
{
    namespace
    {
        /**
         * @brief Prefix for all trace output directories.
         */
        static constexpr const char* k_trace_directory_prefix = "/tmp/aker_trace_";

        /**
         * @brief Prefix for each per-metric latency file.
         */
        static constexpr const char* k_trace_file_prefix = "aker_trace_";

        /**
         * @brief File name for derived latency summary.
         */
        static constexpr const char* k_latency_summary_file = "aker_trace_latency_summary.csv";

        /**
         * @brief File name for per-request cache history.
         */
        static constexpr const char* k_cache_history_file = "aker_trace_cache_history.csv";

        /**
         * @brief File name for hit ratio history.
         */
        static constexpr const char* k_hit_ratio_history_file = "aker_trace_hit_ratio_history.csv";

        /**
         * @brief File name for approximate filter history.
         */
        static constexpr const char* k_approx_filter_history_file = "aker_trace_approx_filter_history.csv";

        /**
         * @brief Safe percentile accessor for a sorted vector.
         */
        double getPercentile(const std::vector<double>& sorted_values, double percentile)
        {
            if (sorted_values.empty())
            {
                return 0.0;
            }

            const double clamped = std::min(1.0, std::max(0.0, percentile));
            const double rank = clamped * static_cast<double>(sorted_values.size() - 1);
            const std::size_t idx = static_cast<std::size_t>(std::llround(rank));
            return sorted_values[std::min(idx, sorted_values.size() - 1)];
        }
    }

    ANNSCacheStats::ANNSCacheStats() noexcept
    {
        /* Initialize latency keys and reserve vectors to minimize runtime allocations.
         */
        latency_series[static_cast<std::size_t>(LatencyMetric::k_get_cache_entry)] = {"getCacheEntry", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_insert_cache_entry)] = {"insertCacheEntry", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_link_cache_entry)] = {"linkCacheEntry", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_insert_write_log_entry)] = {"insertWriteLogEntry", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_mark_vector_deleted)] = {"markVectorDeleted", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_process_write_log_entries)] = {"processWriteLogEntries", {}};

        latency_series[static_cast<std::size_t>(LatencyMetric::k_get_cache_entry_step_1)] = {"getCacheEntry.step1", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_get_cache_entry_step_2)] = {"getCacheEntry.step2", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_get_cache_entry_step_3)] = {"getCacheEntry.step3", {}};

        latency_series[static_cast<std::size_t>(LatencyMetric::k_insert_cache_entry_step_1)] = {"insertCacheEntry.step1", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_insert_cache_entry_step_2)] = {"insertCacheEntry.step2", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_insert_cache_entry_step_3)] = {"insertCacheEntry.step3", {}};

        latency_series[static_cast<std::size_t>(LatencyMetric::k_insert_write_log_entry_step_1)] = {"insertWriteLogEntry.step1", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_insert_write_log_entry_step_2)] = {"insertWriteLogEntry.step2", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_insert_write_log_entry_step_3)] = {"insertWriteLogEntry.step3", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_insert_write_log_entry_step_4)] = {"insertWriteLogEntry.step4", {}};

        latency_series[static_cast<std::size_t>(LatencyMetric::k_evict_cache_entries_step_1)] = {"evictCacheEntries.step1", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_evict_cache_entries_step_2)] = {"evictCacheEntries.step2", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_evict_cache_entries_step_3)] = {"evictCacheEntries.step3", {}};

        latency_series[static_cast<std::size_t>(LatencyMetric::k_write_log_slow_path_step_1)] = {"runWriteLogSlowPath.step1", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_write_log_slow_path_step_2)] = {"runWriteLogSlowPath.step2", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_write_log_slow_path_step_3)] = {"runWriteLogSlowPath.step3", {}};
        latency_series[static_cast<std::size_t>(LatencyMetric::k_write_log_slow_path_step_4)] = {"runWriteLogSlowPath.step4", {}};

        for (LatencySeries& series : latency_series)
        {
            series.samples.reserve(k_default_latency_reserve);
        }

        cache_history.reserve(k_default_history_reserve);
        cache_total_hit_ratios.reserve(k_default_history_reserve);
        cache_exact_hit_ratios.reserve(k_default_history_reserve);
        approx_added_counts.reserve(k_default_history_reserve);
        approx_representative_counts.reserve(k_default_history_reserve);
        global_thresh_history.reserve(k_default_history_reserve);

        clear();
    }

    void ANNSCacheStats::clear() noexcept
    {
        /* Reset counters and clear all series buffers.
         */
        cache_hit = 0;
        cache_miss = 0;
        cache_invalid_detect = 0;
        cache_evict = 0;
        cache_sim_hit = 0;
        cache_dropout = 0;

        for (LatencySeries& series : latency_series)
        {
            series.samples.clear();
        }

        cache_total_hit_ratios.clear();
        cache_exact_hit_ratios.clear();
        approx_added_counts.clear();
        approx_representative_counts.clear();
        global_thresh_history.clear();
        cache_history.clear();
        history_sequence = 0;
    }

    void ANNSCacheStats::recordHitHistory() noexcept
    {
        /* Store hit ratio history without timestamps.
         */
        const float denom = static_cast<float>(cache_hit + cache_sim_hit + cache_miss);
        const float exact_ratio = (denom == 0.0f) ? 0.0f : (static_cast<float>(cache_hit) / denom);
        const float total_ratio = (denom == 0.0f) ? 0.0f : (static_cast<float>(cache_hit + cache_sim_hit) / denom);

        cache_exact_hit_ratios.push_back(exact_ratio);
        cache_total_hit_ratios.push_back(total_ratio);
    }

    void ANNSCacheStats::recordCacheHistorySample(
        const char* op_name,
        double latency_ms,
        size_t cache_entry_count,
        size_t vector_pool_size,
        size_t approx_repr_count) noexcept
    {
        /* Append a concise per-request snapshot.
         */
        std::lock_guard<SpinMutex> stats_guard(stats_lock);

        CacheHistorySample sample;
        sample.sequence = history_sequence++;
        sample.op_name = op_name;
        sample.latency_ms = latency_ms;

        sample.cache_entry_count = cache_entry_count;
        sample.vector_pool_size = vector_pool_size;
        sample.approx_repr_count = approx_repr_count;

        sample.cache_hit = cache_hit;
        sample.cache_sim_hit = cache_sim_hit;
        sample.cache_miss = cache_miss;
        sample.cache_evict = cache_evict;

        const float denom = static_cast<float>(cache_hit + cache_sim_hit + cache_miss);
        sample.exact_hit_ratio = (denom == 0.0f) ? 0.0f : (static_cast<float>(cache_hit) / denom);
        sample.total_hit_ratio = (denom == 0.0f) ? 0.0f : (static_cast<float>(cache_hit + cache_sim_hit) / denom);

        cache_history.push_back(sample);
    }

    void ANNSCacheStats::appendLatencySample(LatencyMetric metric, const ElapsedLatencyPair& sample) noexcept
    {
        /* Append a latency sample with minimal key overhead.
         */
        const std::size_t idx = static_cast<std::size_t>(metric);
        if (idx >= latency_series.size())
        {
            return;
        }

        std::lock_guard<SpinMutex> stats_guard(stats_lock);
        latency_series[idx].samples.push_back(sample);
    }

    const char* ANNSCacheStats::getLatencyKey(LatencyMetric metric) const noexcept
    {
        const std::size_t idx = static_cast<std::size_t>(metric);
        if (idx >= latency_series.size())
        {
            return "unknown";
        }
        return latency_series[idx].key;
    }

    std::string ANNSCacheStats::makeTraceDirectoryPath() noexcept
    {
        /* Build a timestamped directory under /tmp.
         */
        static std::atomic<std::uint64_t> export_generation{0};

        const auto now = std::chrono::system_clock::now();
        const std::time_t t = std::chrono::system_clock::to_time_t(now);

        std::tm tm;
        localtime_r(&t, &tm);

        const std::uint64_t generation = export_generation.fetch_add(1, std::memory_order_relaxed);
        const long pid = static_cast<long>(getpid());

        std::ostringstream oss;
        oss << k_trace_directory_prefix << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
        oss << "_pid" << pid << "_gen" << generation;
        oss << "/";
        return oss.str();
    }

    bool ANNSCacheStats::ensureDirectoryExists(const std::string& path) noexcept
    {
        /* Create a single directory. Parent directories are assumed to exist.
         */
        if (path.empty())
        {
            return false;
        }

        const int rc = mkdir(path.c_str(), 0777);
        if (rc == 0)
        {
            return true;
        }

        // If it already exists, treat as success.
        return errno == EEXIST;
    }

    void ANNSCacheStats::exportLatencySeries(const std::string& directory_path) noexcept
    {
        /* Write one CSV per key plus an aggregated summary.
         */
        std::ofstream summary_file(directory_path + std::string(k_latency_summary_file), std::ios::out);
        if (summary_file.is_open())
        {
            summary_file << "key,count,avg_ms,min_ms,p50_ms,p99_ms,max_ms\n";
        }

        for (LatencySeries& series : latency_series)
        {
            const std::string file_name = directory_path + std::string(k_trace_file_prefix) + series.key + ".csv";
            std::ofstream file(file_name, std::ios::out);
            if (!file.is_open())
            {
                continue;
            }

            file << "elapsed_ms,aux1,aux2\n";

            std::vector<double> elapsed_ms;
            elapsed_ms.reserve(series.samples.size());

            for (ElapsedLatencyPair& sample : series.samples)
            {
                sample.elapsedMs();

                file << std::fixed << std::setprecision(3) << sample.getElapsedMs();
                file << "," << sample.getAux1() << "," << sample.getAux2() << "\n";

                elapsed_ms.push_back(sample.getElapsedMs());
            }

            file.close();

            if (!summary_file.is_open() || elapsed_ms.empty())
            {
                continue;
            }

            std::sort(elapsed_ms.begin(), elapsed_ms.end());

            const double avg_ms = std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) /
                                  static_cast<double>(elapsed_ms.size());
            const double min_ms = elapsed_ms.front();
            const double max_ms = elapsed_ms.back();
            const double p50_ms = getPercentile(elapsed_ms, 0.50);
            const double p99_ms = getPercentile(elapsed_ms, 0.99);

            summary_file << series.key << "," << elapsed_ms.size() << ",";
            summary_file << std::fixed << std::setprecision(3);
            summary_file << avg_ms << "," << min_ms << "," << p50_ms << "," << p99_ms << "," << max_ms << "\n";
        }

        if (summary_file.is_open())
        {
            summary_file.close();
        }
    }

    void ANNSCacheStats::exportCacheHistory(const std::string& directory_path) noexcept
    {
        /* Export a concise per-request cache history.
         */
        const std::string file_name = directory_path + std::string(k_cache_history_file);
        std::ofstream file(file_name, std::ios::out);
        if (!file.is_open())
        {
            return;
        }

        file << "sequence,op,latency_ms,cache_entries,pool_size,approx_repr,hit,sim_hit,miss,evict,exact_hit_ratio,total_hit_ratio\n";
        for (const CacheHistorySample& sample : cache_history)
        {
            file << sample.sequence << ",";
            file << sample.op_name << ",";
            file << std::fixed << std::setprecision(3) << sample.latency_ms << ",";
            file << sample.cache_entry_count << ",";
            file << sample.vector_pool_size << ",";
            file << sample.approx_repr_count << ",";
            file << sample.cache_hit << ",";
            file << sample.cache_sim_hit << ",";
            file << sample.cache_miss << ",";
            file << sample.cache_evict << ",";
            file << sample.exact_hit_ratio << ",";
            file << sample.total_hit_ratio << "\n";
        }

        file.close();
    }

    void ANNSCacheStats::exportDerivedHistories(const std::string& directory_path) noexcept
    {
        /* Export hit ratio and approx filter histories.
         */
        {
            const std::string file_name = directory_path + std::string(k_hit_ratio_history_file);
            std::ofstream file(file_name, std::ios::out);
            if (file.is_open())
            {
                file << "sequence,exact_hit_ratio,total_hit_ratio\n";
                const std::size_t count = std::min(cache_exact_hit_ratios.size(), cache_total_hit_ratios.size());
                for (std::size_t i = 0; i < count; ++i)
                {
                    file << i << "," << cache_exact_hit_ratios[i] << "," << cache_total_hit_ratios[i] << "\n";
                }
                file.close();
            }
        }

        {
            const std::string file_name = directory_path + std::string(k_approx_filter_history_file);
            std::ofstream file(file_name, std::ios::out);
            if (file.is_open())
            {
                file << "sequence,added_count,representative_count\n";
                const std::size_t count = std::min(approx_added_counts.size(), approx_representative_counts.size());
                for (std::size_t i = 0; i < count; ++i)
                {
                    file << i << "," << approx_added_counts[i] << "," << approx_representative_counts[i] << "\n";
                }
                file.close();
            }
        }
    }

    std::string ANNSCacheStats::exportTraceToFiles() noexcept
    {
        /* Export all telemetry under a timestamped directory.
         */
        std::lock_guard<SpinMutex> stats_guard(stats_lock);

        trace_directory_path_ = makeTraceDirectoryPath();
        ensureDirectoryExists(trace_directory_path_);

        exportLatencySeries(trace_directory_path_);
        exportCacheHistory(trace_directory_path_);
        exportDerivedHistories(trace_directory_path_);

        return trace_directory_path_;
    }

    const std::string& ANNSCacheStats::getTraceDirectoryPath() const noexcept
    {
        return trace_directory_path_;
    }

    void ANNSCacheStats::printAll() noexcept
    {
        /* Lightweight log dump for debugging.
         */
        std::lock_guard<SpinMutex> stats_guard(stats_lock);

        AKER_LOG_INFO << "[ANNSCacheStats] cache_hit=" << cache_hit
                      << " cache_sim_hit=" << cache_sim_hit
                      << " cache_miss=" << cache_miss
                      << " cache_evict=" << cache_evict;

        for (const LatencySeries& series : latency_series)
        {
            AKER_LOG_INFO << "[ANNSCacheStats] latency_series(" << series.key
                          << ") count=" << series.samples.size();
        }
    }
}

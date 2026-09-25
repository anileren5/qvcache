#include "core/ak_anns_cache_telemetry.hh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace aker
{
    namespace
    {
        /**
         * @brief Lightweight memory estimate used for telemetry export.
         *
         * This intentionally reports an approximation (not exact RSS) because FAISS and
         * allocator-level behaviors are not exposed in a portable way.
         *
         * The estimate follows the methodology used in the Aker paper evaluation and the
         * accompanying spreadsheet: only the core components are counted.
         *
         * Components counted:
         * - Cache entry metadata (per entry)
         * - Vector pool payload (per pooled vector)
         * - Query filter (HNSW) payload approximation (per representative entry)
         */
        struct MemoryEstimate
        {
            std::uint64_t entry_bytes{0};
            std::uint64_t pool_bytes{0};
            std::uint64_t filter_bytes{0};
            std::uint64_t total_bytes{0};
        };

        /**
         * @brief Returns bytes->MB conversion using decimal MB (1,000,000 bytes).
         */
        double bytesToMb(std::uint64_t bytes) noexcept
        {
            static constexpr double k_mb_divisor = 1000.0 * 1000.0;
            return static_cast<double>(bytes) / k_mb_divisor;
        }

        /**
         * @brief Estimates the memory footprint of core Aker data structures.
         */
        MemoryEstimate estimateCoreMemory(
            size_t total_entry_count,
            size_t physical_entry_count,
            size_t vector_pool_size,
            std::uint32_t dimension) noexcept
        {
            /*
            * NOTE: These constants are intentionally fixed (not derived from sizeof()) to keep
            * results comparable across platforms and match the paper/spreadsheet methodology.
            */
            static constexpr std::uint64_t k_entry_size_bytes = 80;
            static constexpr std::uint64_t k_float_bytes = 4;
            static constexpr std::uint64_t k_pool_meta_bytes = 48;

            /*
            * Query filter estimation (paper-style):
            *   Per-entry bytes = (d * 4) + (m * 8)
            *     - d * 4 : float32 query vector payload
            *     - m * 8 : neighbor identifier list (8-byte identifiers)
            *   Total filter bytes = R * per-entry bytes
            */
            static constexpr std::uint64_t k_neighbor_id_bytes = 8;
            static constexpr std::uint64_t k_query_filter_m = 4;

            const std::uint64_t dim_u64 = static_cast<std::uint64_t>(dimension);

            MemoryEstimate estimate;

            estimate.entry_bytes =
                static_cast<std::uint64_t>(total_entry_count) * k_entry_size_bytes;

            estimate.pool_bytes =
                static_cast<std::uint64_t>(vector_pool_size) *
                (k_pool_meta_bytes + (k_float_bytes * dim_u64));

            estimate.filter_bytes =
                static_cast<std::uint64_t>(physical_entry_count) *
                ((k_float_bytes * dim_u64) +
                (k_query_filter_m * k_neighbor_id_bytes));

            estimate.total_bytes =
                estimate.entry_bytes + estimate.pool_bytes + estimate.filter_bytes;

            return estimate;
        }

        /**
         * @brief Cache summary file name.
         */
        static constexpr const char* k_cache_summary_file = "aker_trace_cache_summary.csv";

        /**
         * @brief Cache status file name.
         */
        static constexpr const char* k_cache_status_file = "aker_trace_cache_status.txt";

        /**
         * @brief Cache parameter snapshot file name.
         */
        static constexpr const char* k_cache_parameters_file = "aker_trace_parameters.csv";

        /**
         * @brief Write-log metric snapshot file name.
         */
        static constexpr const char* k_write_log_metrics_file = "aker_trace_write_log.csv";
    }

    ANNSCacheTelemetry::ANNSCacheTelemetry(ANNSCacheContext* context) noexcept
        : context_(context)
    {
        assert(context_ != nullptr);
    }

    std::string ANNSCacheTelemetry::buildStatusText() noexcept
    {
        /* Build a human-readable cache status string.
         */
        std::ostringstream oss;

        const size_t total_entry_count = context_->lookup_table->map.size();
        size_t virtual_entry_count = 0;
        size_t valid_entry_count = 0;

        for (const auto& pair : context_->lookup_table->map)
        {
            anns_cache_entry_t* entry = pair.second;
            if (entry == nullptr)
                continue;

            if (entry->neighbors_list == nullptr)
                continue;

            if (entry->next != nullptr)
            {
                size_t linked_count = 0;
                anns_cache_entry_t* next_entry = entry->next;
                while (next_entry != nullptr)
                {
                    ++linked_count;
                    next_entry = next_entry->next;
                }
                virtual_entry_count += linked_count;
            }

            if (entry->version != -1)
                ++valid_entry_count;
        }

        const size_t physical_entry_count = (total_entry_count >= virtual_entry_count)
                                               ? (total_entry_count - virtual_entry_count)
                                               : 0;

        const float denom = static_cast<float>(context_->stats.cache_hit + context_->stats.cache_miss + context_->stats.cache_sim_hit);
        const float exact_hit_ratio = (denom == 0.0f) ? 0.0f : (static_cast<float>(context_->stats.cache_hit) / denom);
        const float total_hit_ratio = (denom == 0.0f)
                                          ? 0.0f
                                          : (static_cast<float>(context_->stats.cache_hit + context_->stats.cache_sim_hit) / denom);

        oss << "ANNSCache Status\n";
        oss << "  Total entries: " << total_entry_count << "\n";
        oss << "  Physical entries: " << physical_entry_count << "\n";
        oss << "  Virtual entries: " << virtual_entry_count << "\n";
        oss << "  Valid entries: " << valid_entry_count << "\n";
        oss << "\n";

        oss << "  Vector pool: " << context_->vector_pool->getStatusText() << "\n";
        oss << "  Write log: " << context_->write_log->getStatusText() << "\n";
        oss << "\n";

        oss << "  Cache hit: " << context_->stats.cache_hit << "\n";
        oss << "  Cache sim-hit: " << context_->stats.cache_sim_hit << "\n";
        oss << "  Cache miss: " << context_->stats.cache_miss << "\n";
        oss << "  Cache evict: " << context_->stats.cache_evict << "\n";
        oss << "  Cache invalid-detect: " << context_->stats.cache_invalid_detect << "\n";
        oss << "  Cache dropout: " << context_->stats.cache_dropout << "\n";
        oss << "  Exact hit ratio: " << exact_hit_ratio << "\n";
        oss << "  Total hit ratio: " << total_hit_ratio << "\n";
        oss << "\n";

        oss << "  Eviction queue size: " << context_->eviction_strategy->getCurrSize() << "\n";
        oss << "  Approx repr vectors: " << context_->apprx_filter->getRepresentativeVectorNumber() << "\n";
        oss << "  Approx added vectors: " << context_->apprx_filter->getAddedCounts() << "\n";

        /*
         * Export an approximate memory footprint of core Aker structures.
         * See buildSummaryCsv() for the same values in machine-readable form.
         */
        const MemoryEstimate mem = estimateCoreMemory(
            total_entry_count,
            physical_entry_count,
            context_->vector_pool->getSize(),
            context_->parameter.vector_format.dimension);

        oss << "\n";
        oss << "  Memory estimate (core-only, decimal MB)\n";
        oss << "    Entries: " << std::fixed << std::setprecision(3) << bytesToMb(mem.entry_bytes) << " MB\n";
        oss << "    Pool: " << std::fixed << std::setprecision(3) << bytesToMb(mem.pool_bytes) << " MB\n";
        oss << "    Query filter: " << std::fixed << std::setprecision(3) << bytesToMb(mem.filter_bytes) << " MB\n";
        oss << "    Total: " << std::fixed << std::setprecision(3) << bytesToMb(mem.total_bytes) << " MB\n";

        return oss.str();
    }

    std::string ANNSCacheTelemetry::buildSummaryCsv() noexcept
    {
        /* Build a concise CSV key/value snapshot.
         */
        std::ostringstream oss;
        oss << "metric,value\n";

        const size_t total_entry_count = context_->lookup_table->map.size();
        size_t virtual_entry_count = 0;
        size_t valid_entry_count = 0;

        for (const auto& pair : context_->lookup_table->map)
        {
            anns_cache_entry_t* entry = pair.second;
            if (entry == nullptr)
                continue;

            if (entry->neighbors_list == nullptr)
                continue;

            if (entry->next != nullptr)
            {
                size_t linked_count = 0;
                anns_cache_entry_t* next_entry = entry->next;
                while (next_entry != nullptr)
                {
                    ++linked_count;
                    next_entry = next_entry->next;
                }
                virtual_entry_count += linked_count;
            }

            if (entry->version != -1)
                ++valid_entry_count;
        }

        const size_t physical_entry_count = (total_entry_count >= virtual_entry_count)
                                               ? (total_entry_count - virtual_entry_count)
                                               : 0;

        size_t checkpoint_count = 0;
        for (const auto& pair : context_->lookup_table->map)
        {
            anns_cache_entry_t* entry = pair.second;
            if (entry == nullptr)
                continue;
            if (entry->checkpoint != nullptr)
                ++checkpoint_count;
        }

        oss << "TotalEntryCount," << total_entry_count << "\n";
        oss << "PhysicalEntryCount," << physical_entry_count << "\n";
        oss << "VirtualEntryCount," << virtual_entry_count << "\n";
        oss << "ValidEntryCount," << valid_entry_count << "\n";
        oss << "CheckpointCount," << checkpoint_count << "\n";

        oss << "VectorPoolSize," << context_->vector_pool->getSize() << "\n";
        oss << "VectorPoolCapacity," << context_->vector_pool->getCapacity() << "\n";
        oss << "VectorPoolShards," << context_->vector_pool->getShardCount() << "\n";

        /* Export a few write-log health metrics.
         */
        const WriteLogMetrics write_log_metrics = context_->write_log->getMetrics();
        oss << "WriteLogEntryCount," << write_log_metrics.log_entry_count << "\n";
        oss << "WriteLogCurrentRisk," << write_log_metrics.current_risk << "\n";
        oss << "WriteLogTotalUnseen," << write_log_metrics.total_unseen << "\n";
        oss << "WriteLogRefreshCount," << write_log_metrics.refresh_count << "\n";

        oss << "CacheHit," << context_->stats.cache_hit << "\n";
        oss << "CacheSimHit," << context_->stats.cache_sim_hit << "\n";
        oss << "CacheMiss," << context_->stats.cache_miss << "\n";
        oss << "CacheEvict," << context_->stats.cache_evict << "\n";
        oss << "CacheInvalidDetect," << context_->stats.cache_invalid_detect << "\n";
        oss << "CacheDropout," << context_->stats.cache_dropout << "\n";

        const float denom = static_cast<float>(context_->stats.cache_hit + context_->stats.cache_miss + context_->stats.cache_sim_hit);
        const float exact_hit_ratio = (denom == 0.0f) ? 0.0f : (static_cast<float>(context_->stats.cache_hit) / denom);
        const float total_hit_ratio = (denom == 0.0f)
                                          ? 0.0f
                                          : (static_cast<float>(context_->stats.cache_hit + context_->stats.cache_sim_hit) / denom);

        oss << "ExactHitRatio," << exact_hit_ratio << "\n";
        oss << "TotalHitRatio," << total_hit_ratio << "\n";

        oss << "EvictionQueueSize," << context_->eviction_strategy->getCurrSize() << "\n";
        oss << "ApproxReprCount," << context_->apprx_filter->getRepresentativeVectorNumber() << "\n";
        oss << "ApproxAddedCount," << context_->apprx_filter->getAddedCounts() << "\n";

        /*
         * Memory estimation.
         *
         * This is an approximation intended for comparison across configurations.
         * It counts only core structures and excludes container overheads (unordered_map
         * buckets, allocator fragmentation, std::vector capacities, etc.).
         */
        const MemoryEstimate mem = estimateCoreMemory(
            total_entry_count,
            physical_entry_count,
            context_->vector_pool->getSize(),
            context_->parameter.vector_format.dimension);

        oss << "EstimatedEntryBytes," << mem.entry_bytes << "\n";
        oss << "EstimatedPoolBytes," << mem.pool_bytes << "\n";
        oss << "EstimatedQueryFilterBytes," << mem.filter_bytes << "\n";
        oss << "EstimatedTotalBytes," << mem.total_bytes << "\n";

        oss << "EstimatedEntryMB," << bytesToMb(mem.entry_bytes) << "\n";
        oss << "EstimatedPoolMB," << bytesToMb(mem.pool_bytes) << "\n";
        oss << "EstimatedQueryFilterMB," << bytesToMb(mem.filter_bytes) << "\n";
        oss << "EstimatedTotalMB," << bytesToMb(mem.total_bytes) << "\n";

        oss << "EstimatedMemoryScope,entry_pool_query_filter_only\n";

        return oss.str();
    }

    void ANNSCacheTelemetry::exportTraceToFiles() noexcept
    {
        /* Export all telemetry under a single /tmp timestamped directory.
         */
        if (!context_->has_activity)
        {
            return;
        }

        const std::string directory_path = context_->stats.exportTraceToFiles();
        if (directory_path.empty())
        {
            return;
        }

        {
            const std::string file_name = directory_path + std::string(k_cache_summary_file);
            std::ofstream file(file_name, std::ios::out);
            if (file.is_open())
            {
                file << buildSummaryCsv();
                file.close();
            }
        }

        {
            const std::string file_name = directory_path + std::string(k_cache_status_file);
            std::ofstream file(file_name, std::ios::out);
            if (file.is_open())
            {
                file << buildStatusText();
                file.close();
            }
        }

        {
            /* Export write-log metrics as a concise CSV snapshot.
             */
            const std::string file_name = directory_path + std::string(k_write_log_metrics_file);
            std::ofstream file(file_name, std::ios::out);
            if (file.is_open())
            {
                file << context_->write_log->buildMetricsCsv();
                file.close();
            }
        }

        {
            /* Export cache parameter snapshot for reproducibility.
             */
            const std::string file_name = directory_path + std::string(k_cache_parameters_file);
            std::ofstream file(file_name, std::ios::out);
            if (file.is_open())
            {
                file << "parameter,value\n";
                file << "dimension," << context_->parameter.vector_format.dimension << "\n";
                file << "pool_size," << context_->parameter.capacity.pool_size << "\n";
                file << "vector_in_bytes," << context_->parameter.vector_format.vector_in_bytes << "\n";
                file << "in_topk," << context_->parameter.capacity.in_topk << "\n";
                file << "top_delta," << context_->parameter.capacity.top_delta << "\n";

                file << "global_thresh," << context_->parameter.tuning.global_thresh << "\n";
                file << "dropout," << context_->parameter.tuning.dropout << "\n";
                file << "risk_thresh," << context_->parameter.tuning.risk_thresh << "\n";
                file << "alpha_tighten," << context_->parameter.tuning.alpha_tighten << "\n";
                file << "alpha_loosen," << context_->parameter.tuning.alpha_loosen << "\n";

                file << "distance_metric," << static_cast<int>(context_->parameter.distance_metric) << "\n";
                file.close();
            }
        }
    }
}

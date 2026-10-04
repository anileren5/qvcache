#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <ctime>

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "ak_logger.hh"

#include "core/ak_cache_vector_copy.hh"
#include "core/ak_anns_cache.hh"
#include "core/ak_anns_cache_context.hh"
#include "core/ak_anns_cache_modules.hh"
#include "utils/ak_stack_alloc.hh"

namespace aker
{
    namespace
    {
        static constexpr size_t k_write_log_scan_thresh = 16;
    }

    ANNSCacheContext::ANNSCacheContext(const anns_cache_config_t& parameter_info) noexcept
        : parameter(parameter_info),
          stats(),
          repr_entry_count(0),
          lookup_table(std::make_unique<anns_cache_table_t>()),
          vector_pool(std::make_unique<VectorSlotPool>(parameter_info.capacity.pool_size, parameter_info.vector_format.vector_in_bytes)),
          eviction_strategy(std::make_unique<EvictionStrategyFifo>()),
          apprx_filter(std::make_unique<ApproxFilterDualHNSW2>(parameter_info)),
          evict_entry_count(0),
          write_log(std::make_unique<RiskAwareWriteLog>(parameter_info.capacity.in_topk, k_write_log_scan_thresh, parameter_info.tuning.risk_thresh)),
          try_read_count(0),
          inst_distance_function(),
          has_distance_function(false),
          has_activity(false),
          has_activity_since_last_export(false)
    {
#if defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)
        /* Potluck Mode: global_thresh always starts from 0.0 regardless of configuration.
         * This matches the legacy Potluck baseline behavior.
         */
        parameter.tuning.global_thresh = 0.0f;
#endif
    }

    ANNSCacheContext::~ANNSCacheContext() noexcept = default;

    ANNSCache::ANNSCache(anns_cache_config_t& parameter_info) noexcept
        : context_(std::make_unique<ANNSCacheContext>(parameter_info))
    {
        /* Initializes the cache with the provided parameters.
         */
        entry_store_ = std::make_unique<ANNSCacheEntryStore>(context_.get());
        similarity_engine_ = std::make_unique<ANNSCacheSimilarityEngine>(context_.get(), entry_store_.get());
        maintenance_ = std::make_unique<ANNSCacheMaintenance>(context_.get(), entry_store_.get(), similarity_engine_.get());
        telemetry_ = std::make_unique<ANNSCacheTelemetry>(context_.get());

        /* Emit a configuration snapshot for debugging.
         */
        AKER_LOG_INFO << "[ANNSCache] parameters";
#if defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)
        AKER_LOG_INFO << "  mode=potluck";
#elif defined(AKER_ENABLE_PROXIMITY_MODE) && (AKER_ENABLE_PROXIMITY_MODE != 0)
        AKER_LOG_INFO << "  mode=proximity";
#else
        AKER_LOG_INFO << "  mode=standard";
#endif
        AKER_LOG_INFO << "  dimension=" << context_->parameter.vector_format.dimension;
        AKER_LOG_INFO << "  pool_size=" << context_->parameter.capacity.pool_size;
        AKER_LOG_INFO << "  vector_in_bytes=" << context_->parameter.vector_format.vector_in_bytes;
        AKER_LOG_INFO << "  in_topk=" << context_->parameter.capacity.in_topk;
        AKER_LOG_INFO << "  top_delta=" << context_->parameter.capacity.top_delta;
        AKER_LOG_INFO << "  global_thresh=" << context_->parameter.tuning.global_thresh;
        AKER_LOG_INFO << "  dropout=" << context_->parameter.tuning.dropout;
        AKER_LOG_INFO << "  risk_thresh=" << context_->parameter.tuning.risk_thresh;
        AKER_LOG_INFO << "  alpha_tighten=" << context_->parameter.tuning.alpha_tighten;
        AKER_LOG_INFO << "  alpha_loosen=" << context_->parameter.tuning.alpha_loosen;

        assert(context_->parameter.capacity.getSlotListSize() ==
               (context_->parameter.capacity.in_topk + context_->parameter.capacity.top_delta));
    }

    ANNSCache::~ANNSCache()
    {
        /* Export a final trace snapshot on destruction.
         */
        if (context_ == nullptr || !context_->has_activity_since_last_export)
        {
            return;
        }

        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);

        telemetry_->exportTraceToFiles();

        const std::string summary_csv = telemetry_->buildSummaryCsv();
        const std::string& trace_directory_path = context_->stats.getTraceDirectoryPath();

        if (!trace_directory_path.empty())
        {
            AKER_LOG_INFO << "[ANNSCache] exported telemetry to " << trace_directory_path;
        }

        AKER_LOG_INFO << "[ANNSCache] final state summary (CSV)";
        AKER_LOG_INFO << "\n" << summary_csv;
    }

    anns_cache_entry_t*
    ANNSCache::createCacheEntry(
        VectorSlot* query_vector,
        std::uint32_t list_size,
        VectorSlot** local_neighbors_list) noexcept
    {
        /* Prepares a cache entry.
         * The returned entry is not inserted until insertCacheEntry() succeeds.
         */
        context_->has_activity = true;

        anns_cache_entry_t* entry = new anns_cache_entry_t();

        const size_t vec_data_size = context_->vector_pool->getPayloadSize();
        entry->query_vector = cloneVectorBasic(query_vector, vec_data_size);

        entry->entry_kind = ANNS_CACHE_ENTRY_KIND_PREPARED;
        entry->entry_status = ANNS_CACHE_ENTRY_STATUS_VALID;
        entry->version = 0;

        entry->neighbors = list_size;

        if (local_neighbors_list != nullptr)
        {
            entry->neighbors_list = static_cast<VectorSlot**>(
                aligned_alloc(k_cache_entry_slot_alignment, sizeof(VectorSlot*) * list_size));
            std::memcpy(entry->neighbors_list, local_neighbors_list, sizeof(VectorSlot*) * list_size);

            entry->min_distance = entry->neighbors_list[0]->getDistance();
            entry->max_distance = entry->neighbors_list[list_size - 1]->getDistance();
        }
        else
        {
            entry->neighbors_list = nullptr;
            entry->min_distance = std::numeric_limits<float>::max();
            entry->max_distance = std::numeric_limits<float>::min();
        }
#if defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)
        // Potluck Mode: global threshold is tuned at put(), but entries keep this initial value for bookkeeping.
        entry->thresh = context_->parameter.tuning.global_thresh;
#elif defined(AKER_ENABLE_PROXIMITY_MODE) && (AKER_ENABLE_PROXIMITY_MODE != 0)
        // Proximity Mode: use a fixed global threshold for all entries.
        assert(context_->parameter.tuning.global_thresh > 0.0f);
        entry->thresh = context_->parameter.tuning.global_thresh;
#else
        // Standard Mode: start from a fraction of the closest distance and adapt over time.
        entry->thresh = entry->min_distance / 4.0f;
#endif

        entry->prev = nullptr;
        entry->next = nullptr;
        entry->checkpoint = nullptr;
        entry->risk_factor = 0.0f;

        return entry;
    }

    void
    ANNSCache::destroyCacheEntry(anns_cache_entry_t* entry) noexcept
    {
        /* Releases an externally held entry.
         * - Always deletes the owned query_vector.
         * - Frees the slot pointer array.
         * - Deletes slot VectorSlot objects only for RETURNED_COPY entries.
         */
        context_->has_activity = true;

        if (entry == nullptr)
            return;

        if (entry->entry_kind == ANNS_CACHE_ENTRY_KIND_RETURNED_COPY)
        {
            if (entry->neighbors_list != nullptr)
            {
                for (size_t i = 0; i < entry->neighbors; i++)
                    delete entry->neighbors_list[i];
            }
        }

        delete entry->query_vector;
        entry->query_vector = nullptr;

        free(entry->neighbors_list);
        entry->neighbors_list = nullptr;

        delete entry;
    }

    anns_cache_entry_t*
    ANNSCache::getCacheEntry(
        vector_view_t query_vector_data,
        bool& similar_entry,
        bool& is_invalid,
        distance_function_t distance_function) noexcept
    {
        /* Public API wrapper.
         * Measures end-to-end latency and records a concise history sample.
         */
        ElapsedLatencyPair latency;
        latency.start();

        anns_cache_entry_t* result_entry = nullptr;
        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        /* Convert the query vector before entering the global cache critical section.
         * This reduces the lock hold time without changing cache semantics.
         */
        StackFloatBuffer query_vector_float(static_cast<size_t>(query_vector_data.dimension));
        bool convert_success = query_vector_data.transform_callback(
            query_vector_data.vector_data,
            query_vector_data.vector_in_bytes,
            query_vector_data.dimension,
            query_vector_float.data(),
            query_vector_data.aux);
        (void)convert_success;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
            context_->has_activity = true;
            context_->has_activity_since_last_export = true;

            /* Potluck tuning requires a distance function instance at put().
             * We capture the first seen distance function to keep the insert path deterministic.
             */
            if (!context_->has_distance_function)
            {
                context_->inst_distance_function = distance_function;
                context_->has_distance_function = true;
            }

            result_entry = similarity_engine_->getCacheEntry(
                query_vector_data, similar_entry, is_invalid, distance_function, query_vector_float.data());

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_get_cache_entry, latency);

        context_->stats.recordCacheHistorySample("getCacheEntry", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);

        return result_entry;
    }

    bool
    ANNSCache::insertCacheEntry(
        vector_id_t vector_id,
        anns_cache_entry_t* entry,
        vector_view_t query_vector_data) noexcept
    {
        ElapsedLatencyPair latency;
        latency.start();

        bool inserted = false;
        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
            context_->has_activity = true;
            context_->has_activity_since_last_export = true;

            inserted = maintenance_->insertCacheEntry(vector_id, entry, query_vector_data);

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_insert_cache_entry, latency);

        context_->stats.recordCacheHistorySample("insertCacheEntry", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);

        return inserted;
    }

    bool
    ANNSCache::linkCacheEntry(anns_cache_entry_t* allocated_entry, vector_id_t found_id) noexcept
    {
        ElapsedLatencyPair latency;
        latency.start();

        bool linked = false;
        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
            context_->has_activity = true;
            context_->has_activity_since_last_export = true;

#if defined(AKER_STANDARD_MODE) && (AKER_STANDARD_MODE != 0)
            linked = entry_store_->linkCacheEntry(allocated_entry, found_id);
#else
            /* Proximity/Potluck Mode: linked hits are disabled, so this API is a no-op.
             * We still consume the allocated dummy entry to preserve ownership semantics
             * (callers should not free entries after a successful link call).
             */
            destroyCacheEntry(allocated_entry);
            (void)found_id;
            linked = true;
#endif

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_link_cache_entry, latency);

        context_->stats.recordCacheHistorySample("linkCacheEntry", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);

        return linked;
    }

    void
    ANNSCache::markVectorDeleted(vector_id_t vector_id) noexcept
    {
        ElapsedLatencyPair latency;
        latency.start();

        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
            context_->has_activity = true;
            context_->has_activity_since_last_export = true;

            maintenance_->markVectorDeleted(vector_id);

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_mark_vector_deleted, latency);

        context_->stats.recordCacheHistorySample("markVectorDeleted", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);
    }

    void
    ANNSCache::insertWriteLogEntry(
        vector_view_t write_vector,
        distance_function_t distance_function,
        result_transform_callback_t result_transform_callback) noexcept
    {
        ElapsedLatencyPair latency;
        latency.start();

#if !defined(AKER_STANDARD_MODE) || (AKER_STANDARD_MODE == 0)
        /* Proximity/Potluck Mode: write-log is disabled. */
        (void)write_vector;
        (void)distance_function;
        (void)result_transform_callback;
        return;
#endif

        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        /* Convert the write vector before entering the global cache critical section.
         * This is safe because conversion uses only the caller-provided vector buffer.
         */
        const size_t dimension = context_->parameter.vector_format.dimension;
        StackFloatBuffer write_vector_float(dimension);
        bool convert_success = write_vector.transform_callback(
            write_vector.vector_data,
            write_vector.vector_in_bytes,
            dimension,
            write_vector_float.data(),
            write_vector.aux);
        (void)convert_success;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
            context_->has_activity = true;
            context_->has_activity_since_last_export = true;

            /* Potluck tuning requires a distance function instance at put().
             * We capture the first seen distance function to keep the insert path deterministic.
             */
            if (!context_->has_distance_function)
            {
                context_->inst_distance_function = distance_function;
                context_->has_distance_function = true;
            }

            maintenance_->insertWriteLogEntry(
                write_vector,
                distance_function,
                result_transform_callback,
                write_vector_float.data());

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_insert_write_log_entry, latency);

        context_->stats.recordCacheHistorySample("insertWriteLogEntry", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);
    }

    void
    ANNSCache::processWriteLogEntries(
        distance_function_t distance_function,
        result_transform_callback_t result_transform_callback) noexcept
    {
        ElapsedLatencyPair latency;
        latency.start();

#if !defined(AKER_STANDARD_MODE) || (AKER_STANDARD_MODE == 0)
        /* Proximity/Potluck Mode: write-log is disabled. */
        (void)distance_function;
        (void)result_transform_callback;
        return;
#endif

        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
            context_->has_activity = true;
            context_->has_activity_since_last_export = true;

            /* Potluck tuning requires a distance function instance at put().
             * We capture the first seen distance function to keep the insert path deterministic.
             */
            if (!context_->has_distance_function)
            {
                context_->inst_distance_function = distance_function;
                context_->has_distance_function = true;
            }

            maintenance_->processWriteLogEntries(distance_function, result_transform_callback);

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_process_write_log_entries, latency);

        context_->stats.recordCacheHistorySample("processWriteLogEntries", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);
    }

    void
    ANNSCache::resetCache() noexcept
    {
        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        context_->has_activity = true;
        context_->has_activity_since_last_export = true;
        maintenance_->resetCache();
    }

    void
    ANNSCache::stressTestInvalidateRandom(float percent) noexcept
    {
        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        context_->has_activity = true;
        context_->has_activity_since_last_export = true;
        maintenance_->stressTestInvalidateRandom(percent);
    }

    void
    ANNSCache::collectPooledVectors(std::vector<VectorSlot*>& pooled_list) noexcept
    {
        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        maintenance_->collectPooledVectors(pooled_list);
    }

    std::string
    ANNSCache::getStatusText() noexcept
    {
        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        return telemetry_->buildStatusText();
    }

    std::string
    ANNSCache::getSummaryCsv() noexcept
    {
        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        return telemetry_->buildSummaryCsv();
    }

    void
    ANNSCache::exportTraceToFiles() noexcept
    {
        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        telemetry_->exportTraceToFiles();
    }
}

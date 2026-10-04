#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "core/ak_anns_cache_maintenance.hh"
#include "core/ak_anns_cache_entry_store.hh"
#include "core/ak_anns_cache_similarity_engine.hh"
#include "ak_logger.hh"
#include "utils/ak_malloc_ptr.hh"
#include "utils/ak_stack_alloc.hh"

namespace aker
{
    /* Helper: Frees a cache entry object itself (query vector + slot pointer array).
     * Slot VectorSlot elements are intentionally not deleted here.
     */
    static inline void
    destroyCacheEntryObject(anns_cache_entry_t* entry) noexcept
    {
        if (entry == nullptr)
            return;

        delete entry->query_vector;
        entry->query_vector = nullptr;

        free(entry->neighbors_list);
        entry->neighbors_list = nullptr;

        delete entry;
    }

    ANNSCacheMaintenance::ANNSCacheMaintenance(ANNSCacheContext* context,
        ANNSCacheEntryStore* entry_store,
        ANNSCacheSimilarityEngine* similarity_engine) noexcept
        : context_(context),
          entry_store_(entry_store),
          similarity_engine_(similarity_engine)
    {
        assert(context_ != nullptr);
        assert(entry_store_ != nullptr);
        assert(similarity_engine_ != nullptr);
    }

    bool
    ANNSCacheMaintenance::needEvict(size_t vector_list_size) noexcept
    {
        /* Conservative eviction decision based on pool size and incoming list size.
         */
        size_t current_pool_size = context_->vector_pool->getSize();
        return (current_pool_size + vector_list_size) > context_->parameter.capacity.pool_size;
    }

    void
    ANNSCacheMaintenance::evictCacheEntries(size_t to_evicts, std::vector<vector_id_t>& evicted_list) noexcept
    {
        /* Evicts entries using the configured eviction strategy.
         *
         * IMPORTANT: The eviction target (to_evicts) is expressed in the number of vectors
         * physically removed from the vector pool (i.e., refcount reaches zero and the slot
         * is deleted). It is NOT the number of cache entries to evict. This preserves the
         * original eviction semantics where shared vectors may require evicting additional
         * cache entries until enough pool vectors are actually freed.
         */
        if (to_evicts == 0)
            return;

        size_t eviction_size = 0;

        /* Emit eviction request for debugging.
         */
        AKER_LOG_DEBUG << "[ANNSCacheMaintenance] eviction requested: to_evicts=" << to_evicts;

        while (eviction_size < to_evicts)
        {
            vector_id_t evict_candidate_id = context_->eviction_strategy->nextEvictCandidate();
            if (evict_candidate_id == 0)
                break;

            anns_cache_entry_t* evict_candidate_entry = entry_store_->getCacheEntryFromStorage(evict_candidate_id);
            if (evict_candidate_entry == nullptr)
                continue;

            AKER_LOG_DEBUG << "[ANNSCacheMaintenance] evicting entry: vector_id=" << evict_candidate_id
                          << " neighbors=" << evict_candidate_entry->neighbors;

            /* Release all pooled vectors referenced by this entry.
             * We count only actual deletions (refcount reaches zero) toward the eviction target.
             */
            for (size_t j = 0; j < evict_candidate_entry->neighbors; j++)
            {
                VectorSlot* vector = evict_candidate_entry->neighbors_list[j];
                if (vector == nullptr)
                    continue;

                bool deleted = context_->vector_pool->releaseVectorReference(vector->getVectorId());
                if (deleted)
                    eviction_size++;
            }

            /* Remove linked child entries.
             */
            anns_cache_entry_t* child_entry = evict_candidate_entry->next;
            while (child_entry != nullptr)
            {
                vector_id_t child_vector_id = child_entry->query_vector->getVectorId();
                anns_cache_entry_t* next = child_entry->next;

                context_->lookup_table->map.erase(child_vector_id);
                destroyCacheEntryObject(child_entry);

                child_entry = next;
            }

            /* Remove the root entry from the lookup table. */
            context_->lookup_table->map.erase(evict_candidate_id);

#if defined(AKER_STANDARD_MODE) && (AKER_STANDARD_MODE != 0)
            /* Update write-log statistics based on checkpoint distance.
             */
            epoch_t unseen = 0;
            if (evict_candidate_entry->checkpoint != nullptr)
            {
                unseen = context_->write_log->getUnseenDistance(evict_candidate_entry->checkpoint);
                context_->write_log->releaseCheckpoint(evict_candidate_entry->checkpoint);
                evict_candidate_entry->checkpoint = nullptr;
            }

            context_->write_log->removeCacheEntryFromRoundRobin(static_cast<void*>(evict_candidate_entry));

            context_->repr_entry_count = context_->eviction_strategy->getCurrSize();
            context_->write_log->removeCacheEntryRisk(evict_candidate_entry->risk_factor, unseen, context_->repr_entry_count);
#else
            /* Proximity/Potluck Mode: write-log is disabled. */
            context_->repr_entry_count = context_->eviction_strategy->getCurrSize();
#endif

            context_->evict_entry_count++;
            destroyCacheEntryObject(evict_candidate_entry);

            evicted_list.push_back(evict_candidate_id);
        }

        context_->stats.cache_evict += eviction_size;
#if defined(AKER_STANDARD_MODE) && (AKER_STANDARD_MODE != 0)
        context_->write_log->trimUnreferencedHeadEntries();
#endif

        if (!evicted_list.empty())
        {
            AKER_LOG_INFO << "[ANNSCacheMaintenance] evicted entries: count=" << evicted_list.size();
        }
    }

    bool
    ANNSCacheMaintenance::insertCacheEntry(
        vector_id_t vector_id,
        anns_cache_entry_t* entry,
        vector_view_t query_vector_data) noexcept
    {
        /* Inserts a prepared entry into the cache.
         * This preserves the original insertion semantics.
         */
        /* Step-level latency is recorded as separate series.
         */

        ElapsedLatencyPair latency_int_1;
        ElapsedLatencyPair latency_int_2;
        ElapsedLatencyPair latency_int_3;

        anns_cache_entry_t* allocated_entry = entry;

        const auto emplace_result = context_->lookup_table->map.emplace(vector_id, entry);
        const bool inserted = emplace_result.second;
        if (!inserted)
        {
#if defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)
            /* Potluck Mode: on duplicate insertion, refresh the existing entry's vectors.
             * This preserves the legacy Potluck behavior where put() for the same query id
             * substitutes the stored top-k results.
             */
            anns_cache_entry_t* existing_entry = emplace_result.first->second;
            if (existing_entry != nullptr && existing_entry->neighbors_list != nullptr && entry->neighbors_list != nullptr)
            {
                const size_t refresh_size = std::min(static_cast<size_t>(existing_entry->neighbors), static_cast<size_t>(entry->neighbors));
                for (size_t i = 0; i < refresh_size; i++)
                {
                    VectorSlot* new_slot = entry->neighbors_list[i];
                    if (new_slot == nullptr)
                        continue;

                    vector_id_t new_vector_id = new_slot->getVectorId();
                    const vector_data_t* new_vector_data = new_slot->getVectorData();
                    float new_distance = new_slot->getDistance();

                    VectorSlot* old_slot = existing_entry->neighbors_list[i];
                    VectorSlot* refreshed_slot = nullptr;

                    if (old_slot != nullptr)
                    {
                        refreshed_slot = context_->vector_pool->replaceVectorReference(
                            old_slot->getVectorId(),
                            new_vector_id,
                            new_vector_data);
                    }
                    else
                    {
                        refreshed_slot = context_->vector_pool->acquireOrCreateVector(new_vector_id, new_vector_data);
                    }

                    existing_entry->neighbors_list[i] = refreshed_slot;

                    if (refreshed_slot != nullptr)
                    {
                        float old_distance = refreshed_slot->getDistance();
                        if (old_distance > new_distance)
                            refreshed_slot->setDistance(new_distance);
                    }
                }

                existing_entry->min_distance = entry->min_distance;
                existing_entry->max_distance = entry->max_distance;

                float dist_topk = entry->neighbors_list[context_->parameter.capacity.in_topk - 1]->getDistance();
                float dist_max = entry->max_distance;

                if (dist_max == 0.0f)
                {
                    dist_max = std::numeric_limits<float>::max();
                    dist_topk = std::numeric_limits<float>::max();
                }

                existing_entry->risk_factor = dist_topk / dist_max;

                AKER_LOG_DEBUG << "[ANNSCacheMaintenance] duplicate cache entry refreshed: query_id=" << vector_id;
            }

            destroyCacheEntryObject(entry);
            return true;
#else
            return false;
#endif
        }

        /* Optional eviction path.
         */
        if (needEvict(allocated_entry->neighbors))
        {
            latency_int_1.start();

            std::vector<vector_id_t> evicted_list;
            evictCacheEntries(allocated_entry->neighbors, evicted_list);

            if (!evicted_list.empty())
            {
                context_->apprx_filter->deleteVectors(evicted_list);
                AKER_LOG_DEBUG << "[ANNSCacheMaintenance] approx filter deleted vectors: count=" << evicted_list.size();
                if (context_->apprx_filter->needSwitch())
                {
                    AKER_LOG_INFO << "[ANNSCacheMaintenance] approx filter generation rotate triggered";
                    context_->apprx_filter->rotateGeneration();
                }
            }

            latency_int_1.end();
            latency_int_1.setAux1(evicted_list.size());
            context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_insert_cache_entry_step_1, latency_int_1);
        }

        /* Setup metadata for the inserted entry.
         */
#if defined(AKER_STANDARD_MODE) && (AKER_STANDARD_MODE != 0)
        allocated_entry->checkpoint = context_->write_log->acquireTailCheckpoint();
#else
        allocated_entry->checkpoint = nullptr;
#endif

        allocated_entry->prev = nullptr;
        allocated_entry->next = nullptr;
        allocated_entry->version = 0;

        /* Register representative vector in the approximate filter.
         */
        latency_int_2.start();
        context_->apprx_filter->addVector(query_vector_data);
        AKER_LOG_DEBUG << "[ANNSCacheMaintenance] approx filter added representative vector: query_id="
                      << query_vector_data.vector_id;
        latency_int_2.end();
        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_insert_cache_entry_step_2, latency_int_2);

        /* Transfer vector slots into the pool.
         */
        latency_int_3.start();

        float dist_topk = allocated_entry->neighbors_list[context_->parameter.capacity.in_topk - 1]->getDistance();
        float dist_max = allocated_entry->max_distance;

        for (size_t i = 0; i < allocated_entry->neighbors; i++)
        {
            vector_id_t result_vector_id = allocated_entry->neighbors_list[i]->getVectorId();
            vector_data_t* result_vector_data = allocated_entry->neighbors_list[i]->getVectorData();
            float dist = allocated_entry->neighbors_list[i]->getDistance();

            VectorSlot* pooled_vector = context_->vector_pool->acquireOrCreateVector(result_vector_id, result_vector_data);
            allocated_entry->neighbors_list[i] = pooled_vector;

            float old_dist = pooled_vector->getDistance();
            if (old_dist > dist)
                pooled_vector->setDistance(dist);
        }

        allocated_entry->entry_kind = ANNS_CACHE_ENTRY_KIND_INTERNAL;

        assert(allocated_entry->neighbors >= context_->parameter.capacity.in_topk);
        assert(dist_max >= dist_topk);

        if (dist_max == 0.0f)
        {
            dist_max = std::numeric_limits<float>::max();
            dist_topk = std::numeric_limits<float>::max();
        }

        allocated_entry->risk_factor = dist_topk / dist_max;

#if defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)
        tuneGlobalThresholdAtPut(allocated_entry, query_vector_data);
#endif

        context_->eviction_strategy->addEvictCandidate(vector_id);
        context_->repr_entry_count = context_->eviction_strategy->getCurrSize();

        AKER_LOG_DEBUG << "[ANNSCacheMaintenance] inserted cache entry finalized: vector_id=" << vector_id
                      << " risk_factor=" << allocated_entry->risk_factor
                      << " repr_entry_count=" << context_->repr_entry_count;

#if defined(AKER_STANDARD_MODE) && (AKER_STANDARD_MODE != 0)
        context_->write_log->addCacheEntryToRoundRobin(static_cast<void*>(allocated_entry));
        context_->write_log->addCacheEntryRisk(allocated_entry->risk_factor, 0, context_->repr_entry_count);
#endif

        latency_int_3.end();
        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_insert_cache_entry_step_3, latency_int_3);

        context_->stats.approx_added_counts.push_back(context_->apprx_filter->getAddedCounts());
        context_->stats.approx_representative_counts.push_back(context_->apprx_filter->getRepresentativeVectorNumber());

        return true;
    }

    void
    ANNSCacheMaintenance::processWriteLogEntries(
        distance_function_t distance_function,
        result_transform_callback_t result_transform_callback) noexcept
    {
#if !defined(AKER_STANDARD_MODE) || (AKER_STANDARD_MODE == 0)
        (void)distance_function;
        (void)result_transform_callback;
        return;
#endif

        AKER_LOG_DEBUG << "[ANNSCacheMaintenance] Processing write log entries (if needed)";

        /* Triggers the slow-path write-log scan if the read-to-repr ratio is high enough.
         */
        static constexpr double k_try_read_ratio_thresh = 0.25;

        if (context_->try_read_count > (k_try_read_ratio_thresh * context_->repr_entry_count))
        {
            if (context_->write_log->shouldRunSlowPath())
                runWriteLogSlowPath(distance_function, result_transform_callback);
            context_->try_read_count = 0;
        }
    }

    void
    ANNSCacheMaintenance::updateWriteLogFastPath(
        vector_view_t write_vector,
        distance_function_t distance_function,
        result_transform_callback_t result_transform_callback,
        const float* write_vector_float) noexcept
    {
#if !defined(AKER_STANDARD_MODE) || (AKER_STANDARD_MODE == 0)
        (void)write_vector;
        (void)distance_function;
        (void)result_transform_callback;
        (void)write_vector_float;
        return;
#endif

        /* Fast-path write-log update that opportunistically improves a nearby entry.
         */
        static constexpr faiss::idx_t k_search_per_filter = 1;
        static constexpr size_t k_candidate_count = 2;

        std::array<float, k_candidate_count> distances{};
        std::array<faiss::idx_t, k_candidate_count> labels{};

        context_->apprx_filter->searchSimilarVectors(
            write_vector_float,
            k_search_per_filter,
            distances.data(),
            labels.data());

        for (size_t i = 0; i < static_cast<size_t>(k_search_per_filter); i++)
        {
            if (distances[i] == k_invalid_distance)
                continue;

            if (labels[i] < 0)
                continue;

            vector_id_t vector_id = static_cast<vector_id_t>(labels[i]);
            anns_cache_entry_t* found_entry = entry_store_->getCacheEntryFromStorage(vector_id);
            if (found_entry == nullptr)
                continue;

            float query_distance = distance_function(
                write_vector.vector_data,
                found_entry->query_vector->getVectorData(),
                write_vector.dimension);

            if (query_distance < found_entry->max_distance)
            {
                for (int j = static_cast<int>(found_entry->neighbors - 1); j >= 0; j--)
                {
                    if (!found_entry->neighbors_list[j]->isValid())
                        continue;

                    vector_id_t delete_vector_id = found_entry->neighbors_list[j]->getVectorId();
                    float old_distance = found_entry->neighbors_list[j]->getDistance();

                    vector_id_t write_vector_id = write_vector.vector_id;
                    vector_data_t* write_vector_data = write_vector.vector_data;

                    if (old_distance > query_distance)
                    {
                        VectorSlot* sub_vec = context_->vector_pool->replaceVectorReference(
                            delete_vector_id,
                            write_vector_id,
                            write_vector_data);

                        assert(sub_vec != nullptr);

                        std::uint64_t aux_data_1 = found_entry->query_vector->getAuxData1();
                        std::uint64_t aux_data_2 = found_entry->query_vector->getAuxData2();

                        if (result_transform_callback != nullptr)
                            result_transform_callback(
                                write_vector_id,
                                sub_vec->getVectorData(),
                                context_->vector_pool->getPayloadSize(),
                                aux_data_1,
                                aux_data_2);

                        found_entry->neighbors_list[j] = sub_vec;

                        if (j == static_cast<int>(found_entry->neighbors - 1))
                        {
                            found_entry->max_distance = query_distance;
                            found_entry->neighbors_list[j]->setDistance(query_distance);
                        }

                        break;
                    }
                }
            }
        }
    }

    
    void
    ANNSCacheMaintenance::tuneGlobalThresholdAtPut(
        anns_cache_entry_t* allocated_entry,
        vector_view_t query_vector_data) noexcept
    {
        /* Potluck global threshold tuning at put().
         *
         * This logic is intentionally kept as close as possible to the original Potluck baseline:
         * - It runs only after enough representative entries exist.
         * - It finds the nearest representative entry.
         * - It updates the global threshold based on (distance, top-k set equality).
         */
        static constexpr size_t k_potluck_min_cache_entries = 100;
        static constexpr float k_potluck_loosen_weight = 0.8f;
        static constexpr faiss::idx_t k_search_per_filter = 1;
        static constexpr size_t k_candidate_count = 2;

        if (context_->lookup_table->map.size() <= k_potluck_min_cache_entries)
            return;

        if (!context_->has_distance_function || !static_cast<bool>(context_->inst_distance_function))
        {
            AKER_LOG_WARN << "[ANNSCacheMaintenance] potluck threshold tuning skipped: distance function not set";
            return;
        }

        distance_function_t local_distance_function = context_->inst_distance_function;

        /* Convert query vector to float for approximate filter search.
         */
        StackFloatBuffer query_vector_float(static_cast<size_t>(query_vector_data.dimension));
        bool convert_success = query_vector_data.transform_callback(
            query_vector_data.vector_data,
            query_vector_data.vector_in_bytes,
            query_vector_data.dimension,
            query_vector_float.data(),
            query_vector_data.aux);
        (void)convert_success;

        std::array<float, k_candidate_count> distances{};
        std::array<faiss::idx_t, k_candidate_count> labels{};

        context_->apprx_filter->searchSimilarVectors(
            query_vector_float.data(),
            k_search_per_filter,
            distances.data(),
            labels.data());

        for (size_t i = 0; i < k_candidate_count; i++)
        {
            if (distances[i] == k_invalid_distance)
                continue;

            if (labels[i] < 0)
                continue;

            vector_id_t vector_id = static_cast<vector_id_t>(labels[i]);
            anns_cache_entry_t* found_entry = entry_store_->getCacheEntryFromStorage(vector_id);
            if (found_entry == nullptr)
                continue;

            bool is_valid = similarity_engine_->validateCacheEntry(found_entry);
            if (!is_valid)
                continue;

            float similarity_threshold = context_->parameter.tuning.global_thresh;

            float query_distance = local_distance_function(
                query_vector_data.vector_data,
                found_entry->query_vector->getVectorData(),
                query_vector_data.dimension);

            /* Check the equality of the top-k lists.
             *
             * NOTE: This intentionally reproduces the original Potluck baseline behavior:
             * - It checks only existence.
             */
            bool equal = false;
            for (size_t j = 0; j < context_->parameter.capacity.in_topk; j++)
            {
                vector_id_t topk_vector_id = allocated_entry->neighbors_list[j]->getVectorId();
                for (size_t k = 0; k < context_->parameter.capacity.in_topk; k++)
                {
                    if (topk_vector_id == found_entry->neighbors_list[k]->getVectorId())
                    {
                        equal = true;
                        break;
                    }
                }
                if (equal == false)
                    break;
            }

            if (query_distance < similarity_threshold)
            {
                if (equal == false)
                    context_->parameter.tuning.global_thresh = context_->parameter.tuning.global_thresh * context_->parameter.tuning.alpha_tighten;
            }
            else
            {
                if (equal == true)
                    context_->parameter.tuning.global_thresh = (query_distance * k_potluck_loosen_weight)
                        + (context_->parameter.tuning.global_thresh * (1.0f - k_potluck_loosen_weight));
            }
        }

        context_->stats.global_thresh_history.push_back(context_->parameter.tuning.global_thresh);
    }


    void
    ANNSCacheMaintenance::runWriteLogSlowPath(
        distance_function_t distance_function,
        result_transform_callback_t result_transform_callback) noexcept
    {
#if !defined(AKER_STANDARD_MODE) || (AKER_STANDARD_MODE == 0)
        (void)distance_function;
        (void)result_transform_callback;
        return;
#endif

        /* Slow-path write-log update.
         * This preserves the snapshot implementation (sweep + topk refresh).
         */
        /* Step-level latency is recorded as separate series.
         */

        ElapsedLatencyPair latency_int_1;
        ElapsedLatencyPair latency_int_2;

        cache_entry_handle_t rr_handle = context_->write_log->getNextCacheEntryFromRoundRobin();
        anns_cache_entry_t* saved_entry = static_cast<anns_cache_entry_t*>(rr_handle);
        if (saved_entry == nullptr)
            return;

        vector_id_t vector_id = saved_entry->query_vector->getVectorId();

        const bool found = (context_->lookup_table->map.find(vector_id) != context_->lookup_table->map.end());
        assert(found == true);

        WriteLogScanResult scan_result;

        latency_int_1.start();
        scan_result = context_->write_log->scanLogWindow(
            saved_entry->query_vector->getVectorData(),
            context_->parameter.vector_format.dimension,
            saved_entry->max_distance,
            distance_function,
            saved_entry->checkpoint);

        std::sort(
            scan_result.candidates.begin(),
            scan_result.candidates.end(),
            [](const WriteLogCandidate& a, const WriteLogCandidate& b)
            {
                return a.distance > b.distance;
            });

        latency_int_1.end();
        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_write_log_slow_path_step_1, latency_int_1);

        latency_int_2.start();

        if (!scan_result.candidates.empty())
        {
            /* Merge refreshed candidates into the top-k list in-place.
             *
             * This follows the original snapshot logic: iterate the current top-k
             * list (sorted close-to-far) and opportunistically substitute a
             * candidate when it is closer than the current element.
             */
            const size_t vector_in_bytes = context_->vector_pool->getPayloadSize();

            for (int k = 0; k < static_cast<int>(context_->parameter.capacity.in_topk); k++)
            {
                float current_distance = saved_entry->neighbors_list[k]->getDistance();
                if (!scan_result.candidates.empty())
                {
                    WriteLogCandidate& current_candidate = scan_result.candidates.back();
                    if (current_distance > current_candidate.distance)
                    {
                        vector_id_t delete_vector_id = saved_entry->neighbors_list[k]->getVectorId();

                        vector_id_t found_vector_id = current_candidate.vector_id;
                        const vector_data_t* found_vector_data = current_candidate.vector_data;

                        std::uint64_t aux_data_1 = current_candidate.aux_data_1;
                        std::uint64_t aux_data_2 = current_candidate.aux_data_2;

                        VectorSlot* sub_vec = context_->vector_pool->replaceVectorReference(
                            delete_vector_id,
                            found_vector_id,
                            found_vector_data);
                        sub_vec->setDistance(current_candidate.distance);

                        assert(sub_vec != nullptr);

                        if (result_transform_callback != nullptr)
                            result_transform_callback(
                                found_vector_id,
                                sub_vec->getVectorData(),
                                vector_in_bytes,
                                aux_data_1,
                                aux_data_2);

                        saved_entry->neighbors_list[k] = sub_vec;

                        context_->write_log->recordRefresh();
                        scan_result.candidates.pop_back();
                    }
                }
            }

            /* Keep max_distance consistent with the last element of the stored list.
             * When neighbors == in_topk, the last element may be
             * substituted above.
             */
            if (saved_entry->neighbors != 0 && saved_entry->neighbors_list != nullptr)
            {
                saved_entry->max_distance =
                    saved_entry->neighbors_list[saved_entry->neighbors - 1]->getDistance();
            }
        }

        latency_int_2.end();
        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_write_log_slow_path_step_2, latency_int_2);

        context_->write_log->replaceCheckpoint(saved_entry->checkpoint, scan_result.new_checkpoint);
        context_->write_log->consumeUnseenDistance(scan_result.advanced_epoch_distance, context_->repr_entry_count);
    }

    void
    ANNSCacheMaintenance::insertWriteLogEntry(
        vector_view_t write_vector,
        distance_function_t distance_function,
        result_transform_callback_t result_transform_callback,
        const float* write_vector_float) noexcept
    {
#if !defined(AKER_STANDARD_MODE) || (AKER_STANDARD_MODE == 0)
        (void)write_vector;
        (void)distance_function;
        (void)result_transform_callback;
        (void)write_vector_float;
        return;
#endif

        /* Write-log insertion is staged with latency sub-counters.
         */
        /* Step-level latency is recorded as separate series.
         */

        AKER_LOG_DEBUG << "[ANNSCacheMaintenance] write-log insert begin: vector_id=" << write_vector.vector_id;

        ElapsedLatencyPair latency_int_1;
        ElapsedLatencyPair latency_int_2;
        ElapsedLatencyPair latency_int_3;
        ElapsedLatencyPair latency_int_4;

        latency_int_1.start();

        /*
         * Legacy-compatible risk model update:
         * - Refresh repr_entry_count at the moment of the write-log insertion.
         * - Each write-log insertion increases the aggregate unseen distance by
         *   the current number of representative cache entries.
         */
        context_->repr_entry_count = context_->eviction_strategy->getCurrSize();

        context_->write_log->insertLogEntry(
            write_vector.vector_id,
            write_vector.vector_data,
            static_cast<size_t>(write_vector.vector_in_bytes),
            write_vector.aux_data_1,
            write_vector.aux_data_2);

        if (context_->repr_entry_count > 0)
        {
            context_->write_log->addCacheEntryRisk(
                0.0,
                static_cast<epoch_t>(context_->repr_entry_count),
                context_->repr_entry_count);
        }
        latency_int_1.end();
        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_insert_write_log_entry_step_1, latency_int_1);

        latency_int_2.start();
        updateWriteLogFastPath(write_vector, distance_function, result_transform_callback, write_vector_float);
        latency_int_2.end();
        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_insert_write_log_entry_step_2, latency_int_2);

        latency_int_3.start();
        /*
         * Legacy-compatible slow-path trigger for write-log insertion:
         * The slow-path is evaluated immediately after the fast-path without
         * read-count gating.
         */
        if (context_->write_log->shouldRunSlowPath())
            runWriteLogSlowPath(distance_function, result_transform_callback);
        latency_int_3.end();
        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_insert_write_log_entry_step_3, latency_int_3);

        latency_int_4.start();
        /* Legacy behavior: trimming runs during eviction only. */
        latency_int_4.end();
        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_insert_write_log_entry_step_4, latency_int_4);

        AKER_LOG_DEBUG << "[ANNSCacheMaintenance] write-log insert end";
    }

    void
    ANNSCacheMaintenance::markVectorDeleted(vector_id_t vector_id) noexcept
    {
        AKER_LOG_INFO << "[ANNSCacheMaintenance] markVectorDeleted: vector_id=" << vector_id;
        context_->vector_pool->invalidateVector(vector_id);
    }

    void
    ANNSCacheMaintenance::collectPooledVectors(std::vector<VectorSlot*>& pooled_list) noexcept
    {
        context_->vector_pool->collectPooledVectors(pooled_list);
    }

    void
    ANNSCacheMaintenance::resetCache() noexcept
    {
        /* Reset write-log and eviction structures first to avoid stale pointers.
         */
        AKER_LOG_INFO << "[ANNSCacheMaintenance] resetCache";
        context_->write_log->clear();

        context_->eviction_strategy = std::make_unique<EvictionStrategyFifo>();
        context_->repr_entry_count = 0;
        context_->evict_entry_count = 0;
        context_->try_read_count = 0;

        /* Clear cache storage and representative filter.
         */
        context_->lookup_table->map.clear();
        context_->vector_pool->clear();
        context_->apprx_filter->rotateGeneration();
    }

    void
    ANNSCacheMaintenance::stressTestInvalidateRandom(float percent) noexcept
    {
        if (percent < 0.0f || percent > 100.0f)
            return;

        size_t pool_size = context_->vector_pool->getSize();
        size_t target_invalidate = static_cast<size_t>(pool_size * (percent));

        std::vector<VectorSlot*> pooled_list;
        context_->vector_pool->collectPooledVectors(pooled_list);

        std::random_shuffle(pooled_list.begin(), pooled_list.end());

        for (auto& vec : pooled_list)
        {
            if (target_invalidate == 0)
                break;

            context_->vector_pool->invalidateVector(vec->getVectorId());
            target_invalidate--;
        }

        context_->stats.cache_hit = 0;
        context_->stats.cache_miss = 0;
        context_->stats.cache_invalid_detect = 0;
        context_->stats.cache_evict = 0;
        context_->stats.cache_sim_hit = 0;
    }
}
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <vector>
#include <random>

#include "core/ak_anns_cache_similarity_engine.hh"
#include "core/ak_anns_cache_entry_store.hh"
#include "ak_logger.hh"
#include "utils/ak_stack_alloc.hh"

namespace aker
{
    namespace
    {
        static int getRandomInt(int min_value, int max_exclusive) noexcept
        {
            static thread_local std::mt19937 rng(std::random_device{}());
            /* Match the original Potluck baseline behavior:
             * generate an integer in the half-open interval [min, max).
             */
            std::uniform_int_distribution<int> dist(min_value, max_exclusive - 1);
            return dist(rng);
        }

#if defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)
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

        /* Helper: Purges an invalid cache entry immediately.
         * This is used only in Potluck mode to avoid accumulating invalid representatives.
         */
        static inline void
        purgeCacheEntryPotluck(
            ANNSCacheContext* context,
            anns_cache_entry_t* entry) noexcept
        {
            if (context == nullptr || entry == nullptr)
                return;

            /* Resolve to the root entry defensively. */
            anns_cache_entry_t* root_entry = entry;
            while (root_entry != nullptr && root_entry->prev != nullptr)
                root_entry = root_entry->prev;

            if (root_entry == nullptr || root_entry->query_vector == nullptr)
                return;

            const vector_id_t root_id = root_entry->query_vector->getVectorId();

            /* Remove the representative from the approximate filter first.
             * The filter uses tombstones so repeated deletes are safe.
             */
            context->apprx_filter->deleteVector(root_id);

            /* Release pooled vectors referenced by the root entry.
             * (Linked child entries do not own vector lists.)
             */
            if (root_entry->neighbors_list != nullptr)
            {
                for (size_t i = 0; i < static_cast<size_t>(root_entry->neighbors); i++)
                {
                    VectorSlot* slot = root_entry->neighbors_list[i];
                    if (slot == nullptr)
                        continue;

                    context->vector_pool->releaseVectorReference(slot->getVectorId());
                }
            }

            /* Remove linked child entries (if any). */
            anns_cache_entry_t* child_entry = root_entry->next;
            while (child_entry != nullptr)
            {
                const vector_id_t child_id = child_entry->query_vector->getVectorId();
                anns_cache_entry_t* next = child_entry->next;

                context->lookup_table->map.erase(child_id);
                destroyCacheEntryObject(child_entry);

                child_entry = next;
            }

            /* Remove the root entry from storage and free it. */
            context->lookup_table->map.erase(root_id);
            destroyCacheEntryObject(root_entry);

            /* Potluck does not rely on eviction_strategy size for correctness,
             * but we keep repr_entry_count consistent for telemetry.
             */
            context->repr_entry_count = context->lookup_table->map.size();

            AKER_LOG_DEBUG << "[ANNSCacheSimilarity] potluck purged invalid entry: root_id=" << root_id;
        }
#endif
    }

    ANNSCacheSimilarityEngine::ANNSCacheSimilarityEngine(
        ANNSCacheContext* context,
        ANNSCacheEntryStore* entry_store) noexcept
        : context_(context),
          entry_store_(entry_store)
    {
        assert(context_ != nullptr);
        assert(entry_store_ != nullptr);
    }

    bool
    ANNSCacheSimilarityEngine::validateCacheEntry(anns_cache_entry_t* entry) noexcept
    {
        /* Validates an entry by pushing invalid vectors to the tail.
         * If the number of valid vectors falls below intopk, the entry becomes invalid.
         */
        assert(entry != nullptr);

        if (entry->version < 0)
        {
            context_->stats.cache_miss++;
            context_->stats.cache_invalid_detect++;

#if defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)
            /* Potluck Mode: purge invalid entries immediately so they do not linger as
             * tombstoned representatives.
             */
            purgeCacheEntryPotluck(context_, entry);
#endif
            return false;
        }

        size_t valid_count = 0;
        size_t invalid_count = 0;

        for (int i = 0; i < entry->neighbors; i++)
        {
            VectorSlot* slot = entry->neighbors_list[i];
            if (slot == nullptr || !slot->isValid())
                invalid_count++;
            else
                valid_count++;
        }

        bool is_valid = true;
        if (valid_count < context_->parameter.capacity.in_topk)
        {
            entry->version = -1;
            context_->stats.cache_miss++;
            context_->stats.cache_invalid_detect++;

            context_->apprx_filter->deleteVector(entry->query_vector->getVectorId());

            is_valid = false;

#if defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)
            /* Potluck Mode: purge immediately instead of keeping an invalid entry in storage.
             */
            purgeCacheEntryPotluck(context_, entry);
            return false;
#endif
        }

        if (invalid_count > 0)
        {
            /* Stable reordering: valid vectors keep their relative order,
             * and invalid ones are moved to the tail in their original order.
             */
            reorder_scratch_.clear();
            reorder_scratch_.reserve(static_cast<size_t>(entry->neighbors));

            for (int i = 0; i < entry->neighbors; i++)
            {
                VectorSlot* slot = entry->neighbors_list[i];
                if (slot != nullptr && slot->isValid())
                    reorder_scratch_.push_back(slot);
            }

            for (int i = 0; i < entry->neighbors; i++)
            {
                VectorSlot* slot = entry->neighbors_list[i];
                if (slot == nullptr || !slot->isValid())
                    reorder_scratch_.push_back(slot);
            }

            assert(reorder_scratch_.size() == static_cast<size_t>(entry->neighbors));

            for (int i = 0; i < entry->neighbors; i++)
                entry->neighbors_list[i] = reorder_scratch_[static_cast<size_t>(i)];
        }

        return is_valid;
    }

    anns_cache_entry_t*
    ANNSCacheSimilarityEngine::getCacheEntry(
        vector_view_t query_vector_data,
        bool& similar_entry,
        bool& is_invalid,
        distance_function_t distance_function,
        const float* query_vector_float) noexcept
    {
        /* Performs exact-hit first and then sim-hit using the approximate filter.
         * The returned entry is always a deep copy (caller-owned) on hit.
         */
        context_->try_read_count++;

        similar_entry = false;
        is_invalid = false;

        anns_cache_entry_t* entry = entry_store_->getCacheEntryFromStorage(query_vector_data.vector_id);

        if (entry != nullptr)
        {
            bool is_valid = validateCacheEntry(entry);
            if (!is_valid)
            {
                is_invalid = true;
                entry = nullptr;

                AKER_LOG_DEBUG << "[ANNSCacheSimilarity] exact hit invalidated: query_id=" << query_vector_data.vector_id;
            }
            else
            {
                bool linked_hit = (query_vector_data.vector_id != entry->query_vector->getVectorId());

                if (linked_hit)
                    context_->stats.cache_sim_hit++;
                else
                    context_->stats.cache_hit++;

                context_->stats.recordHitHistory();
                context_->eviction_strategy->recentlyAccessed(entry->query_vector->getVectorId());
#if ((defined(AKER_ENABLE_PROXIMITY_MODE) && (AKER_ENABLE_PROXIMITY_MODE != 0)) ||      (defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)))
                // Proximity/Potluck Mode: keep the threshold fixed (no adaptive update).
#else
                if (!linked_hit)
                {
                    entry->thresh = entry->thresh * context_->parameter.tuning.alpha_loosen;
                    if (entry->thresh > entry->min_distance)
                        entry->thresh = entry->min_distance;
                }
#endif

                AKER_LOG_DEBUG << "[ANNSCacheSimilarity] exact hit: query_id=" << query_vector_data.vector_id
                              << " root_id=" << entry->query_vector->getVectorId()
                              << " linked_hit=" << static_cast<int>(linked_hit);

                anns_cache_entry_t* copy_entry = entry_store_->copyCacheEntry(entry);
                return copy_entry;
            }
        }

        /* Approximate filter lookup.
         */
        static constexpr faiss::idx_t k_search_per_filter = 1;
        static constexpr size_t k_candidate_count = 2;

        /* Convert the query vector outside the critical section when possible.
         * The caller may provide a pre-converted float array.
         */
        const float* faiss_query_vector = query_vector_float;
        if (faiss_query_vector == nullptr)
        {
            /* Defensive fallback: some internal call sites may omit the pre-converted float buffer.
             * The conversion buffer is stack-allocated and valid for this function's lifetime.
             */
            StackFloatBuffer owned_query_vector(static_cast<size_t>(query_vector_data.dimension));

            bool convert_success = query_vector_data.transform_callback(
                query_vector_data.vector_data,
                query_vector_data.vector_in_bytes,
                query_vector_data.dimension,
                owned_query_vector.data(),
                query_vector_data.aux);
            (void)convert_success;

            faiss_query_vector = owned_query_vector.data();
        }

        std::array<float, k_candidate_count> distances{};
        std::array<faiss::idx_t, k_candidate_count> labels{};

        context_->apprx_filter->searchSimilarVectors(
            faiss_query_vector,
            k_search_per_filter,
            distances.data(),
            labels.data());

        bool last_candidate_invalid = false;
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

            bool is_valid = validateCacheEntry(found_entry);
            if (!is_valid)
            {
                last_candidate_invalid = true;
                continue;
            }

            float thresh = 0.0f;
#if defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)
            thresh = context_->parameter.tuning.global_thresh;
#elif defined(AKER_ENABLE_PROXIMITY_MODE) && (AKER_ENABLE_PROXIMITY_MODE != 0)
            thresh = context_->parameter.tuning.global_thresh;
#else
            thresh = std::min(found_entry->min_distance, found_entry->thresh);
            found_entry->thresh = thresh;
#endif

            float query_distance = distance_function(
                query_vector_data.vector_data,
                found_entry->query_vector->getVectorData(),
                query_vector_data.dimension);

            if (query_distance < thresh)
            {
                AKER_LOG_DEBUG << "[ANNSCacheSimilarity] approx hit: query_id=" << query_vector_data.vector_id
                              << " repr_id=" << found_entry->query_vector->getVectorId()
                              << " query_distance=" << query_distance
                              << " thresh=" << thresh;

                similar_entry = true;

#if defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)
                /* Potluck Mode: apply random dropout to force revalidation.
                 */
                context_->eviction_strategy->recentlyAccessed(found_entry->query_vector->getVectorId());

                static constexpr int k_dropout_rand_min = 0;
                static constexpr int k_dropout_rand_max_exclusive = 100;
                const int r = getRandomInt(k_dropout_rand_min, k_dropout_rand_max_exclusive);

                if (r < context_->parameter.tuning.dropout)
                {
                    /* Mark and purge the entry so that a dropped representative is removed
                     * from both the storage table and the approximate filter.
                     */
                    found_entry->version = -10;

                    context_->stats.cache_dropout++;
                    context_->stats.cache_miss++;
                    context_->stats.recordHitHistory();

                    /* Potluck Mode: immediately purge dropped entries so that
                     * invalid representatives do not remain in storage.
                     */
                    purgeCacheEntryPotluck(context_, found_entry);

                    context_->stats.approx_added_counts.push_back(context_->apprx_filter->getAddedCounts());
                    context_->stats.approx_representative_counts.push_back(context_->apprx_filter->getRepresentativeVectorNumber());

                    return nullptr;
                }

                context_->stats.cache_sim_hit++;
                context_->stats.recordHitHistory();

                context_->stats.approx_added_counts.push_back(context_->apprx_filter->getAddedCounts());
                context_->stats.approx_representative_counts.push_back(context_->apprx_filter->getRepresentativeVectorNumber());

                return entry_store_->copyCacheEntry(found_entry);

#elif defined(AKER_ENABLE_PROXIMITY_MODE) && (AKER_ENABLE_PROXIMITY_MODE != 0)
                /* Proximity Mode: keep the threshold fixed (no adaptive update).
                 */
                context_->stats.cache_sim_hit++;
                context_->stats.recordHitHistory();

                context_->eviction_strategy->recentlyAccessed(found_entry->query_vector->getVectorId());

                context_->stats.approx_added_counts.push_back(context_->apprx_filter->getAddedCounts());
                context_->stats.approx_representative_counts.push_back(context_->apprx_filter->getRepresentativeVectorNumber());

                return entry_store_->copyCacheEntry(found_entry);
#else
                context_->stats.cache_sim_hit++;
                context_->stats.recordHitHistory();

                found_entry->thresh = found_entry->thresh * context_->parameter.tuning.alpha_tighten;
                if (found_entry->thresh > found_entry->min_distance)
                    found_entry->thresh = found_entry->min_distance;

                context_->eviction_strategy->recentlyAccessed(found_entry->query_vector->getVectorId());

                context_->stats.approx_added_counts.push_back(context_->apprx_filter->getAddedCounts());
                context_->stats.approx_representative_counts.push_back(context_->apprx_filter->getRepresentativeVectorNumber());

                return entry_store_->copyCacheEntry(found_entry);
#endif
            }
        }

        /* Miss path.
         */
        is_invalid = last_candidate_invalid;
        context_->stats.cache_miss++;
        context_->stats.recordHitHistory();

        AKER_LOG_DEBUG << "[ANNSCacheSimilarity] cache miss: query_id=" << query_vector_data.vector_id
                      << " last_candidate_invalid=" << static_cast<int>(last_candidate_invalid);

        context_->stats.approx_added_counts.push_back(context_->apprx_filter->getAddedCounts());
        context_->stats.approx_representative_counts.push_back(context_->apprx_filter->getRepresentativeVectorNumber());

        return nullptr;
    }
}

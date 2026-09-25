#pragma once

#include <vector>

#include "ak_approx_filter.hh"
#include "core/ak_anns_cache_context.hh"

namespace aker
{
    class ANNSCacheEntryStore;
    class VectorSlot;

    /**
     * @brief SimilarityEngine module for ANNSCache.
     *
     * This module implements exact-hit and sim-hit logic and invalid handling.
     */
    class ANNSCacheSimilarityEngine
    {
    public:
        /**
         * @brief Constructs the module with the shared cache context.
         */
        ANNSCacheSimilarityEngine(ANNSCacheContext* context, ANNSCacheEntryStore* entry_store) noexcept;

        /**
         * @brief Looks up a cache entry by exact id or by approximate similarity.
         */
        anns_cache_entry_t* getCacheEntry(
            vector_view_t query_vector_data,
            bool& similar_entry,
            bool& is_invalid,
            distance_function_t distance_function,
            const float* query_vector_float) noexcept;

        /**
         * @brief Validates an entry by pushing invalid vectors to the tail.
         */
        bool validateCacheEntry(anns_cache_entry_t* entry) noexcept;

    private:
        ANNSCacheContext*  context_;
        ANNSCacheEntryStore* entry_store_;

        /* Scratch buffer reused under the global cache lock.
         * This avoids per-request heap allocations in validateCacheEntry.
         */
        std::vector<VectorSlot*> reorder_scratch_;
    };
}

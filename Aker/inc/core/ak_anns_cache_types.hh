#pragma once

#include <cstddef>
#include <cstdint>

#include <unordered_map>

#include "ak_anns_cache_config.hh"
#include "ak_vector_slot.hh"
#include "ak_write_log.hh"

namespace aker
{
    /**
     * @brief Cache entry ownership kind.
     *
     * - PREPARED: Entry created by makeCEntry() before insertion.
     * - INTERNAL: Entry stored in the cache (vector slots are pool-managed).
     * - RETURNED_COPY: Deep-copied entry returned to callers (owns slot VectorSlot objects).
     */
    typedef std::uint8_t anns_cache_entry_kind_t;

    enum
    {
        ANNS_CACHE_ENTRY_KIND_PREPARED      = 0,
        ANNS_CACHE_ENTRY_KIND_INTERNAL      = 1,
        ANNS_CACHE_ENTRY_KIND_RETURNED_COPY = 2
    };

    /**
     * @brief Cache entry status for CAS-based entry-level locking.
     */
    typedef std::uint8_t anns_cache_entry_status_t;

    enum
    {
        ANNS_CACHE_ENTRY_STATUS_VALID = 0,
        ANNS_CACHE_ENTRY_STATUS_INMOD
    };

    /**
     * @brief ANNS cache entry.
     *
     * This is a legacy POD style structure that is used across C/C++ boundaries.
     * Refactoring should preserve the layout and semantics of this structure.
     */
    typedef struct ANNSCacheEntry
    {
        anns_cache_entry_status_t       entry_status;
        anns_cache_entry_kind_t         entry_kind;
        std::int32_t                    version;

        VectorSlot*                     query_vector;
        std::uint32_t                   neighbors;
        VectorSlot**                    neighbors_list;

        float                           thresh;
        float                           min_distance;
        float                           max_distance;

        struct ANNSCacheEntry* prev;
        struct ANNSCacheEntry* next;

        float                           risk_factor;
        write_log_checkpoint_t      checkpoint;

    } anns_cache_entry_t;

    /**
     * @brief Lookup table for cache entries.
     */
    typedef struct ANNSCacheTable
    {
        std::unordered_map<vector_id_t, anns_cache_entry_t*> map;

    } anns_cache_table_t;

    /**
     * @brief Alignment for pointer-array allocations used in cache entries.
     */
    inline constexpr std::size_t k_cache_entry_slot_alignment = 8;
}

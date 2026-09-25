#pragma once

#include <cstddef>

#include "core/ak_anns_cache_context.hh"

namespace aker
{
    /**
     * @brief EntryStore module for ANNSCache.
     *
     * This module centralizes lookup table access and per-entry CAS locks.
     */
    class ANNSCacheEntryStore
    {
    public:
        /**
         * @brief Constructs the module with the shared cache context.
         */
        explicit ANNSCacheEntryStore(ANNSCacheContext* context) noexcept;

        /**
         * @brief Resolves the given vector id to the root cache entry from storage.
         *
         * This helper only reads the lookup table and traverses the linked chain
         * to return the root entry (prev == nullptr).
         */
        anns_cache_entry_t* getCacheEntryFromStorage(vector_id_t vector_id) noexcept;

        /**
         * @brief Legacy name for getCacheEntryFromStorage().
         */
        [[deprecated("use getCacheEntryFromStorage()")]]
        anns_cache_entry_t* getCEntry(vector_id_t vector_id) noexcept
        {
            return getCacheEntryFromStorage(vector_id);
        }

        /**
         * @brief Creates a deep copy of a cache entry for external consumption.
         */
        anns_cache_entry_t* copyCacheEntry(anns_cache_entry_t* entry) noexcept;

        /**
         * @brief Attempts to acquire the per-entry CAS lock.
         */
        bool tryLockCacheEntry(anns_cache_entry_t* entry) noexcept;

        /**
         * @brief Legacy name for tryLockCacheEntry().
         */
        [[deprecated("use tryLockCacheEntry()")]]
        bool tryLockCEntry(anns_cache_entry_t* entry) noexcept
        {
            return tryLockCacheEntry(entry);
        }

        /**
         * @brief Releases the per-entry CAS lock.
         */
        bool unlockCacheEntry(anns_cache_entry_t* entry) noexcept;

        /**
         * @brief Legacy name for unlockCacheEntry().
         */
        [[deprecated("use unlockCacheEntry()")]]
        bool unlockCEntry(anns_cache_entry_t* entry) noexcept
        {
            return unlockCacheEntry(entry);
        }

        /**
         * @brief Links a dummy entry to an existing root entry.
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

    private:
        ANNSCacheContext* context_;
    };
}

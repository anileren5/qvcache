#include <cassert>
#include <cstdlib>
#include <cstring>

#include "core/ak_anns_cache_entry_store.hh"
#include "core/ak_cache_vector_copy.hh"

namespace aker
{
    ANNSCacheEntryStore::ANNSCacheEntryStore(ANNSCacheContext* context) noexcept
        : context_(context)
    {
        assert(context_ != nullptr);
    }

    anns_cache_entry_t*
    ANNSCacheEntryStore::getCacheEntryFromStorage(vector_id_t vector_id) noexcept
    {
        /* Resolves the entry ID to the root entry.
         * The root entry is defined as the head of the linked chain (prev == nullptr).
         */
        auto it = context_->lookup_table->map.find(vector_id);
        if (it == context_->lookup_table->map.end())
            return nullptr;

        anns_cache_entry_t* entry = it->second;
        while (entry != nullptr && entry->prev != nullptr)
            entry = entry->prev;

        return entry;
    }

    anns_cache_entry_t*
    ANNSCacheEntryStore::copyCacheEntry(anns_cache_entry_t* entry) noexcept
    {
        /* Creates a deep copy of a cache entry for external consumption.
         * The returned copy owns its query vector and slot VectorSlot objects.
         */
        if (entry == nullptr)
            return nullptr;

        anns_cache_entry_t* new_entry = new anns_cache_entry_t();

        const size_t vec_data_size = context_->vector_pool->getPayloadSize();
        new_entry->query_vector = cloneVectorBasic(entry->query_vector, vec_data_size);

        new_entry->entry_kind = ANNS_CACHE_ENTRY_KIND_RETURNED_COPY;
        new_entry->entry_status = entry->entry_status;
        new_entry->version = entry->version;

        new_entry->neighbors = entry->neighbors;

        if (entry->neighbors_list != nullptr && entry->neighbors > 0)
        {
            new_entry->neighbors_list = static_cast<VectorSlot**>(
                aligned_alloc(k_cache_entry_slot_alignment, sizeof(VectorSlot*) * entry->neighbors));

            for (size_t i = 0; i < entry->neighbors; i++)
                new_entry->neighbors_list[i] = cloneVectorBasic(entry->neighbors_list[i], vec_data_size);
        }
        else
        {
            new_entry->neighbors_list = nullptr;
        }

        new_entry->min_distance = entry->min_distance;
        new_entry->max_distance = entry->max_distance;
        new_entry->thresh = entry->thresh;

        new_entry->prev = nullptr;
        new_entry->next = nullptr;
        new_entry->risk_factor = entry->risk_factor;
        new_entry->checkpoint = nullptr;

        return new_entry;
    }

    bool
    ANNSCacheEntryStore::tryLockCacheEntry(anns_cache_entry_t* entry) noexcept
    {
        /* CAS-based entry lock using entry_status.
         */
        anns_cache_entry_status_t expected_state = ANNS_CACHE_ENTRY_STATUS_VALID;
        anns_cache_entry_status_t desired_state = ANNS_CACHE_ENTRY_STATUS_INMOD;

        return __atomic_compare_exchange_n(
            &(entry->entry_status),
            &expected_state,
            desired_state,
            false,
            __ATOMIC_ACQUIRE,
            __ATOMIC_ACQUIRE);
    }

    bool
    ANNSCacheEntryStore::unlockCacheEntry(anns_cache_entry_t* entry) noexcept
    {
        /* CAS-based entry unlock.
         */
        if (entry == nullptr)
            return false;

        anns_cache_entry_status_t expected_state = ANNS_CACHE_ENTRY_STATUS_INMOD;
        anns_cache_entry_status_t desired_state = ANNS_CACHE_ENTRY_STATUS_VALID;

        return __atomic_compare_exchange_n(
            &(entry->entry_status),
            &expected_state,
            desired_state,
            false,
            __ATOMIC_RELEASE,
            __ATOMIC_RELEASE);
    }

    bool
    ANNSCacheEntryStore::linkCacheEntry(anns_cache_entry_t* allocated_entry, vector_id_t found_id) noexcept
    {
        /* Links an allocated dummy entry to an existing root entry.
         * This preserves the original semantics.
         */
        anns_cache_entry_t* root_entry = getCacheEntryFromStorage(found_id);
        if (root_entry == nullptr)
            return false;

        const vector_id_t new_key = allocated_entry->query_vector->getVectorId();
        const auto emplace_result = context_->lookup_table->map.emplace(new_key, allocated_entry);
        const bool inserted = emplace_result.second;
        if (!inserted)
            return false;

        allocated_entry->prev = root_entry;
        allocated_entry->next = root_entry->next;

        root_entry->next = allocated_entry;

        return true;
    }
}

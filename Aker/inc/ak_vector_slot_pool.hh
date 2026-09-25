// 
// Author: Sukjoon Oh
//
// Refactor note:
// - This pool is now optimized for a global cache lock configuration.
// - Internal sharding and per-shard locks are removed to reduce memory/CPU overhead.
// - The public API and semantics remain the same: reference-counted VectorSlot keyed by vector_id.
//

#ifndef AKER_VECTOR_SLOT_POOL_H
#define AKER_VECTOR_SLOT_POOL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ak_vector_slot.hh"

namespace aker
{
    /**
     * @brief Reference-counted pool of shared VectorSlot objects.
     *
     * In Aker's default configuration, the upper ANNSCache layer holds a global
     * lock that serializes all cache mutations. Under that model, internal pool
     * sharding/locking provides little benefit and adds overhead.
     *
     * This pool assumes upper-layer synchronization (global cache lock).
     */
    class VectorSlotPool
    {
    private:
        /**
         * @brief Map from vector_id to pooled VectorSlot.
         */
        std::unordered_map<vector_id_t, VectorSlot*> vector_map_;

        /**
         * @brief Maximum pool capacity (number of pooled vectors).
         */
        const size_t pool_capacity_;

        /**
         * @brief Payload size in bytes for each pooled vector.
         */
        const size_t payload_size_;

        /**
         * @brief Acquires or creates a vector slot while holding the pool lock.
         */
        VectorSlot* acquireOrCreateVectorUnsafe(
            vector_id_t vector_id,
            const vector_data_t* vector_data) noexcept;

        /**
         * @brief Releases one reference while holding the pool lock.
         */
        bool releaseVectorReferenceUnsafe(
            vector_id_t vector_id,
            VectorSlot** deleted_vector) noexcept;

    public:
        /**
         * @brief Legacy shard count default retained for API compatibility.
         */
        static constexpr size_t kDefaultShardCount = 1;

        /**
         * @brief Constructs a pool.
         */
        VectorSlotPool(
            size_t pool_capacity,
            size_t payload_size,
            size_t /*shard_count*/ = kDefaultShardCount) noexcept;

        /**
         * @brief Destructor releases all managed vectors.
         */
        virtual ~VectorSlotPool() noexcept;

        VectorSlotPool(const VectorSlotPool&) = delete;
        VectorSlotPool& operator=(const VectorSlotPool&) = delete;

        /**
         * @brief Returns an existing pooled vector or creates a new one.
         */
        VectorSlot* acquireOrCreateVector(
            vector_id_t vector_id,
            const vector_data_t* vector_data) noexcept;

        /**
         * @brief Releases one reference and deletes the vector when refcount reaches zero.
         */
        bool releaseVectorReference(vector_id_t vector_id) noexcept;

        /**
         * @brief Replaces a reference to `delete_vector_id` with a reference to `alloc_vector_id`.
         */
        VectorSlot* replaceVectorReference(
            vector_id_t delete_vector_id,
            vector_id_t alloc_vector_id,
            const vector_data_t* alloc_vector_data) noexcept;

        /**
         * @brief Marks a pooled vector invalid.
         */
        bool invalidateVector(vector_id_t vector_id) noexcept;

        /**
         * @brief Returns pool capacity.
         */
        size_t getCapacity() const noexcept;

        /**
         * @brief Returns current number of pooled vectors.
         */
        size_t getSize() const noexcept;

        /**
         * @brief Returns payload size.
         */
        size_t getPayloadSize() const noexcept;

        /**
         * @brief Returns the number of internal shards.
         *
         * This pool uses a single map under the global-cache-lock model.
         */
        size_t getShardCount() const noexcept;

        /**
         * @brief Collects all pooled vectors.
         */
        void collectPooledVectors(std::vector<VectorSlot*>& pooled_list) noexcept;

        /**
         * @brief Clears all pooled vectors.
         */
        void clear() noexcept;

        /**
         * @brief Returns a human-readable status string.
         */
        std::string getStatusText() noexcept;
    };
}

#endif

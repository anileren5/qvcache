#include <cassert>
#include <cstring>
#include <sstream>

#include "ak_logger.hh"
#include "ak_vector_slot_pool.hh"

namespace aker
{
    VectorSlotPool::VectorSlotPool(size_t pool_capacity, size_t payload_size, size_t /*shard_count*/) noexcept
        : vector_map_(),
          pool_capacity_(pool_capacity),
          payload_size_(payload_size)
    {
        /* Reserve upfront to reduce rehashing on hot paths. */
        vector_map_.reserve(pool_capacity_ + 16);
    }

    VectorSlotPool::~VectorSlotPool() noexcept
    {
        /* Keep destructor semantics predictable: release all managed vectors. */
        clear();
    }

    VectorSlot*
    VectorSlotPool::acquireOrCreateVectorUnsafe(
        vector_id_t vector_id,
        const vector_data_t* vector_data) noexcept
    {
        /* Hit path: return existing vector with refcount increment. */
        auto it = vector_map_.find(vector_id);
        if (it != vector_map_.end())
        {
            VectorSlot* target_vector = it->second;
            target_vector->lock();
            target_vector->increaseRefCount();
            target_vector->unlock();
            return target_vector;
        }

        /* Miss path: allocate and publish a new VectorSlot. */
        assert(vector_data != nullptr);

        VectorSlot* new_vector = new VectorSlot(payload_size_);
        new_vector->setVectorId(vector_id);
        std::memcpy(new_vector->getVectorData(), vector_data, payload_size_);

        vector_map_.emplace(vector_id, new_vector);

        AKER_LOG_DEBUG << "[VectorSlotPool] created vector slot: vector_id=" << vector_id
                      << " pool_size=" << vector_map_.size();

        return new_vector;
    }

    bool
    VectorSlotPool::releaseVectorReferenceUnsafe(
        vector_id_t vector_id,
        VectorSlot** deleted_vector) noexcept
    {
        /* Decrease refcount; erase from the map when it reaches zero.
         * The caller deletes *deleted_vector outside the pool lock.
         */
        if (deleted_vector == nullptr)
            return false;

        *deleted_vector = nullptr;

        auto it = vector_map_.find(vector_id);
        if (it == vector_map_.end())
            return false;

        VectorSlot* target_vector = it->second;
        target_vector->lock();
        target_vector->decreaseRefCount();

        if (target_vector->getRefCount() == 0)
        {
            vector_map_.erase(it);
            target_vector->unlock();

            *deleted_vector = target_vector;

            AKER_LOG_DEBUG << "[VectorSlotPool] deleted vector slot: vector_id=" << vector_id
                          << " pool_size=" << vector_map_.size();
            return true;
        }

        target_vector->unlock();
        return false;
    }

    VectorSlot*
    VectorSlotPool::acquireOrCreateVector(vector_id_t vector_id, const vector_data_t* vector_data) noexcept
    {
        return acquireOrCreateVectorUnsafe(vector_id, vector_data);
    }

    bool
    VectorSlotPool::releaseVectorReference(vector_id_t vector_id) noexcept
    {
        VectorSlot* deleted_vector = nullptr;

        bool deleted = releaseVectorReferenceUnsafe(vector_id, &deleted_vector);

        if (deleted && deleted_vector != nullptr)
        {
            deleted_vector->freeVectorData();
            delete deleted_vector;
        }

        return deleted;
    }

    VectorSlot*
    VectorSlotPool::replaceVectorReference(
        vector_id_t delete_vector_id,
        vector_id_t alloc_vector_id,
        const vector_data_t* alloc_vector_data) noexcept
    {
        VectorSlot* deleted_vector = nullptr;
        VectorSlot* allocated_vector = nullptr;

        bool deleted = releaseVectorReferenceUnsafe(delete_vector_id, &deleted_vector);
        allocated_vector = acquireOrCreateVectorUnsafe(alloc_vector_id, alloc_vector_data);

        if (deleted && deleted_vector != nullptr)
        {
            deleted_vector->freeVectorData();
            delete deleted_vector;
        }

        return allocated_vector;
    }

    bool
    VectorSlotPool::invalidateVector(vector_id_t vector_id) noexcept
    {
        auto it = vector_map_.find(vector_id);
        if (it == vector_map_.end())
            return false;

        VectorSlot* target_vector = it->second;
        target_vector->lock();
        target_vector->makeInvalid();
        target_vector->unlock();

        return true;
    }

    size_t
    VectorSlotPool::getCapacity() const noexcept
    {
        return pool_capacity_;
    }

    size_t
    VectorSlotPool::getSize() const noexcept
    {
        return vector_map_.size();
    }

    size_t
    VectorSlotPool::getPayloadSize() const noexcept
    {
        return payload_size_;
    }

    size_t
    VectorSlotPool::getShardCount() const noexcept
    {
        return 1;
    }

    void
    VectorSlotPool::collectPooledVectors(std::vector<VectorSlot*>& pooled_list) noexcept
    {
        /* Collect a snapshot of pooled vectors. */
        pooled_list.clear();
        pooled_list.reserve(vector_map_.size());

        for (const auto& kv : vector_map_)
            pooled_list.push_back(kv.second);
    }

    void
    VectorSlotPool::clear() noexcept
    {
        /* Clear the pool. The upper layer is responsible for synchronization. */
        std::unordered_map<vector_id_t, VectorSlot*> local_map;

        local_map.swap(vector_map_);

        for (auto& kv : local_map)
        {
            VectorSlot* vector = kv.second;
            vector->freeVectorData();
            delete vector;
        }

        local_map.clear();
    }

    std::string
    VectorSlotPool::getStatusText() noexcept
    {
        /* Human-readable status snapshot. */
        size_t observed_elements = 0;
        size_t valid_count = 0;
        size_t pool_size = 0;

        pool_size = vector_map_.size();

        for (const auto& kv : vector_map_)
        {
            ++observed_elements;
            if (kv.second->isValid())
                ++valid_count;
        }

        std::ostringstream oss;
        oss << "VectorSlotPool Status: " << pool_size << " inserted out of total " << pool_capacity_;
        oss << ", observed elements: " << observed_elements << "\n";
        oss << "    Valid vectors: " << valid_count << "/" << observed_elements << "\n";

        return oss.str();
    }
}
